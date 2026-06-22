# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "1"
# ///
# MAGIC %md
# MAGIC
# MAGIC # Notebook Purpose: Sorting in Spark
# MAGIC
# MAGIC This notebook demonstrates the difference between **global sorting** and **sorting within partitions** in Apache Spark using PySpark DataFrames.
# MAGIC
# MAGIC - **Global Sorting (`sort`)**: Sorts the entire DataFrame across all partitions, producing a fully ordered dataset. This is useful when you need a total order, but it can be expensive for large datasets.
# MAGIC - **Sorting Within Partitions (`sortWithinPartitions`)**: Sorts data only within each partition. The order is guaranteed within each …Add visualizations or sample queries to illustrate how sorting affects data retrieval and ordering. Consider including a section comparing query results from the sorted and unsorted tables to reinforce the concepts.

# COMMAND ----------

from pyspark.sql.functions import expr, col

events = spark.read.option("header", "true") \
            .csv('/Volumes/tabular/dataexpert/mg_volume/events.csv') \
            .withColumn("event_date", expr("DATE_TRUNC('day', event_time)")) \
            .withColumn("event_year", expr("YEAR(event_date)"))

devices = spark.read.option("header","true") \
            .csv('/Volumes/tabular/dataexpert/mg_volume/devices.csv')

df = events.join(devices,on="device_id",how="left")
df = df.withColumnsRenamed({'browser_type': 'browser_family', 'os_type': 'os_family'})

df.show(10)



# COMMAND ----------

sorted = df.repartition(10, col("event_date"))\
    .sortWithinPartitions(col("event_date"), col("host"))\
    .withColumn("event_time", col("event_time").cast("timestamp")) 

sortedTwo = df.repartition(10, col("event_date"))\
    .sort(col("event_date"), col("host"))\
    .withColumn("event_time", col("event_time").cast("timestamp")) 

sorted.show()
sortedTwo.show()

# COMMAND ----------

sorted.explain()

# COMMAND ----------

sortedTwo.explain()

# COMMAND ----------

# DBTITLE 1,Untitled
# MAGIC %%sql
# MAGIC
# MAGIC DROP TABLE IF EXISTS tabular.dataexpert.mg_events;
# MAGIC DROP TABLE IF EXISTS tabular.dataexpert.mg_events_sorted;
# MAGIC DROP TABLE IF EXISTS tabular.dataexpert.mg_events_unsorted;
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS tabular.dataexpert.mg_events (
# MAGIC     url STRING,
# MAGIC     referrer STRING,
# MAGIC     browser_family STRING,
# MAGIC     os_family STRING,
# MAGIC     device_family STRING,
# MAGIC     host STRING,
# MAGIC     event_time TIMESTAMP,
# MAGIC     event_date DATE,
# MAGIC     event_year INT
# MAGIC )
# MAGIC PARTITIONED BY (event_year);

# COMMAND ----------

# MAGIC %%sql
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS tabular.dataexpert.mg_events_sorted (
# MAGIC     url STRING,
# MAGIC     referrer STRING,
# MAGIC     browser_family STRING,
# MAGIC     os_family STRING,
# MAGIC     device_family STRING,
# MAGIC     host STRING,
# MAGIC     event_time TIMESTAMP,
# MAGIC     event_date DATE,
# MAGIC     event_year INT
# MAGIC )
# MAGIC PARTITIONED BY (event_year);
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS tabular.dataexpert.mg_events_unsorted (
# MAGIC     url STRING,
# MAGIC     referrer STRING,
# MAGIC     browser_family STRING,
# MAGIC     os_family STRING,
# MAGIC     device_family STRING,
# MAGIC     host STRING,
# MAGIC     event_time TIMESTAMP,
# MAGIC     event_date DATE,
# MAGIC     event_year INT
# MAGIC )
# MAGIC PARTITIONED BY (event_year);

# COMMAND ----------

start_df = df.repartition(4, col("event_date")) \
    .withColumn("event_time", col("event_time").cast("timestamp")) \
    .withColumn("event_date", col("event_date").cast("date"))
    
first_sort_df = start_df.sortWithinPartitions(col("event_date"), col("host"))

start_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("tabular.dataexpert.mg_events_unsorted")
first_sort_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("tabular.dataexpert.mg_events_sorted")

# COMMAND ----------

# DBTITLE 1,Sorting Strategy Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from pyspark.sql.functions import spark_partition_id, col

def sample_with_partitions(sdf, frac=0.15, seed=42):
    """Sample across all partitions; returns a pandas df with event_date + pid."""
    pdf = (
        sdf.withColumn("pid", spark_partition_id())
           .select("event_date", "pid")
           .sample(fraction=frac, seed=seed)
           .toPandas()
    )
    pdf["event_date"] = pd.to_datetime(pdf["event_date"])
    return pdf

# ── Collect the three strategies ─────────────────────────────────────────────
unsort_pdf = sample_with_partitions(start_df)                               # repartitioned, no sort
local_pdf  = sample_with_partitions(first_sort_df)                          # sortWithinPartitions
global_pdf = sample_with_partitions(                                        # global sort
    start_df.sort(col("event_date"), col("host"))
)

# ── Plot: partition lanes (y = partition, x = event_date) ────────────────────
rng = np.random.default_rng(0)
palette = plt.cm.tab10.colors

configs = [
    (unsort_pdf,  "1. repartition() — No Sort",
     "Dates land in partitions by hash.\nAll partitions cover the same date range.\nNo ordering within lanes."),
    (local_pdf,   "2. sortWithinPartitions",
     "Dates are sorted left → right within\neach lane, but lanes overlap in date range.\nNo global ordering guarantee."),
    (global_pdf,  "3. sort() — Global Sort",
     "A range-shuffle assigns each partition a\nnon-overlapping date slice.\nLanes are collectively in strict order."),
]

all_pids = pd.Series(list(
    set(unsort_pdf["pid"]) | set(local_pdf["pid"]) | set(global_pdf["pid"])
)).sort_values().tolist()

fig, axes = plt.subplots(1, 3, figsize=(19, 4), sharey=True, sharex=True)

for ax, (pdf, title, note) in zip(axes, configs):
    pid_list = pdf["pid"].sort_values().unique()
    for pid in pid_list:
        grp = pdf[pdf["pid"] == pid]
        jitter = rng.uniform(-0.25, 0.25, len(grp))
        ax.scatter(grp["event_date"], pid + jitter,
                   color=palette[pid % 10], s=8, alpha=0.55, zorder=2)

    ax.set_yticks(all_pids)
    ax.set_yticklabels([f"P{p}" for p in all_pids], fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Event Date", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
    ax.axhline(y=ax.get_ylim()[0], color="grey", lw=0.5)  # baseline
    ax.text(0.5, -0.24, note, transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, color="#555", style="italic")

axes[0].set_ylabel("Partition", fontsize=10)

handles = [mpatches.Patch(color=palette[p % 10], label=f"Partition {p}") for p in all_pids]
fig.legend(handles=handles, loc="upper right", fontsize=9,
           title="Partition", framealpha=0.9)
fig.suptitle("Spark Sorting Strategies: Partition Lanes View",
             fontsize=14, fontweight="bold", y=1.03)
fig.subplots_adjust(bottom=0.25)
plt.show()
