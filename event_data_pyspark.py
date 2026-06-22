# Databricks notebook source
from pyspark.sql.functions import expr, col

events = spark.read.option("header", "true") \
            .csv('/Volumes/tabular/dataexpert/mg_volume/events.csv') \
            .withColumn("event_date", expr("DATE_TRUNC('day', event_time)")) 

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
# MAGIC     event_date DATE
# MAGIC )
# MAGIC PARTITIONED BY (year(event_date));

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
# MAGIC     event_date DATE
# MAGIC )
# MAGIC PARTITIONED BY (year(event_date));
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS tabular.dataexpert.mg_events_unsorted (
# MAGIC     url STRING,
# MAGIC     referrer STRING,
# MAGIC     browser_family STRING,
# MAGIC     os_family STRING,
# MAGIC     device_family STRING,
# MAGIC     host STRING,
# MAGIC     event_time TIMESTAMP,
# MAGIC     event_date DATE
# MAGIC )
# MAGIC PARTITIONED BY (year(event_date));

# COMMAND ----------

start_df = df.repartition(4, col("event_date")) \
    .withColumn("event_time", col("event_time").cast("timestamp")) \
    .withColumn("event_date", col("event_date").cast("date"))
    
first_sort_df = start_df.sortWithinPartitions(col("event_date"), col("host"))

start_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("tabular.dataexpert.mg_events_unsorted")
first_sort_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("tabular.dataexpert.mg_events_sorted")