# Cleaning of Kaggle Worldometer Covid Data
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, to_date, month, year, round

spark = SparkSession.builder.appName("Hadoop PySpark Project").getOrCreate()
df = spark.read.csv("hdfs://cluster-ngene-m/worldometer_coronavirus_daily_data.csv", header = True, inferSchema = True) 

# Filter the DataFrame to eliminate rows where all numerical columns are zero

cleaned_df = df.filter(
    (col("cumulative_total_cases") != 0) |
    (col("daily_new_cases") != 0) |
    (col("active_cases") != 0) |
    (col("cumulative_total_deaths") != 0) |
    (col("daily_new_deaths") != 0)
)


# Null values are replaced with the mean value of the column to avoid data loss

numeric_columns = ["cumulative_total_cases", "daily_new_cases", "active_cases", "cumulative_total_deaths", "daily_new_deaths"]


cleaned_df = cleaned_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))

cleaned_df = cleaned_df.withColumn("month", month(col("date")))
cleaned_df = cleaned_df.withColumn("year", year(col("date")))


mean_values = {}
for col_name in numeric_columns:
    mean_values[col_name] = cleaned_df.groupBy("country", "year", "month").agg(round(mean(col_name),2).alias(f"{col_name}_mean"))


for col_name in numeric_columns:
    cleaned_df = cleaned_df.join(mean_values[col_name], on=["country", "year", "month"], how="left")

    # Fill null values in the original column with the corresponding group-wise mean
    cleaned_df = cleaned_df.withColumn(
        col_name,
        when(col(col_name).isNull(), col(f"{col_name}_mean")).otherwise(col(col_name))
    ).drop(f"{col_name}_mean")



# Remove negative values
for column in numeric_columns:
    cleaned_df = cleaned_df.withColumn(
        column, 
        when(col(column) < 0, 0).otherwise(col(column))
    )


cleaned_df.show()

cleaned_df.write.csv("/cleaned_data.csv", header=True, mode="overwrite")

spark.stop()
