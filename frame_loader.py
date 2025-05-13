from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
import os

def load_frames_to_spark(path):
    spark = SparkSession.builder.appName("ViolenceDetection").getOrCreate()
    df = spark.read.format("image").option("dropInvalid", True).load(path)
    df = df.withColumn("filename", input_file_name())
    return df
