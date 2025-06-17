# Databricks notebook source
# MAGIC %pip install ../dist/hotel_reservations-0.1.13-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src")))

import time

import requests
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig
from hotel_reservations.serving.fe_model_serving import FeatureLookupServing

# spark session

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = "alubiss-hotel-reservations-model-serving-fe"

# COMMAND ----------

# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(
    model_name=f"{catalog_name}.{schema_name}.hotel_reservations_model_fe",
    endpoint_name=endpoint_name,
    feature_table_name=f"{catalog_name}.{schema_name}.hotel_reservations_features",
)

# Create the online table for house features
feature_model_server.create_online_table()

# COMMAND ----------

# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint()

# COMMAND ----------

# Create a sample request body
required_columns = [
    "type_of_meal_plan",
    "required_car_parking_space",
    "room_type_reserved",
    "market_segment_type",
    "country",
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "lead_time",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
    "arrival_month",
    "Booking_ID",
    "Client_ID",
]

spark = SparkSession.builder.getOrCreate()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()
sampled_records = train_set[required_columns].sample(n=3, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

logger.info(train_set.dtypes)
logger.info(dataframe_records[0])

# COMMAND ----------


# Call the endpoint with one sample record
def call_endpoint(record):
    """Calls the model serving endpoint with a given input record."""
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------

# Load test
for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2)

# COMMAND ----------
