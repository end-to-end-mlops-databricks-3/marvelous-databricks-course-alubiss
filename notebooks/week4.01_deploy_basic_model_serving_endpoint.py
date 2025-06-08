# Databricks notebook source
# MAGIC %pip install ../dist/hotel_reservations-0.1.5-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src")))

import time

import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig
from hotel_reservations.serving.model_serving import ModelServing

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

# COMMAND ----------

# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.model_basic",
    endpoint_name="alubiss-base-hotel-reservations-model-serving",
)

# COMMAND ----------

# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()


# COMMAND ----------

import pandas as pd

columns = [
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
data = [
    ["Meal Plan 1", 0, "Room_Type 1", "Online", "PL", 2, 1, 2, 1, 26, 0, 0, 0, 161, 0, 10, "INN25630", "ABCDE"],
    ["Meal Plan 1", 0, "Room_Type 1", "Online", "PL", 2, 1, 2, 1, 26, 0, 0, 0, 161, 0, 10, "INN25630", "1sw2221"],
]

df = pd.DataFrame(data, columns=columns)

cols_types = {
    "required_car_parking_space": "int32",
    "no_of_adults": "int32",
    "no_of_children": "int32",
    "no_of_weekend_nights": "int32",
    "no_of_week_nights": "int32",
    "lead_time": "int32",
    "repeated_guest": "int32",
    "no_of_previous_cancellations": "int32",
    "no_of_previous_bookings_not_canceled": "int32",
    "avg_price_per_room": "float32",
    "no_of_special_requests": "int32",
    "arrival_month": "int32",
}

df = df.astype(cols_types)

# Sample 3 records from the training set
sampled_records = df.sample(n=2, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

# Call the endpoint with one sample record

"""
Each dataframe record in the request body should be list of json with columns looking like:

[{'LotFrontage': 78.0,
  'LotArea': 9317,
  'OverallQual': 6,
  'OverallCond': 5,
  'YearBuilt': 2006,
  'Exterior1st': 'VinylSd',
  'Exterior2nd': 'VinylSd',
  'MasVnrType': 'None',
  'Foundation': 'PConc',
  'Heating': 'GasA',
  'CentralAir': 'Y',
  'SaleType': 'WD',
  'SaleCondition': 'Normal'}]
"""


def call_endpoint(record):
    """Calls the model serving endpoint with a given input record."""
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/alubiss-base-hotel-reservations-model-serving/invocations"

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
