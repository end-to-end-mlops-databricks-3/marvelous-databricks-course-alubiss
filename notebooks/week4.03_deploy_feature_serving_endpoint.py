# Databricks notebook source
# MAGIC %pip install ../dist/hotel_reservations-0.1.5-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src")))
import time

import mlflow
import requests
from databricks import feature_engineering
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig
from hotel_reservations.serving.feature_serving import FeatureServing

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

fe = feature_engineering.FeatureEngineeringClient()
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

catalog_name = config.catalog_name
schema_name = config.schema_name
feature_table_name = f"{catalog_name}.{schema_name}.alubiss_hotel_reservations_with_preds"
feature_spec_name = f"{catalog_name}.{schema_name}.alubiss_return_predictions"
endpoint_name = "alubiss-hotel-reservations-feature-serving"

# COMMAND ----------

import os
import sys

import mlflow
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src")))

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.feature_lookup_model import FeatureLookUpModel

spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "abcd12345", "branch": "week2", "job_run_id": "1234"}
tags = Tags(**tags_dict)

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

test_set = spark.table("mlops_dev.olalubic.test_set").limit(10)
X_test = test_set.drop(
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
    "booking_status",
)
X_test = X_test.withColumn("Client_ID", col("Client_ID").cast("string"))
X_test = X_test.filter(X_test.Client_ID.isin(["27633", "95890"]))
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)
predictions = fe_model.load_latest_model_and_predict(X_test)
logger.info(predictions)

# COMMAND ----------

# 27633 90
# 95890 48
preds_df = predictions[
    [
        "Client_ID",
        "repeated_guest",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "avg_price_per_room",
        "no_of_special_requests",
        "prediction",
    ]
]

# COMMAND ----------

fe.create_table(
    name=feature_table_name,
    primary_keys=["Client_ID"],
    df=preds_df,
    description="Hotel Reservation predictions feature table",
)

spark.sql(f"""
          ALTER TABLE {feature_table_name}
          SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)

# Initialize feature store manager
feature_serving = FeatureServing(
    feature_table_name=feature_table_name, feature_spec_name=feature_spec_name, endpoint_name=endpoint_name
)


# COMMAND ----------

# Create online table
feature_serving.create_online_table()

# COMMAND ----------

# Create feature spec
feature_serving.create_feature_spec()

# COMMAND ----------

# Deploy feature serving endpoint
feature_serving.deploy_or_update_serving_endpoint()

# COMMAND ----------

start_time = time.time()
serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
    json={"dataframe_records": [{"Client_ID": "27633"}]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")


# COMMAND ----------

# another way to call the endpoint

response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
    json={"dataframe_split": {"columns": ["Client_ID"], "data": [["27633"]]}},
)
response.text
