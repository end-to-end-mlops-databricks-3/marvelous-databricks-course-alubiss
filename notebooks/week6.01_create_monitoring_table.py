# Databricks notebook source
# MAGIC %pip install ../dist/hotel_reservations-1.1.23-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance

# COMMAND ----------

import os
import sys

import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
import numpy as np
from pyspark.sql import SparkSession
from loguru import logger
from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.modeling_pipeline import PocessModeling

spark = SparkSession.builder.getOrCreate()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2", "job_run_id": "1234567890"})

modeling_ppl = PocessModeling(
    config=config, tags=tags, spark=spark, code_paths=["../src/hotel_reservations/models/modeling_pipeline.py"]
)
logger.info("Model initialized.")

# Load data and prepare features
modeling_ppl.load_data()
modeling_ppl.prepare_features()
modeling_ppl.train()
logger.info("Loaded data, prepared features.")

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

features = modeling_ppl.pipeline[:-1].transform(modeling_ppl.X_train)
target = modeling_ppl.y_train.copy()

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(features, target)

features_array = features
feature_names = modeling_ppl.preprocessor.get_feature_names_out()
features_df = pd.DataFrame(features_array, columns=feature_names)

# Identify the most important features
feature_importances = pd.DataFrame({
    'Feature': features_df.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Top 5 important features:")
print(feature_importances.head(5))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Synthetic Data

# COMMAND ----------

from hotel_reservations.data_processor import generate_synthetic_data

inference_data_skewed = generate_synthetic_data(modeling_ppl.X_train, drift= True, num_rows=200)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Tables and Update hotel_features_online

# COMMAND ----------

inference_data_skewed_spark = spark.createDataFrame(inference_data_skewed).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

inference_data_skewed_spark = inference_data_skewed_spark.withColumn("avg_price_per_room", col("avg_price_per_room").cast("float"))

inference_data_skewed_spark.write.mode("overwrite").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.inference_data_skewed"
)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ALTER TABLE mlops_dev.olalubic.hotel_reservations_features SET TBLPROPERTIES
# MAGIC -- (delta.enableChangeDataFeed = true)
# MAGIC
# MAGIC -- VACUUM mlops_dev.olalubic.hotel_reservations_features RETAIN 168 HOURS
# MAGIC
# MAGIC -- RESTORE TABLE mlops_dev.olalubic.hotel_reservations_features TO VERSION AS OF 35;

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient

workspace = WorkspaceClient()

#write into feature table; update online table
import time
from databricks.sdk import WorkspaceClient

workspace = WorkspaceClient()

#write into feature table; update online table
spark.sql(f"""
    INSERT INTO {config.catalog_name}.{config.schema_name}.hotel_reservations_features
    SELECT Client_ID, repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled, avg_price_per_room, no_of_special_requests
    FROM {config.catalog_name}.{config.schema_name}.inference_data_skewed
""")

online_table_name = f"{config.catalog_name}.{config.schema_name}.hotel_reservations_features_online"

existing_table = workspace.online_tables.get(online_table_name)
logger.info("Online table already exists. Inititating table update.")
pipeline_id = existing_table.spec.pipeline_id
update_response = workspace.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=False)

# COMMAND ----------

update_info = workspace.pipelines.get_update(pipeline_id=pipeline_id,
                        update_id=update_response.update_id)
state = update_info.update.state.value
print(state)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Send Data to the Endpoint

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
import numpy as np
import datetime
import itertools
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

test_set = modeling_ppl.X_test.copy()

inference_data_skewed = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed") \
                        .withColumn("Client_ID", col("Client_ID").cast("string")) \
                        .toPandas()
inference_data_skewed = inference_data_skewed.drop(columns=["update_timestamp_utc"], errors="ignore")

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import requests
import time

workspace = WorkspaceClient()

inference_data_skewed = inference_data_skewed.sort_values(by="Client_ID").reset_index(drop=True)
test_set = test_set.sort_values(by="Client_ID").reset_index(drop=True)

inference_data_skewed2 = inference_data_skewed.head(10)
test_set2 = test_set[test_set["Client_ID"].isin(['95890', '42942', '59530', '67761', '50264', '30130', '47704', '78754', '86194', '86480'])]

# Sample records from inference datasets
sampled_skewed_records = inference_data_skewed2.to_dict(orient="records")
test_set_records = test_set2.to_dict(orient="records")

# COMMAND ----------

# Two different way to send request to the endpoint
# 1. Using https endpoint
def send_request_https(dataframe_record):
    model_serving_endpoint = f"https://{host}/serving-endpoints/alubiss-custom-hotel-reservations-model-serving/invocations"
    response = requests.post(
        model_serving_endpoint,
        headers={"Authorization": f"Bearer {token}"},
        json={"dataframe_records": [dataframe_record]},
    )
    return response

# 2. Using workspace client
def send_request_workspace(dataframe_record):
    response = workspace.serving_endpoints.query(
        name="alubiss-custom-hotel-reservations-model-serving",
        dataframe_records=[dataframe_record]
    )
    return response

# COMMAND ----------

for index, record in enumerate(test_set_records):
    print(f"Sending request for test data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)

# COMMAND ----------

# Loop over test records and send requests for 10 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=5)
for index, record in enumerate(itertools.cycle(test_set_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for test data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)


# COMMAND ----------

# Loop over skewed records and send requests for 10 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=5)
for index, record in enumerate(itertools.cycle(sampled_skewed_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for skewed data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Refresh Monitoring

# COMMAND ----------

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient

from hotel_reservations.config import ProjectConfig
from hotel_reservations.monitoring import create_or_refresh_monitoring

spark = DatabricksSession.builder.getOrCreate()
workspace = WorkspaceClient()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)

# COMMAND ----------
