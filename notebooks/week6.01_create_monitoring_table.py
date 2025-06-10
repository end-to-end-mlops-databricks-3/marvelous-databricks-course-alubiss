# Databricks notebook source
# MAGIC %pip install ../dist/hotel_reservations-0.1.5-py3-none-any.whl

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
logger.info("Loaded data, prepared features.")

# COMMAND ----------


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

features = modeling_ppl.X_train.copy()
target = modeling_ppl.y_train.copy()

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(features, target)

# Identify the most important features
feature_importances = pd.DataFrame({
    'Feature': features.columns,
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

inference_data_skewed_spark.write.mode("overwrite").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.inference_data_skewed"
)

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
update_response = workspace.pipelines.start_update(
    pipeline_id=pipeline_id, full_refresh=False)
while True:
    update_info = workspace.pipelines.get_update(pipeline_id=pipeline_id, 
                            update_id=update_response.update_id)
    state = update_info.update.state.value
    if state == 'COMPLETED':
        break
    elif state in ['FAILED', 'CANCELED']:
        raise SystemError("Online table failed to update.")
    elif state == 'WAITING_FOR_RESOURCES':
        print("Pipeline is waiting for resources.")
    else:
        print(f"Pipeline is in {state} state.")
    time.sleep(30)

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
config = ProjectConfig.from_yaml(config_path="project_config.yml", env="dev")

test_set = modeling_ppl.X_test.copy()

inference_data_skewed = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed") \
                        .withColumn("Client_ID", col("Client_ID").cast("string")) \
                        .toPandas()
inference_data_skewed = inference_data_skewed.drop(columns=["update_timestamp_utc"], errors="ignore")
inference_data_skewed = inference_data_skewed[
            modeling_ppl.num_features + modeling_ppl.cat_features + modeling_ppl.date_features + ["Client_ID", "Booking_ID"]
        ]
inference_data_skewed = modeling_ppl.pipeline[:-1].transform(inference_data_skewed)


# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------


from databricks.sdk import WorkspaceClient
import requests
import time

workspace = WorkspaceClient()

# Sample records from inference datasets
sampled_skewed_records = inference_data_skewed.to_dict(orient="records")
test_set_records = test_set.to_dict(orient="records")

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

# Loop over test records and send requests for 10 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=20)
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
end_time = datetime.datetime.now() + datetime.timedelta(minutes=30)
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
# MAGIC

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
