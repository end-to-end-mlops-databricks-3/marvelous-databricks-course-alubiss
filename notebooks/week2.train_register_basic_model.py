# Databricks notebook source
# MAGIC %pip install ../dist/hotel_reservations-0.1.5-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys

import mlflow
from pyspark.sql import SparkSession

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src")))

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.basic_model import BasicModel

from dotenv import load_dotenv

# COMMAND ----------

# If you have DEFAULT profile and are logged in with DEFAULT profile,
# skip these lines

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2", "job_run_id": "1234567890"})

# COMMAND ----------

# Initialize model with the config path
basic_model = BasicModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

basic_model.load_data()
basic_model.prepare_features()

# COMMAND ----------

# Train + log the model (runs everything including MLflow logging)
basic_model.train()
basic_model.log_model()

# COMMAND ----------

run_id = mlflow.search_runs(
    experiment_names=["/Shared/alubiss-model-basic"], filter_string="tags.branch='week2'"
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/base-model")

# COMMAND ----------

# Retrieve dataset for the current run
basic_model.retrieve_current_run_dataset()

# COMMAND ----------

# Retrieve metadata for the current run
basic_model.retrieve_current_run_metadata()

# COMMAND ----------

# Register model
basic_model.register_model()

# COMMAND ----------

# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_df = basic_model.load_latest_model_and_predict(X_test)

# COMMAND ----------
