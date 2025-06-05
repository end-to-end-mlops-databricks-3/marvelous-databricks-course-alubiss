# Databricks notebook source
import hashlib

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from mlflow.models import infer_signature
from pyspark.sql import SparkSession

import os
import sys
from typing import Literal

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src")))
from hotel_reservations.models.modeling_pipeline import PocessModeling
from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.utils import serving_pred_function
from dotenv import load_dotenv
import os
import requests
import time

# COMMAND ----------

def is_databricks() -> bool:
    """Check if the code is running in a Databricks environment.

    :return: True if running in Databricks, False otherwise.
    """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

# COMMAND ----------

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="prd")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week4", "job_run_id": "1234"})

# COMMAND ----------

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# train model A
basic_model = PocessModeling(config=config, tags=tags, spark=spark, code_paths=["../src/hotel_reservations/models/modeling_pipeline.py"]
)
basic_model.model_name = "model_basic_A"
basic_model.load_data()
basic_model.prepare_features()
basic_model.train()
basic_model.log_model()
basic_model.register_model()
model_A_uri = f"models:/mlops_dev.olalubic.{basic_model.model_name}@latest-model"

# COMMAND ----------

# train model B
basic_model_b = PocessModeling(config=config, tags=tags, spark=spark, code_paths=["../src/hotel_reservations/models/modeling_pipeline.py"]
)
basic_model_b.paramaters = {"learning_rate": 0.01,
                            "n_estimators": 1000,
                            "max_depth": 6}
basic_model_b.model_name = f"model_basic_B"
basic_model_b.load_data()
basic_model_b.prepare_features()
basic_model_b.train()
basic_model_b.log_model()
basic_model_b.register_model()
model_B_uri = f"models:/mlops_dev.olalubic.{basic_model_b.model_name}@latest-model"

# COMMAND ----------

import pandas as pd
from loguru import logger

# define wrapper
class ModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model_a = mlflow.sklearn.load_model(
            context.artifacts["pyfunc-alubiss-model_basic_A"]
        )
        self.model_b = mlflow.sklearn.load_model(
            context.artifacts["pyfunc-alubiss-model_basic_B"]
        )

    def predict(self, context, model_input):
        house_id = str(model_input["Client_ID"].values[0])
        hashed_id = hashlib.md5(house_id.encode(encoding="UTF-8")).hexdigest()
        # convert a hexadecimal (base-16) string into an integer
        if int(hashed_id, 16) % 2:
            banned_client_list = pd.read_csv(context.artifacts["banned_client_list"], sep=";")
            client_ids = model_input["Client_ID"].values

            predictions = self.model_a.predict_proba(model_input)
            proba_canceled = predictions[:, 1]

            adjusted_predictions = serving_pred_function(client_ids, banned_client_list, proba_canceled)

            comment = [
                "Banned" if client_id in banned_client_list["banned_clients_ids"].values else "None"
                for client_id in client_ids
            ]
            logger.info("MODEL A")
            return pd.DataFrame(
                {
                    "Client_ID": client_ids,
                    "Proba": adjusted_predictions,
                    "Comment": comment,
                }
            )
        else:
            banned_client_list = pd.read_csv(context.artifacts["banned_client_list"], sep=";")
            client_ids = model_input["Client_ID"].values

            predictions = self.model_b.predict_proba(model_input)
            proba_canceled = predictions[:, 1]

            adjusted_predictions = serving_pred_function(client_ids, banned_client_list, proba_canceled)

            comment = [
                "Banned" if client_id in banned_client_list["banned_clients_ids"].values else "None"
                for client_id in client_ids
            ]
            logger.info("MODEL B")
            return pd.DataFrame(
                {
                    "Client_ID": client_ids,
                    "Proba": adjusted_predictions,
                    "Comment": comment,
                }
            )

# COMMAND ----------

train_set_spark = spark.table(f"{basic_model_b.catalog_name}.{basic_model_b.schema_name}.train_set")
train_set = train_set_spark.toPandas().drop(columns=["update_timestamp_utc"], errors="ignore")
test_set = (
    spark.table(f"{basic_model_b.catalog_name}.{basic_model_b.schema_name}.test_set")
    .toPandas()
    .drop(columns=["update_timestamp_utc"], errors="ignore")
)
data_version = "0"  # describe history -> retrieve

X_train = train_set[
    basic_model_b.num_features + basic_model_b.cat_features + basic_model_b.date_features + ["Client_ID", "Booking_ID"]
]
y_train = train_set[basic_model_b.target].map({"Not_Canceled": 0, "Canceled": 1})
X_test = test_set[
    basic_model_b.num_features + basic_model_b.cat_features + basic_model_b.date_features + ["Client_ID", "Booking_ID"]
]
y_test = test_set[basic_model_b.target].map({"Not_Canceled": 0, "Canceled": 1})
train_set = train_set.drop(columns=[basic_model_b.target])
test_set = test_set.drop(columns=[basic_model_b.target])

X_train = train_set
X_test = test_set

# COMMAND ----------

from mlflow.data.dataset_source import DatasetSource
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, Schema
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlflow.set_experiment(experiment_name="/Shared/model-ab-testing")
model_name = f"{catalog_name}.{schema_name}.alubiss_model_pyfunc_ab_test"
wrapped_model = ModelWrapper()

with mlflow.start_run() as run:
    run_id = run.info.run_id
    input_signature = Schema(
            [
                ColSpec("integer", "required_car_parking_space"),
                ColSpec("integer", "no_of_adults"),
                ColSpec("integer", "no_of_children"),
                ColSpec("integer", "no_of_weekend_nights"),
                ColSpec("integer", "no_of_week_nights"),
                ColSpec("integer", "lead_time"),
                ColSpec("integer", "repeated_guest"),
                ColSpec("integer", "no_of_previous_cancellations"),
                ColSpec("integer", "no_of_previous_bookings_not_canceled"),
                ColSpec("float", "avg_price_per_room"),
                ColSpec("integer", "no_of_special_requests"),
                ColSpec("string", "type_of_meal_plan"),
                ColSpec("string", "room_type_reserved"),
                ColSpec("string", "market_segment_type"),
                ColSpec("string", "country"),
                ColSpec("integer", "arrival_month"),
                ColSpec("string", "Client_ID"),
                ColSpec("string", "Booking_ID"),
            ]
        )

    output_signature = Schema(
        [
            ColSpec("string", "Client_ID"),
            ColSpec("double", "Proba"),
            ColSpec("string", "Comment"),
        ]
    )

    signature = ModelSignature(inputs=input_signature, outputs=output_signature)

    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-alubiss-model-ab",
        artifacts={
            "pyfunc-alubiss-model_basic_A": model_A_uri,
            "pyfunc-alubiss-model_basic_A": model_B_uri,
            "banned_client_list": basic_model_b.banned_client_path},
        code_paths=basic_model_b.code_paths,
        signature=signature
    )
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-alubiss-model-ab", name=model_name, tags=tags.dict()
)

# COMMAND ----------


