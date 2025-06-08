# Databricks notebook source
# MAGIC %pip install ../dist/hotel_reservations-0.1.5-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import hashlib
import os
import sys

import mlflow
from pyspark.sql import SparkSession

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src")))
import os

from dotenv import load_dotenv

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.modeling_pipeline import PocessModeling

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
basic_model = PocessModeling(
    config=config, tags=tags, spark=spark, code_paths=["../src/hotel_reservations/models/modeling_pipeline.py"]
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
basic_model_b = PocessModeling(
    config=config, tags=tags, spark=spark, code_paths=["../src/hotel_reservations/models/modeling_pipeline.py"]
)
basic_model_b.paramaters = {"learning_rate": 0.01, "n_estimators": 1000, "max_depth": 6}
basic_model_b.model_name = "model_basic_B"
basic_model_b.load_data()
basic_model_b.prepare_features()
basic_model_b.train()
basic_model_b.log_model()
basic_model_b.register_model()
model_B_uri = f"models:/mlops_dev.olalubic.{basic_model_b.model_name}@latest-model"

# COMMAND ----------

import pandas as pd


# define wrapper
class ModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model_a = mlflow.pyfunc.load_model(context.artifacts["pyfunc-alubiss-model_basic_A"])
        self.model_b = mlflow.pyfunc.load_model(context.artifacts["pyfunc-alubiss-model_basic_B"])

    def predict(self, context, model_input):
        house_id = str(model_input["Client_ID"].values[0])
        hashed_id = hashlib.md5(house_id.encode(encoding="UTF-8")).hexdigest()
        # convert a hexadecimal (base-16) string into an integer
        if int(hashed_id, 16) % 2:
            predictions = self.model_a.predict(model_input)
            return {"Prediction": predictions, "model": "Model A"}
        else:
            predictions = self.model_b.predict(model_input)
            return {"Prediction": predictions, "model": "Model B"}


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

from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, Schema
from pyspark.sql import SparkSession

mlflow.set_experiment(experiment_name="/Shared/model-ab-testing")
model_name = f"{catalog_name}.{schema_name}.model_pyfunc_ab_test"
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
            "pyfunc-alubiss-model_basic_B": model_B_uri,
            "banned_client_list": basic_model_b.banned_client_path,
        },
        code_paths=basic_model_b.code_paths,
        signature=signature,
    )
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-alubiss-model-ab", name=model_name, tags=tags.dict()
)

latest_version = model_version.version

from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="latest-ab-model",
    version=latest_version,
)


# COMMAND ----------

import mlflow

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

# COMMAND ----------


model_uri = f"models:/{model_name}@latest-ab-model"
model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(df)
predictions

# COMMAND ----------

data2 = [["Meal Plan 1", 0, "Room_Type 1", "Online", "PL", 2, 1, 2, 1, 26, 0, 0, 0, 161, 0, 10, "INN25630", "4"]]

df2 = pd.DataFrame(data2, columns=columns)
df2 = df2.astype(cols_types)

# COMMAND ----------


model_uri = f"models:/{model_name}@latest-ab-model"
model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(df2)
predictions
