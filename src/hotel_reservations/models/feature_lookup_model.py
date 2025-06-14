"""FeatureLookUp model implementation."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src")))

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hotel_reservations.config import ProjectConfig, Tags


class DateFeatureEngineer(BaseEstimator, TransformerMixin):
    """Date features engineering class."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "DateFeatureEngineer":
        """Fit method for date feature engineering."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform method for date feature engineering."""
        X = X.copy()
        X["month_sin"] = np.sin(2 * np.pi * X["arrival_month"] / 12)
        X["month_cos"] = np.cos(2 * np.pi * X["arrival_month"] / 12)
        X["is_first_quarter"] = X["arrival_month"].apply(lambda x: 1 if x in [1, 2, 3] else 0)
        X["is_second_quarter"] = X["arrival_month"].apply(lambda x: 1 if x in [4, 5, 6] else 0)
        X["is_third_quarter"] = X["arrival_month"].apply(lambda x: 1 if x in [7, 8, 9] else 0)
        X["is_fourth_quarter"] = X["arrival_month"].apply(lambda x: 1 if x in [10, 11, 12] else 0)
        return X


class FeatureLookUpModel:
    """A class to manage FeatureLookupModel."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration."""
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.date_features = self.config.date_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.hotel_reservations_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_cancelled_rate"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

        # define code path
        self.code_path = ["../dist/hotel_reservations-0.1.12-py3-none-any.whl"]

    def create_feature_table(self) -> None:
        """Create or update the hotel_reservations_features table and populate it.

        This table stores features related to hotel reservations.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (Client_ID STRING NOT NULL, repeated_guest INT, no_of_previous_cancellations INT, no_of_previous_bookings_not_canceled INT, avg_price_per_room FLOAT, no_of_special_requests INT);
        """)
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT hotel_res_pk PRIMARY KEY(Client_ID);")
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Client_ID, repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled, avg_price_per_room, no_of_special_requests FROM {self.catalog_name}.{self.schema_name}.train_set"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Client_ID, repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled, avg_price_per_room, no_of_special_requests FROM {self.catalog_name}.{self.schema_name}.test_set"
        )
        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self) -> None:
        """Define a function to calculate the house's age.

        This function subtracts the year built from the current year.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(canceled INT, not_canceled INT)
        RETURNS FLOAT
        LANGUAGE PYTHON AS
        $$
        return -1 if (canceled + not_canceled) == 0 else canceled / (canceled + not_canceled)
        $$
        """)
        logger.info("✅ Feature function defined.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Drops specified columns and casts 'YearBuilt' to integer type.
        """
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").drop(
            "repeated_guest",
            "no_of_previous_cancellations",
            "no_of_previous_bookings_not_canceled",
            "avg_price_per_room",
            "no_of_special_requests",
        )
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

        self.X_test = self.test_set[
            self.num_features + self.cat_features + self.date_features + ["Client_ID", "Booking_ID"]
        ]

        self.train_set = self.train_set.withColumn("Client_ID", self.train_set["Client_ID"].cast("string"))

        logger.info("✅ Data successfully loaded.")

    def feature_engineering(self) -> None:
        """Perform feature engineering by linking data with feature tables.

        Creates a training set using FeatureLookup and FeatureFunction.
        """
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=[
                        "repeated_guest",
                        "no_of_previous_cancellations",
                        "no_of_previous_bookings_not_canceled",
                        "avg_price_per_room",
                        "no_of_special_requests",
                    ],
                    lookup_key="Client_ID",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="cancelled_rate",
                    input_bindings={
                        "canceled": "no_of_previous_cancellations",
                        "not_canceled": "no_of_previous_bookings_not_canceled",
                    },
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        self.test_set["cancelled_rate"] = self.test_set.apply(
            lambda row: -1
            if (row["no_of_previous_cancellations"] + row["no_of_previous_bookings_not_canceled"]) == 0
            else row["no_of_previous_cancellations"]
            / (row["no_of_previous_cancellations"] + row["no_of_previous_bookings_not_canceled"]),
            axis=1,
        )

        self.X_train = self.training_df[
            self.num_features + self.cat_features + self.date_features + ["cancelled_rate", "Client_ID", "Booking_ID"]
        ]
        self.y_train = self.training_df[self.target].map({"Not_Canceled": 0, "Canceled": 1})
        self.X_test = self.test_set[
            self.num_features + self.cat_features + self.date_features + ["cancelled_rate", "Client_ID", "Booking_ID"]
        ]
        self.y_test = self.test_set[self.target].map({"Not_Canceled": 0, "Canceled": 1})

        logger.info("✅ Feature engineering completed.")

    def train(self) -> None:
        """Train the model and log results to MLflow.

        Uses a pipeline with preprocessing and LightGBM regressor.
        """
        logger.info("🚀 Starting training...")

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features),
                ("drop_ids", "drop", ["arrival_month", "Client_ID", "Booking_ID"]),
            ],
            remainder="passthrough",
        )

        pipeline = Pipeline(
            steps=[
                ("date_features", DateFeatureEngineer()),
                ("preprocessor", preprocessor),
                ("regressor", LGBMClassifier(**self.parameters)),
            ]
        )

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

            # Evaluate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            logger.info(f"📊 Accuracy: {accuracy}")
            logger.info(f"📊 Precision: {precision}")
            logger.info(f"📊 Recall: {recall}")
            logger.info(f"📊 F1 Score: {f1}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "LightGBMClassifier with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1score", f1)

            signature = infer_signature(self.X_train, y_pred)
            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="alubiss-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

    def register_model(self) -> str:
        """Register the trained model to MLflow registry.

        Registers the model and sets alias to 'latest-model'.
        """
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/alubiss-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe",
            alias="latest-model",
            version=latest_version,
        )

        return latest_version

    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Loads the model with the alias 'latest-model' and scores the batch.
        :param X: DataFrame containing the input features.
        :return: DataFrame containing the predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions
