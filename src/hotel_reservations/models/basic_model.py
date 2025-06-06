"""Basic model implementation.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.

num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
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


class BasicModel:
    """A basic model class for house price prediction using LightGBM.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration.

        :param config: Project configuration object
        :param tags: Tags object
        :param spark: SparkSession object
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.date_features = self.config.date_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.model_basic"
        self.tags = tags.dict()

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas().drop(columns=["update_timestamp_utc"], errors="ignore")
        self.test_set = (
            self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
            .toPandas()
            .drop(columns=["update_timestamp_utc"], errors="ignore")
        )
        self.data_version = "0"  # describe history -> retrieve

        self.X_train = self.train_set[
            self.num_features + self.cat_features + self.date_features + ["Client_ID", "Booking_ID"]
        ]
        self.y_train = self.train_set[self.target].map({"Not_Canceled": 0, "Canceled": 1})
        self.X_test = self.test_set[
            self.num_features + self.cat_features + self.date_features + ["Client_ID", "Booking_ID"]
        ]
        self.y_test = self.test_set[self.target].map({"Not_Canceled": 0, "Canceled": 1})
        self.train_set = self.train_set.drop(columns=[self.target])
        self.test_set = self.test_set.drop(columns=[self.target])
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self) -> None:
        """Encode categorical features and define a preprocessing pipeline.

        Creates a ColumnTransformer for one-hot encoding categorical features while passing through numerical
        features. Constructs a pipeline combining preprocessing and LightGBM regression model.
        """
        logger.info("🔄 Defining preprocessing pipeline...")
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features),
                ("drop_ids", "drop", ["arrival_month", "Client_ID", "Booking_ID"]),
            ],
            remainder="passthrough",
        )

        self.pipeline = Pipeline(
            steps=[
                ("date_features", DateFeatureEngineer()),
                ("preprocessor", self.preprocessor),
                ("regressor", LGBMClassifier(**self.parameters)),
            ]
        )
        logger.info("✅ Preprocessing pipeline defined.")

    def train(self) -> None:
        """Train the model."""
        logger.info("🚀 Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)

    def log_model(self) -> None:
        """Log the model using MLflow."""
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            y_pred = self.pipeline.predict(self.X_test)

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

            # Log the model
            signature = infer_signature(model_input=self.X_train, model_output=y_pred)
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")
            mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="base-model",
                signature=signature,
                pip_requirements=[
                    "mlflow==2.17.0",
                    "cloudpickle==3.1.0",
                    "defusedxml==0.7.1",
                    "lz4==4.3.2",
                    "matplotlib==3.9.2",
                    "numpy==1.26.4",
                    "pandas==2.2.3",
                    "pyarrow==14.0.1",
                    "pyspark==3.5.0",
                    "scikit-learn==1.5.2",
                    "scipy==1.14.1",
                ],
            )

    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/base-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=latest_version,
        )

    def retrieve_current_run_dataset(self) -> DatasetSource:
        """Retrieve MLflow run dataset.

        :return: Loaded dataset source
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        logger.info("✅ Dataset source loaded.")
        return dataset_source.load()

    def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
        """Retrieve MLflow run metadata.

        :return: Tuple containing metrics and parameters dictionaries
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        logger.info("✅ Dataset metadata loaded.")
        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Load the latest model from MLflow (alias=latest-model) and make predictions.

        Alias latest is not allowed -> we use latest-model instead as an alternative.

        :param input_data: Pandas DataFrame containing input features for prediction.
        :return: Pandas DataFrame with predictions.
        """
        logger.info("🔄 Loading model from MLflow alias 'production'...")

        model_uri = f"models:/{self.model_name}@latest-model"
        model = mlflow.sklearn.load_model(model_uri)

        logger.info("✅ Model successfully loaded.")

        # Make predictions
        predictions = model.predict(input_data)

        # Return predictions as a DataFrame
        return predictions
