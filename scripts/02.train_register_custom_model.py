import argparse

import os
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent / 'src'))
base_dir = os.path.abspath(str(Path.cwd().parent))
config_path = os.path.join(base_dir, "project_config.yml")

import mlflow
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.custom_model import CustomModel

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

try:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path",
        action="store",
        default=config_path,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--env",
        action="store",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--git_sha",
        action="store",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--job_run_id",
        action="store",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--branch",
        action="store",
        default=None,
        type=str,
        required=True,
    )
    args = parser.parse_args()
except:
    args = argparse.Namespace(
        root_path=config_path,
        env='dev',
        git_sha='123',
        job_run_id='unique_id',
        branch='alubiss'
    )

root_path = args.root_path
config_path = f"{root_path}"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize model
custom_model = CustomModel(
    config=config, tags=tags, spark=spark, code_paths=["../dist/house_price-1.0.1-py3-none-any.whl"]
)
logger.info("Model initialized.")

# Load data and prepare features
custom_model.load_data()
custom_model.prepare_features()
logger.info("Loaded data, prepared features.")

# Train + log the model (runs everything including MLflow logging)
custom_model.train()
custom_model.log_model()
logger.info("Model training completed.")

custom_model.register_model()
logger.info("Registered model")