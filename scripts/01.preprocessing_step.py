"""Data preprocessing module."""

import os
import sys
from pathlib import Path

import yaml
from loguru import logger
from pyspark.sql import SparkSession

sys.path.append(str(Path.cwd().parent / "src"))

from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
config_path = os.path.join(base_dir, "project_config.yml")

config = ProjectConfig.from_yaml(config_path=config_path, env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))
spark = SparkSession.builder.getOrCreate()
is_local = False

if is_local:
    df = spark.read.csv(
        "C:/Users/tomek/Documents/GitHub/marvelous-databricks-course-alubiss/tests/test_data/sample.csv",
        sep=";",
        header=True,
        inferSchema=True,
    ).toPandas()
else:
    # Load the house prices dataset
    df = spark.read.csv(
        f"/Volumes/{config.catalog_name}/{config.schema_name}/alubiss/hotel_reservations.csv",
        sep=";",
        header=True,
        inferSchema=True,
    ).toPandas()


# Preprocess the data
data_processor = DataProcessor(df, config, spark)
data_processor.preprocess()

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)
