"""Data preprocessing module."""

import sys
from pathlib import Path

import yaml
from loguru import logger
from pyspark.sql import SparkSession

sys.path.append(str(Path.cwd().parent / "src"))

from hotel_reservations.common import create_parser_by_country
from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor

args = create_parser_by_country()

root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the  dataset
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/alubiss/hotel_reservations.csv",
    header=True,
    inferSchema=True,
    sep=";",
).toPandas()

country_df = df[df["country"] == args.country]

# Preprocess the data
data_processor = DataProcessor(country_df, config, spark)
data_processor.preprocess()

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# Save to catalog
logger.info(f"Saving data to catalog for {args.country}")
data_processor.save_by_country(X_train, X_test, args.country)
