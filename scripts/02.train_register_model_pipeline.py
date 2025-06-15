"""Modeling Pipeline module."""

from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations.common import create_parser
from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.modeling_pipeline import PocessModeling

args = create_parser()

root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize model
modeling_ppl = PocessModeling(
    config=config, tags=tags, spark=spark, code_paths=["../src/hotel_reservations/models/modeling_pipeline.py"]
)
logger.info("Model initialized.")

# Load data and prepare features
modeling_ppl.load_data()
modeling_ppl.prepare_features()
logger.info("Loaded data, prepared features.")

# Train + log the model (runs everything including MLflow logging)
if config.hyperparameters_tuning:
    modeling_ppl.tune_hyperparameters()
modeling_ppl.train()
modeling_ppl.log_model()

# Evaluate model
# Load test set from Delta table
spark = SparkSession.builder.getOrCreate()
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(100)
test_set = test_set.toPandas()

model_improved = modeling_ppl.model_improved(test_set=test_set)
logger.info(f"Model evaluation completed, model improved: {model_improved}")

is_test = args.is_test

# when running test, always register and deploy
if is_test == 1:
    model_improved = True

if model_improved:
    # Register the model
    latest_version = modeling_ppl.register_model()
    logger.info(f"New model registered with version: {latest_version}")
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)

else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)
