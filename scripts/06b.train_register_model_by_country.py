"""Modeling Pipeline module."""

from loguru import logger
from pyspark.sql import SparkSession

from hotel_reservations.common import create_parser_by_country
from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.modeling_pipeline import PocessModeling

args = create_parser_by_country()

root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "aaa", "branch": "aaa", "job_run_id": "123"}
tags = Tags(**tags_dict)

# Initialize model
modeling_ppl = PocessModeling(
    config=config, tags=tags, spark=spark, code_paths=["../src/hotel_reservations/models/modeling_pipeline.py"]
)
logger.info("Model initialized.")

# Prepare model by country
modeling_ppl.model_name = f"hotel_reservations_model_{args.country}"

# Load data and prepare features
modeling_ppl.load_data_by_country(args.country)
modeling_ppl.prepare_features()
logger.info("Loaded data, prepared features.")

# Train + log the model (runs everything including MLflow logging)
if config.hyperparameters_tuning:
    modeling_ppl.tune_hyperparameters()
modeling_ppl.train()
modeling_ppl.log_model()
