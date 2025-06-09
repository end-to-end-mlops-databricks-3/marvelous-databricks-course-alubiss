"""Deploy module."""

from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations.common import create_parser
from hotel_reservations.config import ProjectConfig
from hotel_reservations.serving.model_serving import ModelServing

args = create_parser()

root_path = args.root_path
is_test = args.is_test
config_path = f"{root_path}/files/project_config.yml"

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")
logger.info(f"Model version: {model_version}")

# Load project config
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
logger.info("Loaded config file.")

catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = f"alubiss-model-serving-pipeline-{args.env}"

# Initialize Feature Lookup Serving Manager
feature_model_server = ModelServing(
    f"{catalog_name}.{schema_name}.hotel_reservations_model_custom",
    endpoint_name=endpoint_name,
)

# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint(version=model_version)
logger.info("Started deployment/update of the serving endpoint.")

# Delete endpoint if test
if is_test == 1:
    workspace = WorkspaceClient()
    workspace.serving_endpoints.delete(name=endpoint_name)
    logger.info("Deleting serving endpoint.")
