"""Model monitoring module."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.modeling_pipeline import PocessModeling


def create_or_refresh_monitoring(config: ProjectConfig, spark: SparkSession, workspace: WorkspaceClient) -> None:
    """Create or refresh a monitoring table for model serving data.

    This function processes the inference data from a Delta table,
    parses the request and response JSON fields, joins with test and inference sets,
    and writes the resulting DataFrame to a Delta table for monitoring purposes.

    :param config: Configuration object containing catalog and schema names.
    :param spark: Spark session used for executing SQL queries and transformations.
    :param workspace: Workspace object used for managing quality monitors.
    """
    tags = Tags(**{"git_sha": "abcd12345", "branch": "week2", "job_run_id": "1234567890"})
    inf_table = spark.sql(
        f"SELECT * FROM {config.catalog_name}.{config.schema_name}.`model-serving-payload_payload`"
    )

    request_schema = StructType(
        [
            StructField(
                "dataframe_records",
                ArrayType(
                    StructType(
                        [
                            StructField("required_car_parking_space", IntegerType(), True),
                            StructField("no_of_adults", IntegerType(), True),
                            StructField("no_of_children", IntegerType(), True),
                            StructField("no_of_weekend_nights", IntegerType(), True),
                            StructField("no_of_week_nights", IntegerType(), True),
                            StructField("lead_time", IntegerType(), True),
                            StructField("repeated_guest", IntegerType(), True),
                            StructField("no_of_previous_cancellations", IntegerType(), True),
                            StructField("no_of_previous_bookings_not_canceled", IntegerType(), True),
                            StructField("avg_price_per_room", DoubleType(), True),
                            StructField("no_of_special_requests", IntegerType(), True),
                            StructField("type_of_meal_plan", StringType(), True),
                            StructField("room_type_reserved", StringType(), True),
                            StructField("market_segment_type", StringType(), True),
                            StructField("country", StringType(), True),
                            StructField("arrival_month", IntegerType(), True),
                            StructField("Client_ID", StringType(), True),
                            StructField("Booking_ID", StringType(), True),
                        ]
                    )
                ),
                True,
            )
        ]
    )

    response_schema = StructType(
        [
            StructField("Client_ID", StringType(), True),
            StructField("Proba", DoubleType(), True),
            StructField("Comment", StringType(), True),
            StructField(
                "databricks_output",
                StructType(
                    [StructField("trace", StringType(), True), StructField("databricks_request_id", StringType(), True)]
                ),
                True,
            ),
        ]
    )

    inf_table_parsed = inf_table.withColumn("parsed_request", F.from_json(F.col("request"), request_schema))

    inf_table_parsed = inf_table_parsed.withColumn("parsed_response", F.from_json(F.col("response"), response_schema))

    df_exploded = inf_table_parsed.withColumn("record", F.explode(F.col("parsed_request.dataframe_records")))

    df_final = df_exploded.withColumn(
        "timestamp_ms", (F.col("request_time").cast("long") * 1000)
    ).select(
        F.col("request_time").alias("timestamp"),  # Use request_time as the timestamp
        F.col("timestamp_ms"),  # Select the newly created timestamp_ms column
        "databricks_request_id",
        "execution_duration_ms",
        F.col("record.required_car_parking_space").alias("required_car_parking_space"),
        F.col("record.no_of_adults").alias("no_of_adults"),
        F.col("record.no_of_children").alias("no_of_children"),
        F.col("record.no_of_weekend_nights").alias("no_of_weekend_nights"),
        F.col("record.no_of_week_nights").alias("no_of_week_nights"),
        F.col("record.lead_time").alias("lead_time"),
        F.col("record.repeated_guest").alias("repeated_guest"),
        F.col("record.no_of_previous_cancellations").alias("no_of_previous_cancellations"),
        F.col("record.no_of_previous_bookings_not_canceled").alias("no_of_previous_bookings_not_canceled"),
        F.col("record.avg_price_per_room").alias("avg_price_per_room"),
        F.col("record.no_of_special_requests").alias("no_of_special_requests"),
        F.col("record.type_of_meal_plan").alias("type_of_meal_plan"),
        F.col("record.room_type_reserved").alias("room_type_reserved"),
        F.col("record.market_segment_type").alias("market_segment_type"),
        F.col("record.country").alias("country"),
        F.col("record.arrival_month").alias("arrival_month"),
        F.col("record.Client_ID").alias("Client_ID"),
        F.col("record.Booking_ID").alias("Booking_ID"),
        F.col("parsed_response.Proba")[0].alias("prediction"),
        F.lit("hotel_reservations").alias("model_name"),
    )

    modeling_ppl = PocessModeling(
    config=config, tags=tags, spark=spark, code_paths=["../src/hotel_reservations/models/modeling_pipeline.py"]
    )
    logger.info("Model initialized.")

    # Load data and prepare features
    modeling_ppl.load_data()
    modeling_ppl.prepare_features()
    logger.info("Loaded data, prepared features.")

    test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")
    inference_data_skewed = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed") \
                        .withColumn("Client_ID", col("Client_ID").cast("string")) \
                        .toPandas()
    inference_data_skewed = inference_data_skewed.drop(columns=["update_timestamp_utc"], errors="ignore")
    inference_data_skewed = inference_data_skewed[
                modeling_ppl.num_features + modeling_ppl.cat_features + modeling_ppl.date_features + ["Client_ID", "Booking_ID"]
            ]
    inference_set_skewed = modeling_ppl.pipeline[:-1].transform(inference_data_skewed)

    df_final_with_status = (
        df_final.join(test_set.select("Client_ID", "booking_status"), on="Client_ID", how="left")
        .withColumnRenamed("Client_ID", "booking_status_test")
        .join(inference_set_skewed.select("Client_ID", "booking_status"), on="Client_ID", how="left")
        .withColumnRenamed("Client_ID", "booking_status_inference")
        .select("*", F.coalesce(F.col("booking_status_test"), F.col("booking_status_inference")).alias("booking_status"))
        .drop("booking_status_test", "booking_status_inference")
        .withColumn("booking_status", F.col("booking_status").cast("integer"))
        .withColumn("prediction", F.col("prediction").cast("double"))
        .dropna(subset=["booking_status", "prediction"])
    )

    hotel_reservations_features = spark.table(f"{config.catalog_name}.{config.schema_name}.hotel_reservations_features")

    df_final_with_features = df_final_with_status.join(hotel_reservations_features, on="Client_ID", how="left")

    df_final_with_features.write.format("delta").mode("append").saveAsTable(
        f"{config.catalog_name}.{config.schema_name}.model_monitoring"
    )

    try:
        workspace.quality_monitors.get(f"{config.catalog_name}.{config.schema_name}.model_monitoring")
        workspace.quality_monitors.run_refresh(
            table_name=f"{config.catalog_name}.{config.schema_name}.model_monitoring"
        )
        logger.info("Lakehouse monitoring table exist, refreshing.")
    except NotFound:
        create_monitoring_table(config=config, spark=spark, workspace=workspace)
        logger.info("Lakehouse monitoring table is created.")


def create_monitoring_table(config: ProjectConfig, spark: SparkSession, workspace: WorkspaceClient) -> None:
    """Create a new monitoring table for model monitoring.

    This function sets up a monitoring table using the provided configuration,
    SparkSession, and workspace. It also enables Change Data Feed for the table.

    :param config: Configuration object containing catalog and schema names
    :param spark: SparkSession object for executing SQL commands
    :param workspace: Workspace object for creating quality monitors
    """
    logger.info("Creating new monitoring table..")

    monitoring_table = f"{config.catalog_name}.{config.schema_name}.model_monitoring"

    workspace.quality_monitors.create(
        table_name=monitoring_table,
        assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
        output_schema_name=f"{config.catalog_name}.{config.schema_name}",
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
            prediction_col="prediction",
            timestamp_col="timestamp",
            granularities=["30 minutes"],
            model_id_col="model_name",
            label_col="booking_status",
        ),
    )

    # Important to update monitoring
    spark.sql(f"ALTER TABLE {monitoring_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
