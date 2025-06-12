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
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType

from hotel_reservations.config import ProjectConfig, Tags


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
    inf_table = spark.sql(f"SELECT * FROM {config.catalog_name}.{config.schema_name}.`model-serving-payload_payload`")

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

    df_parsed = inf_table.withColumn("parsed_request", F.from_json(F.col("request"), request_schema))
    df_exploded = df_parsed.withColumn("record", F.explode(F.col("parsed_request.dataframe_records")))
    final_df_with_requests = df_exploded.select(
        [
            F.col(f"record.{col.name}").alias(col.name)
            for col in request_schema["dataframe_records"].dataType.elementType.fields
        ]
    )
    final_df_with_requests = final_df_with_requests.withColumn("model_name", (F.lit("hotel_reservations")))

    df_parsed = inf_table.withColumn("parsed_response", F.from_json(F.col("response"), response_schema))
    final_df_with_response = df_parsed.select(
        F.col("parsed_response.Client_ID").alias("Client_ID"),
        F.col("parsed_response.Proba").alias("Proba"),
        F.col("parsed_response.Comment").alias("Comment"),
        F.col("parsed_response.databricks_output.trace").alias("trace"),
        F.col("parsed_response.databricks_output.databricks_request_id").alias("databricks_request_id"),
    )

    main_df = spark.sql("SELECT request_date, request_time  FROM mlops_dev.olalubic.`model-serving-payload_payload`")

    main_df = main_df.withColumn("row_id", monotonically_increasing_id())
    final_df_with_response = final_df_with_response.withColumn("row_id", monotonically_increasing_id())
    final_df_with_requests = final_df_with_requests.withColumn("row_id", monotonically_increasing_id())

    result_df = (
        main_df.join(final_df_with_response, on="row_id", how="inner")
        .join(final_df_with_requests.drop("Client_ID"), on="row_id", how="inner")
        .drop("row_id")
    )

    train = spark.sql("SELECT Client_ID, booking_status FROM mlops_dev.olalubic.train_set")
    test = spark.sql("SELECT Client_ID, booking_status  FROM mlops_dev.olalubic.test_set")
    combined_df = test.unionByName(train)

    result_df = result_df.join(combined_df, on="Client_ID", how="inner")

    result_df = result_df.dropna(subset=["booking_status", "prediction"])
    result_df = result_df.withColumn("timestamp", F.col("request_time"))
    result_df = result_df.withColumn("prediction_nr", when(col("Proba") > 0.5, 1).otherwise(0))
    result_df = result_df.withColumn(
        "booking_status_nr",
        when(col("booking_status") == "Canceled", 1).when(col("booking_status") == "Not_Canceled", 0),
    )

    # hotel_reservations_features = spark.table(f"{config.catalog_name}.{config.schema_name}.hotel_reservations_features")

    # df_final_with_features = df_final_with_status.join(hotel_reservations_features, on="Client_ID", how="left")

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
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            prediction_col="prediction_nr",
            timestamp_col="timestamp",
            granularities=["30 minutes"],
            model_id_col="model_name",
            label_col="booking_status_nr",
        ),
    )

    # Important to update monitoring
    spark.sql(f"ALTER TABLE {monitoring_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
