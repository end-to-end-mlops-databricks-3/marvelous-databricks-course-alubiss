"""Data preprocessing module."""
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from hotel_reservations.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing DataFrame operations.

    This class handles data preprocessing, splitting, and saving to Databricks tables.
    """

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark

    def preprocess(self) -> None:
        """Preprocess the DataFrame stored in self.df.

        This method handles missing values, converts data types, and performs feature engineering.
        """

        self.df['arrival_date'] = '01'
        # Handle date features
        self.df['arrival_full_date'] = pd.to_datetime(
            self.df['arrival_date'].astype(str) + '-' +
            self.df['arrival_month'].astype(str) + '-' +
            self.df['arrival_year'].astype(str),
            format='%d-%m-%Y'
        )

        self.df['month_sin'] = np.sin(2 * np.pi * self.df['arrival_month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['arrival_month'] / 12)

        self.df['is_first_quarter'] = self.df['arrival_month'].apply(lambda x: 1 if x in [1, 2, 3] else 0)
        self.df['is_second_quarter'] = self.df['arrival_month'].apply(lambda x: 1 if x in [4, 5, 6] else 0)
        self.df['is_third_quarter'] = self.df['arrival_month'].apply(lambda x: 1 if x in [7, 8, 9] else 0)
        self.df['is_fourth_quarter'] = self.df['arrival_month'].apply(lambda x: 1 if x in [10, 11, 12] else 0)


        created_columns = [
            'month_sin',
            'month_cos',
            'is_first_quarter',
            'is_second_quarter',
            'is_third_quarter',
            'is_fourth_quarter'
        ]

        # Handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Convert categorical features to the appropriate type
        cat_features = self.config.cat_features
        for cat_col in cat_features:
            self.df[cat_col] = self.df[cat_col].astype("category")

        # Extract target and relevant features
        target = self.config.target
        relevant_columns = cat_features + num_features + [target] + self.config.id_cols + created_columns
        self.df = self.df[relevant_columns]
        for col in self.config.id_cols:
            self.df[col] = self.df[col].astype("str")

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

