import logging
from typing import Any, Set
import pandas as pd  # type: ignore
from pyspark.sql.types import StructType, StructField, StringType  # type: ignore

logger = logging.getLogger(__name__)

class ProgressAndErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filters log records to allow only ERROR level logs and specific INFO logs.

        Parameters:
            record (logging.LogRecord): The log record to be evaluated.

        Returns:
            bool: True if the record is an ERROR or an INFO message containing "[STATUS]" or "PMCID", otherwise False.
        """
        if record.levelno >= logging.ERROR:
            return True
        if record.levelno == logging.INFO and (
            "[STATUS]" in record.getMessage() or "PMCID" in record.getMessage()
        ):
            return True
        return False

def initialize_logging() -> logging.Logger:
    """
    Initializes the logging configuration and returns the root logger with a custom filter.

    Returns:
        logging.Logger: The configured root logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    root_logger = logging.getLogger()
    root_logger.addFilter(ProgressAndErrorFilter())
    # Reduce verbosity for external libraries
    logging.getLogger("py4j").setLevel(logging.ERROR)
    logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)
    logging.getLogger("org.apache.spark").setLevel(logging.ERROR)
    return root_logger

def load_existing_delta_data(table_name: str, spark: Any) -> pd.DataFrame:
    """
    Loads existing Delta data from a Spark table if it exists.

    Parameters:
        table_name (str): The name of the Delta table to load data from.
        spark: The Spark session instance used to interact with Spark tables.

    Returns:
        pd.DataFrame: A DataFrame containing the existing table data; returns an empty DataFrame if the table does not exist or an error occurs.
    """
    try:
        if spark.catalog.tableExists(table_name):
            existing_df = spark.table(table_name).toPandas()
            logger.debug("Existing table found. Loaded current data.")
        else:
            existing_df = pd.DataFrame()
            logger.debug("No existing table found. Starting fresh.")
    except Exception as e:
        logger.exception("Error loading existing data from table %s: %s", table_name, e)
        existing_df = pd.DataFrame()
    return existing_df

def write_results_to_delta_table(merged_df: pd.DataFrame, table_name: str, spark: Any) -> pd.DataFrame:
    """
    Writes merged results to a Delta table using Spark and reloads the updated data.

    Parameters:
        merged_df (pd.DataFrame): The DataFrame containing merged results to write to the table.
        table_name (str): The target Delta table name for saving the data.
        spark: The Spark session instance for writing to and reading from the table.

    Returns:
        pd.DataFrame: The DataFrame reloaded from the Delta table after writing the results.
    """
    try:
        schema = StructType([
            StructField("pmcid", StringType(), True),
            StructField("llm", StringType(), True),
            StructField("output_type", StringType(), True),
            StructField("verbatim_output", StringType(), True),
            StructField("interpretation", StringType(), True),
            StructField("original_llm_output", StringType(), True),
            StructField("input_prompt", StringType(), True),
            StructField("llm_raw_llm_output", StringType(), True),
            StructField("error_log", StringType(), True)
        ])
        df_spark = spark.createDataFrame(merged_df, schema)
        df_spark.write.format("delta").mode("append").saveAsTable(table_name)
        merged_df = spark.sql(f"SELECT * FROM {table_name}").toPandas()
        logger.debug("Results written to Spark table: %s", table_name)
    except Exception as e:
        logger.exception("Error writing to Delta table %s: %s", table_name, e)
    return merged_df

def get_processed_pmids(existing_df: pd.DataFrame) -> Set[str]:
    """
    Extracts and returns a set of processed PMCIDs from the provided DataFrame.

    Parameters:
        existing_df (pd.DataFrame): A DataFrame that may contain a 'pmcid' column.

    Returns:
        Set[str]: A set containing unique PMCIDs extracted from the DataFrame; an empty set if not applicable.
    """
    if not existing_df.empty and "pmcid" in existing_df.columns:
        return set(existing_df["pmcid"].unique())
    return set()
