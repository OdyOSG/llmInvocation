import logging
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType

logger = logging.getLogger(__name__)


class ProgressAndErrorFilter(logging.Filter):
    def filter(self, record):
        if record.levelno >= logging.ERROR:
            return True
        if record.levelno == logging.INFO and (
            "[STATUS]" in record.getMessage() or "PMCID" in record.getMessage()
        ):
            return True
        return False


def initialize_logging():
    """Initializes logging configuration."""
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


def load_existing_delta_data(table_name, spark):
    """Loads existing Delta data from a Spark table if it exists."""
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


def write_results_to_delta_table(merged_df, table_name, spark):
    """Writes merged results to a Delta table using Spark."""
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
        df_spark.write.format("delta").mode("overwrite").saveAsTable(table_name)
        merged_df = spark.sql(f"SELECT * FROM {table_name}").toPandas()
        logger.debug("Results written to Spark table: %s", table_name)
    except Exception as e:
        logger.exception("Error writing to Delta table %s: %s", table_name, e)
    return merged_df


def get_processed_pmids(existing_df):
    """Returns a set of processed PMCIDs from the existing dataframe."""
    if not existing_df.empty and "pmcid" in existing_df.columns:
        return set(existing_df["pmcid"].unique())
    return set()
