def initialize_logging():
    import logging
    class ProgressAndErrorFilter(logging.Filter):
        def filter(self, record):
            if record.levelno >= logging.ERROR:
                return True
            if record.levelno == logging.INFO and ("[STATUS]" in record.getMessage() or "PMCID" in record.getMessage()):
                return True
            return False

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger()
    logger.addFilter(ProgressAndErrorFilter())
    logging.getLogger("py4j").setLevel(logging.ERROR)
    logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)
    logging.getLogger("org.apache.spark").setLevel(logging.ERROR)
    return logger

def load_existing_delta_data(table_name, spark):
    import logging
    import pandas as pd
    try:
        if spark.catalog.tableExists(table_name):
            existing_df = spark.table(table_name).toPandas()
            logging.debug("Existing table found. Loaded current data.")
        else:
            existing_df = pd.DataFrame()
            logging.debug("No existing table found. Starting fresh.")
    except Exception as e:
        import pandas as pd
        existing_df = pd.DataFrame()
        logging.error(f"Error loading existing data: {e}")
    return existing_df

def write_results_to_delta_table(merged_df, table_name, spark):
    import logging
    from pyspark.sql.types import StructType, StructField, StringType
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
        df = spark.createDataFrame(merged_df, schema)
        df.write.format("delta").mode("overwrite").saveAsTable(table_name)
        merged_df = spark.sql(f"SELECT * FROM {table_name}").toPandas()
        logging.debug(f"Results written to Spark table: {table_name}")
    except Exception as e:
        logging.error(f"Error writing to Delta table: {e}")
    return merged_df

def get_processed_pmids(existing_df):
    import pandas as pd
    if not existing_df.empty and "pmcid" in existing_df.columns:
        return set(existing_df["pmcid"].unique())
    return set()
