import re
from typing import Any, Optional, Pattern
import pandas as pd  # type: ignore
from .data_management import (
    initialize_logging,
    load_existing_delta_data,
    write_results_to_delta_table,
    get_processed_pmids
)
from .llm_invocation import (
    default_prompt_for_cohort_extraction,
    get_llm_model,
    process_pmcid_row_sync
)

logger = initialize_logging()

def default_regex_expression() -> Pattern:
    """
    Returns a compiled regex pattern for cleaning output types by removing non-alphanumeric characters.

    Returns:
        Pattern: A compiled regular expression pattern that matches any character except letters, numbers, underscores, or spaces.
    """
    return re.compile(r'[^a-zA-Z0-9_ ]')

# Compile regex once at module level.
DEFAULT_REGEX: Pattern = default_regex_expression()

def main(api_key: str, df: pd.DataFrame, text_column: str, table_name: str, azure_endpoint: str, api_version: str,
         temperature: float = 0.0, llm_model: Optional[str] = None, spark: Any = None) -> pd.DataFrame:
    """
    Main function to process a DataFrame using LLMs and write results to a Delta table.
    
    Parameters:
        api_key (str): API key for the LLM service.
        df (pd.DataFrame): Input DataFrame containing at least columns "pmcid" and "methods".
        text_column (str): Name of the column containing the text to process.
        table_name (str): Name of the Delta table for storing results.
        azure_endpoint (str): Azure endpoint for LLM.
        api_version (str): API version for the LLM.
        temperature (float, optional): Temperature setting for the LLM. Default is 0.0.
        llm_model (Optional[str], optional): Specific LLM model to use. Default is None.
        spark: Spark session instance.
        
    Returns:
        pd.DataFrame: Aggregated results from the LLM processing.
    """
    # Validate that there is at least one populated row in the 'methods' column.
    if not any(
        (isinstance(value, str) and value.strip() != "") or (value is not None and not isinstance(value, str))
        for value in df["methods"]
    ):
        logger.error("The 'methods' column must have at least one populated row.")
        raise ValueError("The 'methods' column must have at least one populated row.")
    else:
        logger.info("The 'methods' column is properly populated.")

    # Select relevant columns and remove rows with empty 'methods'
    df = df[["pmcid", "methods"]].copy()
    df = df[df["methods"].str.strip().str.len() > 0]

    # Load existing Delta table to filter out already processed PMCIDs
    existing_df = load_existing_delta_data(table_name, spark)
    processed_pmids = get_processed_pmids(existing_df)

    # Filter out rows that have already been processed.
    new_df = df[~df["pmcid"].isin(processed_pmids)].copy()
    total_tasks = new_df.shape[0]
    logger.info("Skipping processing %d PMCID(s) that have already been processed.", len(processed_pmids))
    logger.info("Processing %d new PMCID(s) out of %d total.", total_tasks, df.shape[0])

    # Prepare LLM models using the provided API key and other parameters
    llm_dict = get_llm_model(
        api_key=api_key,
        llm_model=llm_model,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        temperature=temperature
    )

    # Get the input prompt
    base_prompt = default_prompt_for_cohort_extraction()

    # Use the module-level compiled regex pattern
    regex = DEFAULT_REGEX

    all_results = []
    for _, row in new_df.iterrows():
        results = process_pmcid_row_sync(row, llm_dict, base_prompt, text_column, logger, regex)
        all_results.extend(results)

    aggregated_df = pd.DataFrame(all_results)
    # Write aggregated results to a Delta table using Spark
    write_results_to_delta_table(aggregated_df, table_name, spark)

    return aggregated_df

if __name__ == "__main__":
    # Note: For standalone execution, you must supply the required parameters.
    logger.error("The main function requires specific parameters to run.")
