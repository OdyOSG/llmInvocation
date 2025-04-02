def main(
  api_key, 
  df,
  columnName,
  tableName,
  llmModel=None,
  spark=None
):
    # All necessary imports are done within the function
    import re
    import pandas as pd
    from .data_management import (
        initialize_logging,
        load_existing_delta_data,
        write_results_to_delta_table,
        get_processed_pmids
    )
    from .llm_invocation import (
        defaultPromptForCohortExtraction,
        getLLMmodel,
        process_pmcid_row_sync
    )
    
    logger = initialize_logging()
    
    # Load initial Delta table to filter out already processed PMCID values
    existing_df = load_existing_delta_data(table_name=tableName, spark=spark)
    processed_pmids = get_processed_pmids(existing_df)
    
    # Filter out rows that have already been processed.
    new_df = df[~df["pmcid"].isin(processed_pmids)].copy()
    total_tasks = new_df.shape[0]
    logger.info(f"Skipping processing {processed_pmids} PMCID(s) that have already been processed.")
    logger.info(f"Processing {total_tasks} new PMCID(s) out of {df.shape[0]} total.")
    
    # Prepare LLM models using the provided API key
    llm_dict = getLLMmodel(api_key=api_key, llmModel=llmModel)
    
    # Get the input prompt
    base_prompt = defaultPromptForCohortExtraction()
    
    # Compile a regex to clean output types (e.g., remove non-alphanumeric characters)
    regex = re.compile(r'[^a-zA-Z0-9_ ]')
    
    
    all_results = []
    for _, row in df.iterrows():
        results = process_pmcid_row_sync(row, llm_dict, base_prompt, columnName, logger, regex)
        all_results.extend(results)
    
    # Write aggregated results to a Delta table using Spark:
    merged_df = write_results_to_delta_table(pd.DataFrame(all_results), tableName, spark)
    
    # For demonstration, print the results
    for res in all_results:
        print(res)

if __name__ == "__main__":
    main(api_key)
