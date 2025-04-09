import logging
import time
from langchain_openai import AzureChatOpenAI  # type: ignore

logger = logging.getLogger(__name__)


def default_prompt_for_cohort_extraction():
    """
    Returns the prompt instructions for extracting cohort details from the Methods/Materials section of a scientific article.

    Parameters:
        None

    Returns:
        str: A formatted string containing the detailed prompt instructions to extract 27 specific categories in a markdown table.
    """
    prompt = """
    **SYSTEM INSTRUCTIONS (FOLLOW EXACTLY)**
    1. You are given the Methods/Materials section of a scientific article describing real-world patient selection and cohort creation.
    2. You must extract information into exactly 27 specific categories (listed below). Each category must correspond to one row in **a single markdown table**.
    3. You may **not** add or remove any categories beyond those specified.
    4. For each category, do the following:
       - Search the provided text to retrieve relevant direct quotes (i.e., the exact wording).
       - If multiple relevant pieces appear, combine them into one string **separated by line breaks** (e.g., using `\n`).
       - Place that combined string in the **verbatim** column (still enclosed in double quotes if you wish).
       - In the **interpretation** column, briefly interpret or clarify the content for that category.
    5. If a category is **mentioned** but does **not** include any direct quotes, place `""` in the **verbatim** column and briefly explain in the **interpretation** column.
       If a category is **not mentioned at all**, place `""` in the **verbatim** column and write “Not mentioned.” in the **interpretation** column.
    6. **Special case for `medical_codes`**:
       - If ICD, SNOMED, CPT-4, HCPCS, or ATC codes are present, place them (in quotes) under **verbatim** and set **interpretation** to `codes_reported = Yes.` 
       - If none, set **verbatim** = `""` and **interpretation** = `codes_reported = No.` 
    7. **OUTPUT FORMAT** — You must return **only** one markdown table with the following columns:
       - A header row exactly like this: 
         `| category | verbatim | interpretation |`
       - A divider row exactly like this: 
         `|----------|----------|----------------|`
       - One row for each of the 27 categories below.
       - **No numbering, code blocks, or extra text** of any kind before or after the table.
    8. **CATEGORIES** (one row per item, in any order):
       - medical_codes 
       - demographic_restriction 
       - entry_event 
       - index_date_definition 
       - inclusion_rule 
       - exclusion_rule 
       - exit_criterion 
       - attrition_criteria 
       - washout_period 
       - follow_up_period 
       - exposure_definition 
       - treatment_definition 
       - outcome_definition 
       - severity_definition 
       - outcome_ascertainment 
       - study_period 
       - study_design 
       - comparator_cohort 
       - covariate_adjustment 
       - statistical_analysis 
       - sensitivity_analysis 
       - algorithm_validation 
       - data_provenance 
       - data_source_type 
       - healthcare_setting 
       - data_access 
       - ethics_approval 
    9. **PROHIBITED**:
       - Do not add any extra rows for categories not listed.
       - Do not produce any text outside the markdown table.
       - Do not include numbering, code fences, or extraneous markdown elements.
    **USER PROMPT**:
       Extract the relevant details from the following text based on the categories above. Provide each category as one row in the single markdown table described, following the rules exactly.
    """
    return prompt.strip()


def get_llm_model(api_key, azure_endpoint, api_version, temperature=0.0, llm_model=None):
    """
    Constructs and returns a dictionary of LLM model instances based on the provided parameters.

    Parameters:
        api_key (str): API key for the LLM service.
        azure_endpoint (str): Azure endpoint for the LLM service.
        api_version (str): API version for the LLM service.
        temperature (float, optional): Temperature setting for the LLM. Default is 0.0.
        llm_model (str, optional): Specific LLM model to use ("claude" or "gpt-4o-full"). Default is None.

    Returns:
        dict: A dictionary mapping model names (str) to their respective AzureChatOpenAI instance.
    """
    if llm_model is None:
        llm_dict = {
            "claude_sonnet": AzureChatOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                model="anthropic.claude-v3-sonnet",
                temperature=temperature,
            ),
            "gpt-4o-full": AzureChatOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                model="gpt-4o",
                temperature=temperature,
            ),
        }
    elif llm_model == "claude":
        llm_dict = {
            "claude_sonnet": AzureChatOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                model="anthropic.claude-v3-sonnet",
                temperature=temperature,
            )
        }
    elif llm_model == "gpt-4o-full":
        llm_dict = {
            "gpt-4o-full": AzureChatOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                model="gpt-4o",
                temperature=temperature,
            )
        }
    else:
        logger.error("Wrong LLM model selected: %s", llm_model)
        llm_dict = {}
    return llm_dict


def create_prompt(row, input_prompt, text_col):
    """
    Combines the base prompt with row-specific text and retrieves the PMCID.

    Parameters:
        row (dict or pandas.Series): Data record containing at least a 'pmcid' key and the text column specified by text_col.
        input_prompt (str): Base prompt string.
        text_col (str): The column name in 'row' that contains additional text to append to the prompt.

    Returns:
        tuple: A tuple containing:
            - pmcid (str): The PMCID extracted from the row.
            - prompt (str): The combined prompt consisting of the base prompt and the extracted text.
    """
    pmcid = row["pmcid"]
    text = row.get(text_col, "")
    prompt = f"{input_prompt}\n\n{text}"
    return pmcid, prompt


def invoke_llm_sync(llm_instance, prompt, pmcid, llm_name, logger, attempts=5, sleep_time=1):
    """
    Synchronously calls an LLM instance with a retry mechanism on error.

    Parameters:
        llm_instance: LLM model instance that has an 'invoke' method.
        prompt (str): The prompt text to be sent to the LLM.
        pmcid (str): Identifier for the document/record being processed.
        llm_name (str): Name of the LLM model being called.
        logger (logging.Logger): Logger instance for logging progress and errors.
        attempts (int, optional): Number of retry attempts. Default is 5.
        sleep_time (int or float, optional): Time in seconds to wait between retry attempts. Default is 1.

    Returns:
        tuple: A tuple containing:
            - raw_llm_output (str or None): The output returned by the LLM, or None if unsuccessful.
            - error_log (str or None): The error message if an error occurred; otherwise, None.
    """
    for attempt in range(1, attempts + 1):
        logger.debug("PMCID %s: Attempt %d for LLM '%s'.", pmcid, attempt, llm_name)
        try:
            start = time.time()
            output_obj = llm_instance.invoke(prompt)
            raw_llm_output = (
                output_obj.content if hasattr(output_obj, "content") else str(output_obj)
            )
            elapsed = time.time() - start
            logger.debug(
                "PMCID %s: LLM '%s' succeeded in %.2f sec on attempt %d.",
                pmcid,
                llm_name,
                elapsed,
                attempt,
            )
            return raw_llm_output, None
        except Exception as e:
            logger.exception("PMCID %s: LLM '%s' error on attempt %d: %s", pmcid, llm_name, attempt, e)
            time.sleep(sleep_time)
    return None, str(e)


def parse_llm_response(raw_llm_output, pmcid, llm_name, prompt, error_log, regex):
    """
    Parses the LLM's raw response into a structured list of result dictionaries.

    Parameters:
        raw_llm_output (str or None): The raw string output from the LLM.
        pmcid (str): Identifier for the processed document.
        llm_name (str): Name of the LLM model used.
        prompt (str): The prompt that was provided to the LLM.
        error_log (str or None): Error log text if an error occurred; otherwise, None.
        regex (Pattern): A compiled regular expression object used to clean the output text.

    Returns:
        list: A list of dictionaries, each containing keys such as "pmcid", "llm", "output_type",
              "verbatim_output", "interpretation", "original_llm_output", "input_prompt",
              "llm_raw_llm_output", and "error_log".
    """
    results = []
    if raw_llm_output and not error_log:
        for line in raw_llm_output.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|")
            if len(parts) == 5:
                header_keywords = ["category", "verbatim", "interpretation"]
                if any(keyword in parts[1].lower() for keyword in header_keywords):
                    continue
                if set(parts[2].strip()) == {"-"} or set(parts[3].strip()) == {"-"}:
                    continue
                category = parts[1].strip()
                verbatim_output = parts[2].strip()
                interpretation = parts[3].strip()
                cleaned_output_type = regex.sub("", category)
                results.append({
                    "pmcid": pmcid,
                    "llm": llm_name,
                    "output_type": cleaned_output_type,
                    "verbatim_output": verbatim_output,
                    "interpretation": interpretation,
                    "original_llm_output": raw_llm_output,
                    "input_prompt": prompt,
                    "llm_raw_llm_output": raw_llm_output,
                    "error_log": error_log,
                })
    if not results:
        results.append({
            "pmcid": pmcid,
            "llm": llm_name,
            "output_type": None,
            "verbatim_output": raw_llm_output.strip() if raw_llm_output else "",
            "interpretation": "",
            "original_llm_output": raw_llm_output,
            "input_prompt": prompt,
            "llm_raw_llm_output": raw_llm_output,
            "error_log": error_log,
        })
    return results


def process_llm_for_pmcid_sync(row, llm_name, llm_instance, input_prompt, text_col, logger, regex):
    """
    Processes a single row for a specific LLM synchronously and returns the parsed result.

    Parameters:
        row (dict or pandas.Series): Data record containing at least a 'pmcid' key and text under the column specified by text_col.
        llm_name (str): The name of the LLM model to be used.
        llm_instance: Instance of the LLM model with an 'invoke' method.
        input_prompt (str): Base prompt string to be combined with row-specific text.
        text_col (str): The column name in 'row' that contains additional text for the prompt.
        logger (logging.Logger): Logger instance for logging progress and errors.
        regex (Pattern): A compiled regular expression object used for cleaning the LLM output.

    Returns:
        list: A list of result dictionaries produced after processing the row with the specified LLM.
    """
    pmcid, prompt = create_prompt(row, input_prompt, text_col)
    logger.info("Starting processing for PMCID: %s with LLM: %s", pmcid, llm_name)
    raw_llm_output, error_log = invoke_llm_sync(llm_instance, prompt, pmcid, llm_name, logger)
    results = parse_llm_response(raw_llm_output, pmcid, llm_name, prompt, error_log, regex)
    logger.info("Finished processing for PMCID: %s with LLM: %s", pmcid, llm_name)
    return results


def process_pmcid_row_sync(row, llm_dict, input_prompt, text_col, logger, regex):
    """
    Processes a single row synchronously across all available LLMs and aggregates their responses.

    Parameters:
        row (dict or pandas.Series): Data record containing a 'pmcid' and text under the column specified by text_col.
        llm_dict (dict): Dictionary mapping LLM names (str) to their corresponding LLM model instances.
        input_prompt (str): Base prompt string to be combined with row-specific text.
        text_col (str): The column name in 'row' that contains text for the prompt.
        logger (logging.Logger): Logger instance for logging progress and errors.
        regex (Pattern): A compiled regular expression object used for cleaning each LLM output.

    Returns:
        list: An aggregated list of result dictionaries from all LLMs. Each dictionary contains keys such as "pmcid",
              "llm", "output_type", "verbatim_output", "interpretation", "original_llm_output",
              "input_prompt", "llm_raw_llm_output", and "error_log".
    """
    all_results = []
    for llm_name, llm_instance in llm_dict.items():
        res = process_llm_for_pmcid_sync(row, llm_name, llm_instance, input_prompt, text_col, logger, regex)
        all_results.extend(res)
    groups = {}
    for res in all_results:
        key = (res["pmcid"], res["llm"], res["output_type"])
        if key not in groups:
            groups[key] = {"verbatim_outputs": set(), "interpretations": set(), "base": res}
        groups[key]["verbatim_outputs"].add(res["verbatim_output"])
        groups[key]["interpretations"].add(res["interpretation"])
    aggregated_results = []
    for key, value in groups.items():
        base = value["base"]
        aggregated_results.append({
            "pmcid": base["pmcid"],
            "llm": base["llm"],
            "output_type": base["output_type"],
            "verbatim_output": "\n".join(value["verbatim_outputs"]),
            "interpretation": "\n".join(value["interpretations"]),
            "original_llm_output": base["original_llm_output"],
            "input_prompt": base["input_prompt"],
            "llm_raw_llm_output": base["llm_raw_llm_output"],
            "error_log": base["error_log"],
        })
    return aggregated_results
