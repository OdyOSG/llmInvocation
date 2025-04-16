import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from langchain_openai import AzureChatOpenAI  # type: ignore

logger = logging.getLogger(__name__)

def default_prompt_for_cohort_extraction() -> str:
    """
    Returns the prompt instructions for extracting cohort details from the Methods/Materials section of a scientific article.

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

def get_llm_model(api_key: str, azure_endpoint: str, api_version: str, temperature: float = 0.0, llm_model: Optional[str] = None) -> Dict[str, Any]:
    """
    Constructs and returns a dictionary of LLM model instances based on the provided parameters.

    Returns:
        Dict[str, Any]: A dictionary mapping model names to their respective AzureChatOpenAI instance.
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
            "deepseek-r1": AzureChatOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                model="deepseek-r1",
                temperature=temperature,
            )
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
    elif llm_model == "deepseek-r1":
        llm_dict = {
            "deepseek-r1": AzureChatOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                model="deepseek-r1",
                temperature=temperature,
            )
        }
    else:
        logger.error("Wrong LLM model selected: %s", llm_model)
        llm_dict = {}
    return llm_dict

def create_prompt(row: Union[Dict[str, Any], Any], input_prompt: str, text_col: str) -> Tuple[str, str]:
    """
    Combines the base prompt with row-specific text and retrieves the PMCID.

    Returns:
        Tuple[str, str]: A tuple containing the PMCID and the combined prompt.
    """
    pmcid: str = row["pmcid"]
    text: str = row.get(text_col, "")
    prompt: str = f"{input_prompt}\n\n{text}"
    return pmcid, prompt

def invoke_llm_sync(
    llm_instance: Any, 
    prompt: str, 
    pmcid: str, 
    llm_name: str, 
    logger: logging.Logger, 
    attempts: int = 5, 
    sleep_time: Union[int, float] = 1
) -> Tuple[Optional[str], Optional[str]]:
    """
    Synchronously calls an LLM instance with a retry mechanism on error.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing the raw LLM output and error message.
    """
    for attempt in range(1, attempts + 1):
        logger.debug("PMCID %s: Attempt %d for LLM '%s'.", pmcid, attempt, llm_name)
        try:
            start = time.time()
            output_obj = llm_instance.invoke(prompt)
            raw_llm_output: str = output_obj.content if hasattr(output_obj, "content") else str(output_obj)
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

def parse_llm_response(
    raw_llm_output: Optional[str], 
    pmcid: str, 
    llm_name: str, 
    prompt: str, 
    error_log: Optional[str], 
    regex: Any
) -> List[Dict[str, Any]]:
    """
    Parses the LLM's raw response into a structured list of result dictionaries.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the parsed response.
    """
    results: List[Dict[str, Any]] = []
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
                category: str = parts[1].strip()
                verbatim_output: str = parts[2].strip()
                interpretation: str = parts[3].strip()
                cleaned_output_type: str = regex.sub("", category)
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

def process_llm_for_pmcid_sync(
    row: Union[Dict[str, Any], Any], 
    llm_name: str, 
    llm_instance: Any, 
    input_prompt: str, 
    text_col: str, 
    logger: logging.Logger, 
    regex: Any
) -> List[Dict[str, Any]]:
    """
    Processes a single row for a specific LLM synchronously and returns the parsed result.

    Returns:
        List[Dict[str, Any]]: A list of result dictionaries.
    """
    pmcid, prompt = create_prompt(row, input_prompt, text_col)
    logger.info("Starting processing for PMCID: %s with LLM: %s", pmcid, llm_name)
    raw_llm_output, error_log = invoke_llm_sync(llm_instance, prompt, pmcid, llm_name, logger)
    results = parse_llm_response(raw_llm_output, pmcid, llm_name, prompt, error_log, regex)
    logger.info("Finished processing for PMCID: %s with LLM: %s", pmcid, llm_name)
    return results

def process_pmcid_row_sync(
    row: Union[Dict[str, Any], Any], 
    llm_dict: Dict[str, Any], 
    input_prompt: str, 
    text_col: str, 
    logger: logging.Logger, 
    regex: Any
) -> List[Dict[str, Any]]:
    """
    Processes a single row synchronously across all available LLMs and aggregates their responses.

    Returns:
        List[Dict[str, Any]]: An aggregated list of result dictionaries from all LLMs.
    """
    all_results: List[Dict[str, Any]] = []
    for llm_name, llm_instance in llm_dict.items():
        res = process_llm_for_pmcid_sync(row, llm_name, llm_instance, input_prompt, text_col, logger, regex)
        all_results.extend(res)
    groups: Dict[Tuple[str, str, Optional[str]], Dict[str, Any]] = {}
    for res in all_results:
        key = (res["pmcid"], res["llm"], res["output_type"])
        if key not in groups:
            groups[key] = {"verbatim_outputs": set(), "interpretations": set(), "base": res}
        groups[key]["verbatim_outputs"].add(res["verbatim_output"])
        groups[key]["interpretations"].add(res["interpretation"])
    aggregated_results: List[Dict[str, Any]] = []
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
