def defaultPromptForCohortExtraction():
    """
    Returns the prompt instructions for extracting cohort details from the Methods/Materials section
    of a scientific article. The prompt instructs the LLM to parse the text into 27 specific categories,
    each corresponding to one row in a single markdown table.
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


def getLLMmodel(
  api_key, 
  azure_endpoint, 
  api_version, 
  temperature=0.0, 
  llm_model=None
):
  
    from langchain_openai import AzureChatOpenAI
    
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
        print("Wrong LLM model selected.")
        llm_dict = {}
    return llm_dict



def create_prompt(row, input_prompt, text_col):
    """
    Combines the base prompt with row-specific text.
    """
    pmcid = row["pmcid"]
    text = row.get(text_col, "")
    prompt = f"{input_prompt}\n\n{text}"
    return pmcid, prompt


def invoke_llm_sync(llm_instance, prompt, pmcid, llm_name, logger, attempts=5, sleep_time=1):
    """
    Synchronously calls an LLM instance with retries on error.
    """
    import time
    for attempt in range(1, attempts + 1):
        logger.debug(f"PMCID {pmcid}: Attempt {attempt} for LLM '{llm_name}'.")
        try:
            start = time.time()
            output_obj = llm_instance.invoke(prompt)
            raw_llm_output = output_obj.content if hasattr(output_obj, "content") else str(output_obj)
            elapsed = time.time() - start
            logger.debug(f"PMCID {pmcid}: LLM '{llm_name}' succeeded in {elapsed:.2f} sec on attempt {attempt}.")
            return raw_llm_output, None
        except Exception as e:
            error_msg = str(e)
            logger.error(f"PMCID {pmcid}: LLM '{llm_name}' error on attempt {attempt}: {error_msg}")
            time.sleep(sleep_time)
    return None, error_msg


def call_llm_sync(llm_instance, prompt, pmcid, llm_name, logger, attempts=5, sleep_time=1):
    """
    Wrapper for synchronous LLM invocation.
    """
    return invoke_llm_sync(llm_instance, prompt, pmcid, llm_name, logger, attempts, sleep_time)


def parse_llm_response(raw_llm_output, pmcid, llm_name, prompt, error_log, regex):
    """
    Parses the LLM's response into structured output.
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
    Processes a single row for a specific LLM synchronously.
    """
    pmcid, prompt = create_prompt(row, input_prompt, text_col)
    logger.info(f"Starting processing for PMCID: {pmcid} with LLM: {llm_name}")
    raw_llm_output, error_log = call_llm_sync(llm_instance, prompt, pmcid, llm_name, logger)
    results = parse_llm_response(raw_llm_output, pmcid, llm_name, prompt, error_log, regex)
    logger.info(f"Finished processing for PMCID: {pmcid} with LLM: {llm_name}")
    return results


def process_pmcid_row_sync(row, llm_dict, input_prompt, text_col, logger, regex):
    """
    Processes a single row synchronously for all LLMs and aggregates the responses.
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
