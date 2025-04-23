import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from langchain_openai import AzureChatOpenAI  # type: ignore

logger = logging.getLogger(__name__)

def default_prompt_for_cohort_extraction() -> str:
    """
    Returns the prompt instructions for extracting cohort details from the Methods/Materials section of a scientific article.

    Returns:
        str: A formatted string containing the detailed prompt instructions to extract 30 specific categories in a markdown table.
    """
    prompt = """
    **SYSTEM INSTRUCTIONS (FOLLOW EXACTLY)**
    1. You are given the Methods/Materials section of a scientific article describing real-world patient selection and cohort creation.
    2. You must extract information into exactly 30 specific categories (listed below). Each category must correspond to one row in **a single markdown table**.
    3. You may **not** add or remove any categories beyond those specified.
    4. For each category, do the following:
       - Search the provided text to retrieve relevant direct quotes (i.e., the exact wording).
       - If multiple relevant pieces appear, combine them into one string **separated by line breaks** (e.g., using `\n`).
       - **Escape any vertical bar `|` inside the quote as `\\|`, and escape inner double-quotes `"` as `\"` so the markdown table renders correctly.**  
       - Place that combined string in the **verbatim** column (still enclosed in double quotes if you wish).
       - In the **interpretation** column, briefly interpret or clarify the content for that category.
    5. If a category is **mentioned** but does **not** include any direct quotes, place `""` in the **verbatim** column and briefly explain in the **interpretation** column.
       If a category is **not mentioned at all**, place `""` in the **verbatim** column and write “Not mentioned.” in the **interpretation** column.
    6. **Special case for `medical_codes`**:
       - If ICD, SNOMED, CPT-4, HCPCS, ATC, RxNorm, or LOINC codes are present, place them (in quotes) under **verbatim** and set **interpretation** to `codes_reported = Yes.` 
       - If none, set **verbatim** = `""` and **interpretation** = `codes_reported = No.` 
    7. **OUTPUT FORMAT** — You must return **only** one markdown table with the following columns:
       - A header row exactly like this: 
         `| category | verbatim | interpretation |`
       - A divider row exactly like this: 
         `|----------|----------|----------------|`
       - **The 30 body rows must appear in the exact order listed in Section 8 below. Do not reorder, omit, or add rows.**  
       - **No numbering, code blocks, or extra text** of any kind before or after the table.
    8. **CATEGORIES** (one row per item, fixed order):
        *Source Population & Eligibility*
        1. **population** – Overall source population frame (e.g., “all adults with ≥1 encounter in XXX database”).
        2. **demographic_restriction** – A priori filters on age (e.g., ≥18 y), sex, race/ethnicity, socioeconomic status, or other human attributes (a.k.a. eligibility strata, sampling frame limits).
        3. **entry_event** – The recorded trigger (“qualifying event,” “anchor,” “index encounter”)—such as a diagnosis code, procedure, prescription, or lab result—that grants cohort membership and pins the timeline.
        4. **index_date_definition** – Rule converting the entry event (or nearest related record) into a single calendar date (“time 0”, “t₀”) from which exposure, washout, and follow-up windows are calculated.
        5. **indication** – Disease or clinical scenario motivating exposure (e.g., “type 2 diabetes”).
        6. **inclusion_rule** – Additional eligibility clauses (e.g., ≥365 days continuous enrollment, specific clinical history) that must be satisfied **after** the entry event to remain in the analytic cohort.
        7. **exclusion_rule** – Disqualifying conditions at or before index (e.g., prior outcome, contraindicated therapy) that remove otherwise eligible individuals; synonyms: “pre-index exclusion,” “restriction criterion.”
        8. **exit_criterion** – Pre-specified events or cut-offs (death, disenrollment, treatment stop, database end, fixed k-years) that censor or terminate person-time.
        9. **attrition_criteria** – The ordered set of screening steps (often visualized in a CONSORT-style flow diagram) with counts of participants dropped at each stage (“N after filter”).
        10. **medical_codes** – Any structured alphanumeric vocabulary that maps clinical concepts (e.g., ICD-10 “E11.\*”, SNOMED CT 22298006, CPT-4 “99213”, HCPCS “J3490”, ATC “A10BA02”, RxNorm (drugs) and LOINC (labs)); often grouped into “code lists” or “value sets” and used to flag diagnoses, procedures, or drugs in electronic data.
        11. **study_period** – Inclusive calendar bounds (start YYYY-MM-DD to end YYYY-MM-DD) within which entry events, exposures, and outcomes are eligible; distinguishes historical look-back from prospective accrual.
        12. **washout_period** – A look-back period or interval (commonly 6–12 months) with **no** exposure or outcome records, ensuring incident/new-user status and mitigating left-truncation bias.
        13. **follow_up_period** – Observational span **after** index during which outcomes are accrued; may be fixed (e.g., 180 days) or variable until an exit criterion (“open-ended follow-up”).
        14. **time_at_risk** – Window during which outcomes are attributed (e.g., “index to 365 days”).
        15. **exposure_definition** – Operational logic translating raw records into an exposure flag: drug name(s), dose, formulation, days’ supply, permissible gaps (“grace period”), route, or device identifiers. Include medical codes when available.
        16. **treatment_definition** – Granular description of the therapeutic regimen (line of therapy, combination rules, titration schedule) apart from simple exposure status.  Include medical codes when available.
        17. **comparator_cohort** – Reference group (active comparator drug, placebo, usual-care, external control) constructed under parallel rules to the exposure cohort, enabling relative effect estimation. Include medical codes when available.
        18. **outcome_definition** – Case-finding algorithm using codes, laboratory thresholds, narrative NLP, or chart review triggers to capture the event of interest with temporal rule logic if specified; may specify positive predictive value (PPV) when known.
        19. **severity_definition** – Quantitative or categorical grading of disease or adverse-event intensity (e.g., NIHSS > 15 = severe stroke, CTCAE Grade 3-4), sometimes derived from composite scores.
        20. **outcome_ascertainment** – Source(s) and workflow for verifying outcome events—claims linkage, EHR abstraction, blinded adjudication committee—including any lag windows for data maturation.
        21. **algorithm_validation** – Empirical assessment of code-based algorithms against a gold standard (chart review, registry) reporting PPV, sensitivity, specificity, or F-measure.
        22. **study_design** – High-level analytic or study design architecture (retrospective cohort, new-user active-comparator, case-control, SCCS, self-controlled risk-interval) that dictates temporality and confounding mitigation strategy. 
        23. **covariate_adjustment** – Confounding control via multivariable regression, propensity score matching/weighting, high-dimensional PS, IPTW, or doubly robust estimators; usually has covariate time window (baseline vs. time-varying).
        24. **statistical_analysis** – Core modeling and inference plan (e.g., Cox proportional-hazards with robust variance, Poisson regression for rates, Kaplan–Meier for survival) plus software/versions.
        25. **sensitivity_analysis** – Pre-planned perturbations (vary washout length, redefine exposure, lag outcome, exclude early events) to test robustness; may include quantitative bias analysis or negative controls.
        26. **data_source_type** – Macro-level classification of the underlying repository (administrative claims, integrated EHR network, disease registry, national survey, multi-country distributed network). 
        27. **data_provenance** – End-to-end lineage of data elements: original source, extraction, transformation logic (ETL), versioning, audit trails—supporting reproducibility and traceability.
        28. **healthcare_setting** – Care context reflected in the records (primary care clinic, inpatient hospital, ED, specialty outpatient, integrated delivery system), affecting coding density and capture completeness.
        29. **data_access** – Statement of availability and governance (open data portal, public use file, data use agreement, secure enclave, upon reasonable request) specifying who may obtain the analytic dataset.
        30. **ethics_approval** – Documentation of IRB/REC review or exemption, with protocol ID, governing regulations (e.g., 45 CFR 46), and affirmation of Declaration of Helsinki adherence.
       31. **funding_source** – Grant numbers, industry sponsorship, “unrestricted educational grant.”  
       32. **conflict_of_interest** – Author disclosures, “consulting fees,” “honoraria,” equity holdings.  
       33. **author_affiliations** – Country/region clues, multisite collaborations.  
       34. **study_objective** – Primary aim / hypothesis, “to estimate,” “to compare.”  
       35. **rationale** – Clinical or policy motivation, gap statement.  
       36. **sample_size_calculation** – Power analysis, assumed effect size, Type I error.  
       37. **randomization_scheme** – For randomized RWE hybrids; block, stratified, simple.  
       38. **blinding_masking** – Participant, investigator, assessor blinding status.  
       39. **data_linkage_method** – Probabilistic vs deterministic linkage across sources.  
       40. **missing_data_handling** – Multiple imputation, complete-case, LOCF.  
       41. **quality_control** – Data cleaning rules, duplicate removal, range checks.  
       42. **time_at_risk_definition** – Exact start/stop logic distinct from follow-up window (“risk window” for SCCS).  
       43. **immortal_time_handling** – Methods to avoid immortal-time bias (time-dependent exposure coding, landmarking).  
       44. **negative_control_exposures_outcomes** – Specified falsification endpoints or exposures.  
       45. **causal_framework** – Target-trial emulation, causal diagram, G-methods notation.
       46. **software_environment** – R/4.3.2, SAS v9.4, Python/3.11 with packages.  
       47. **multiple_testing_correction** – Bonferroni, FDR, Holm.  
       48. **heterogeneity_assessment** – I², Cochran’s Q, subgroup interaction terms.  
       49. **instrumental_variable** – Relevance/validity checks, F-statistics.  
       50. **effect_measure_modification** – Pre-specified interaction terms, stratified analyses.  
       51. **summary_baseline_characteristics** – Table 1 demographics/comorbidities snippet.  
       52. **effect_estimates** – HRs, ORs, RRs, RD with 95 % CI.  
       53. **absolute_risk_numbers** – Incidence rates per 1 000 PY, NNT, NNH.  
       54. **adverse_events_profile** – Safety outcomes, SAE counts.  
       55. **event_counts_flow** – Number at risk, events, censoring, KM curve data.  
       56. **limitations** – Internal/external validity threats (“residual confounding,” “coding error”).  
       57. **generalizability** – Applicability to other settings/populations.  
       58. **future_research** – Suggested next steps, unmet needs.


    9. **PROHIBITED**:
       - Do not add any extra rows for categories not listed.
       - Do not produce any text outside the markdown table.
       - Do not include numbering, code fences, or extraneous markdown elements.
    **USER PROMPT**:
       Extract the relevant details from the following text based on the categories above. Provide each category as one row—in the fixed order—within the single markdown table described, following the rules exactly.
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
