# llm_invocation

**llm_invocation** is a Python package that provides an LLM engine to process PubMed search results. It is designed to work with DataFrames containing scientific article methods sections, leveraging Azure-based LLM models (via langchain_openai) to extract structured information. The package integrates with Apache Spark to read and write Delta tables, ensuring efficient data management and scalable processing.

## Features

- **Data Management:**  
  Functions for loading existing Delta table data, writing aggregated results to a new Delta table, and tracking processed identifiers (PMCIDs).  

- **LLM Invocation:**  
  A set of functions to build prompts, invoke AzureChatOpenAI models synchronously with retry logic, and parse the responses into structured output.  

- **Logging Configuration:**  
  Customized logging setup with filters to allow important information (status messages and errors) to be logged while minimizing verbosity.  

- **Regex Optimization:**  
  Compiles regex expressions only once at the module level, improving performance by reusing the compiled pattern across function calls.  

## Installation

You can install the package directly from GitHub. To install the latest version from the default branch, run:

    pip install git+https://github.com/OdyOSG/llmInvocation.git

If you need to install from a specific branch (for example, `feature-branch`), use:

    pip install git+https://github.com/OdyOSG/llmInvocation.git@feature-branch

The package requires Python 3.8 or later and depends on libraries such as pyspark, numpy, pandas, and langchain_openai among others. For full dependency details, see the [setup.py](./setup.py).  

## Usage

After installation, you can use the package by importing its functions in your Python scripts or notebooks. The package’s main entry point is the `main` function, which processes a given DataFrame and writes the aggregated LLM output to a Delta table.

### Example

Below is an example of how you might use the package:

    import pandas as pd
    from pyspark.sql import SparkSession
    from llm_invocation import main

    # Create a Spark session
    spark = SparkSession.builder \
        .appName("LLMInvocationExample") \
        .getOrCreate()

    # Example DataFrame containing PubMed search results.
    data = {
        "pmcid": ["PMC123456", "PMC234567"],
        "methods": [
            "This is the methods section of the first article.",
            "This is the methods section of the second article."
        ]
    }
    df = pd.DataFrame(data)

    # Define required parameters
    api_key = "YOUR_API_KEY"
    text_column = "methods"
    table_name = "pubmed_results"
    azure_endpoint = "https://your-azure-endpoint"
    api_version = "2021-06-01"
    temperature = 0.0  # or another desired value

    # Run the LLM processing
    results_df = main(
        api_key=api_key,
        df=df,
        text_column=text_column,
        table_name=table_name,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        temperature=temperature,
        spark=spark
    )

    print(results_df)

> **Note:**  
> When running as a standalone script (i.e. via `python main.py`), the package logs an error noting that the required parameters must be provided. It is designed to be used programmatically or as part of a larger data pipeline.  

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request on the [GitHub repository](https://github.com/OdyOSG/llmInvocation).

## License

This project is licensed under the MIT License. For more details, refer to the [LICENSE](LICENSE) file.
