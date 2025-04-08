from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm_invocation",
    version="0.0.1",
    author="",
    author_email="",
    description="LLM engine to process results of the PubMed Search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OdyOSG/llmInvocation",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pyspark",
        "numpy",
        "pandas",
        "langchain_openai",
        "requests",
        "XlsxWriter",
        "IPython"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="LLM PubMed data-management AI",
    license="MIT",
    python_requires=">=3.8",
)
