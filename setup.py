from setuptools import setup, find_packages

# Standard Library Imports
import os
import re
import time
import xml.etree.ElementTree as ET


setup(
    name="LLM-engine",  
    version="0.1",          
    author="EPAM",
    author_email="",
    description="LLM engine to process results of the PubMed Search",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    #url="https://github.com/OdyOSG/pubMedSearch/tree/main",  
    packages=find_packages(),  
    install_requires=[
      "pyspark",
      "numpy",
      "pandas",
      "langchain_openai",
      "requests",
      "PubMedFetcher",
      "XlsxWriter",
      "IPython"
    ]
)
