import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Install duckduckgo-search for this example:
# !pip install -U duckduckgo-search

# from langchain.tools import DuckDuckGoSearchRun

# search_tool = DuckDuckGoSearchRun()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# os.environ["OPENAI_ORGANIZATION"] = config("OPENAI_ORGANIZATION_ID")


# This is the main function that you will use to run your application.
if __name__ == "__main__":
    print("## This is a placeholder")
    print("-------------------------------")
    result = input("Please enter your query: ")

    print(result)
