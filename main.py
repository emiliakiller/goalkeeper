from ollama import chat
from ollama import ChatResponse
from pydantic import BaseModel

# Install duckduckgo-search for this example:
# !pip install -U duckduckgo-search

# from langchain.tools import DuckDuckGoSearchRun

# search_tool = DuckDuckGoSearchRun()

class CalendarEvent(BaseModel):
    event_name:str
    date:str
    participants:list[str]

if __name__ == "__main__":
    print("## This is a placeholder")
    print("-------------------------------")
    input_query = "Alice and Bob are going to the science fair on Friday" # input("Please enter your query: ")

    response: ChatResponse = chat(model='llama3.2', messages=[
    {
      'role': 'system',
      'content': 'Extract the event information.'
    },
    {
        'role': 'user',
        'content': input_query,
    },
    ],
    format = CalendarEvent.model_json_schema()
    )

    event = CalendarEvent.model_validate_json(response.message.content)
    print(event)
