from ollama import chat
from ollama import ChatResponse

# Install duckduckgo-search for this example:
# !pip install -U duckduckgo-search

# from langchain.tools import DuckDuckGoSearchRun

# search_tool = DuckDuckGoSearchRun()

if __name__ == "__main__":
    print("## This is a placeholder")
    print("-------------------------------")
    input_query = input("Please enter your query: ")

    response: ChatResponse = chat(model='llama3.2', messages=[
    {
      'role': 'system',
      'content': 'You are a helpful assistant. You answer questions clearly, consisely, and to the best of your ability'
    },
    {
        'role': 'user',
        'content': input_query,
    },
    ],
    stream = True
    )

    for chunk in response:
      print(chunk.message.content, end='', flush=True)
