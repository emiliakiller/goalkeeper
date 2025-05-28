import json
import requests

from ollama import chat, ChatResponse
from pydantic import BaseModel, Field

# Install duckduckgo-search for this example:
# !pip install -U duckduckgo-search

# from langchain.tools import DuckDuckGoSearchRun

# search_tool = DuckDuckGoSearchRun()


def get_weather(latitude:float, longitude:float):
    """
    This is a publically available API that returns the weather for a given location.

    Args:
      latitude: The location's latitude as a float, negative numbers indicate South of the Equator
      longitude: The location's longitude as a float, negative numbers indicate West of the Prime Meridian

    Returns:
      Any: The details of the current weather
    """

    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=apparent_temperature,precipitation_probability,cloud_cover,wind_speed_10m"
    )
    data = response.json()
    return data["current"]


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current apparent temperature in celsius, precipitation probability, cloud cover, and wind speed for provided coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

class WeatherResponse(BaseModel):
    response: str = Field(
        description="A natural language response to the user's question."
    )
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    location:str = Field(
        description="The name of the location"
    )
    date:str = Field(
        description="The date of the weather data"
    )

system_prompt = "You are a helpful weather assistant. Remember that latitudes (North/South) South of the Equator are negative, and longitudes (East/West) West of the Greenwich Meridian are negative. Make certain you use the correct latitude and longitude"
input_query = "What's the weather like in Paris currently?" # input("Please enter your query: ")

messages = [
    { 'role': 'system', 'content': system_prompt },
    { 'role': 'user', 'content': input_query },
]

available_functions = {
  'get_weather': get_weather,
}

completion = chat(model='llama3.2', messages=messages, tools=tools
# format = Weather.model_json_schema()
)

print(completion.message)
messages.append(completion.message.__dict__)

for tool in completion.message.tool_calls or []:

    function_to_call = available_functions.get(tool.function.name)
    if function_to_call:
        result = function_to_call(**tool.function.arguments)
        messages.append(
            {"role": "tool", "tool_name": tool.function.name, "content": json.dumps(result)}
        )
    else:
        print('Function not found:', tool.function.name)


completion = chat(model='llama3.2', messages=messages, tools=tools,
    format = WeatherResponse.model_json_schema()
)

output = WeatherResponse.model_validate_json(completion.message.content)
# output = completion.message.content
print(output)
