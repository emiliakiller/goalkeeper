"""This file covers the retrieval section of the followed tutorial. It is here solely for practice, and will be deleted when appropriate"""

import json
import os

from ollama import chat
from pydantic import BaseModel, Field

# --------------------------------------------------------------
# Define the knowledge base retrieval tool
# --------------------------------------------------------------


def search_kb(question: str):
    """
    Load the whole knowledge base from the JSON file.
    (This is a mock function for demonstration purposes, we don't search)
    """
    with open("kb.json", "r") as f:
        return json.load(f)


# --------------------------------------------------------------
# Step 1: Call model with search_kb tool defined
# --------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Get the answer to the user's question from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the return policy?"},
]

completion = chat(
    model="llama3.2",
    messages=messages,
    tools=tools,
)

# --------------------------------------------------------------
# Step 2: Model decides to call function(s)
# --------------------------------------------------------------

completion.model_dump()

# --------------------------------------------------------------
# Step 3: Execute search_kb function
# --------------------------------------------------------------


def call_function(name, args):
    if name == "search_kb":
        return search_kb(**args)


for tool_call in completion.message.tool_calls or []:
    name = tool_call.function.name
    args = tool_call.function.arguments
    messages.append(completion.message.__dict__)

    result = call_function(name, args)
    messages.append(
        {"role": "tool", "tool_call_name": name, "content": json.dumps(result)}
    )

# --------------------------------------------------------------
# Step 4: Supply result and call model again
# --------------------------------------------------------------


class KBResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    source: int = Field(description="The record id of the answer.")


completion_2 = chat(
    model="llama3.2",
    messages=messages,
    tools=tools,
    format=KBResponse.model_json_schema(),
)

# --------------------------------------------------------------
# Step 5: Check model response
# --------------------------------------------------------------

final_response = KBResponse.model_validate_json(completion_2.message.content)
print(final_response.source)
print(final_response.answer)

# --------------------------------------------------------------
# Question that doesn't trigger the tool
# --------------------------------------------------------------

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the weather in Tokyo?"},
]

completion_3 = chat(
    model="llama3.2",
    messages=messages,
    tools=tools,
)

print(completion_3.message.content)
