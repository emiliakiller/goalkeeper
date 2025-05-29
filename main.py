from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from ollama import chat
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

model = "llama3.1"

ISO8601_info = "The ISO 8601 date and time format is a standard way to represent dates and times. It follows a specific order: year-month-day-hour:minute:second."


# --------------------------------------------------------------
# Step 1: Define the data models for routing and responses
# --------------------------------------------------------------


class GoalRequestType(BaseModel):
    """Router LLM call: Determine the type of goal request"""

    request_type: Literal["new_goal", "modify_goal", "retrieve_goal", "other"] = Field(
        description="Type of goal request being made"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")
    goal_details: str = Field(description="Cleaned details of the request")


class NewGoalDetails(BaseModel):
    """Details for creating a new goal"""

    goal_name: str = Field(description="Brief description of the goal")
    deadline: datetime = Field(description="Date and time that the goal is due (ISO 8601)")
    priority: int = Field(description="Importance of the goal on a scale from 0 as least priority to 10 as highest priority")
    # component_tasks: list[str] = Field(description="List of tasks that need to be done to achieve this goal")


class Change(BaseModel):
    """Details for changing an existing goal"""

    field: str = Field(description="Field to change")
    new_value: str = Field(description="New value for the field")


class ModifyGoalDetails(BaseModel):
    """Details for modifying an existing goal"""

    goal_identifier: str = Field(
        description="Description to identify the existing goal"
    )
    changes: list[Change] = Field(description="List of changes to make")


class GoalResponse(BaseModel):
    """Final response format"""

    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="User-friendly response message")


# --------------------------------------------------------------
# Step 2: Define the routing and processing functions
# --------------------------------------------------------------

def route_goal_request(user_input: str) -> GoalRequestType:
    """Router LLM call to determine the type of goal request"""
    logger.info("Routing goal request")
    logger.info(f"Input text: {user_input}")

    completion = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Determine if this is a request to create a new goal entry, modify an existing one, or if it's an irrelevant input",
            },
            {"role": "user", "content": user_input},
        ],
        format=GoalRequestType.model_json_schema(),
    )
    result = GoalRequestType.model_validate_json(completion.message.content)
    # result.goal_details = user_input
    logger.info(
        f"Request routed as: {result.request_type} with confidence: {result.confidence_score}. Details passed through: {result.goal_details}"
    )
    return result


# Parse details
# f"{date_context} Extract detailed event information. When dates reference 'next Tuesday' or similar relative dates, use this current date as a reference. Assume all events to be in the future, and to start at mentioned times unless otherwise specified. Format dates and times using ISO8601: {ISO8601_info} Ensure you extract the correct date.",

def handle_new_goal(description: str, events: dict) -> GoalResponse:
    """Process a new goal request"""
    logger.info("Processing new goal request")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %d %B, %Y')}."

    # Get event details
    completion = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"Extract details for creating a new goal entry. {date_context}",
            },
            {"role": "user", "content": description},
        ],
        format=NewGoalDetails.model_json_schema(),
    )
    details = NewGoalDetails.model_validate_json(completion.message.content)

    logger.info(f"New goal: {details.model_dump_json(indent=2)}")
    events.update({"goal_name": details.goal_name, "deadline": details.deadline, "priority": details.priority})

    # Generate response
    return GoalResponse(
        success=True,
        message=f"Created new goal '{details.goal_name}' for {details.deadline}",
    )


def handle_modify_goal(description: str, events: dict) -> GoalResponse:
    """Process a goal modification request"""
    logger.info("Processing goal modification request")


    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %d %B, %Y')}."

    # Get modification details
    completion = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"Extract details for modifying an existing goal entry. {date_context}",
            },
            {"role": "user", "content": description},
        ],
        format=ModifyGoalDetails.model_json_schema(),
    )
    details = ModifyGoalDetails.model_validate_json(completion.message.content)

    logger.info(f"Modified goal: {details.model_dump_json(indent=2)}")
    for change in details.changes:
        events.update({change.field: change.new_value})

    # Generate response
    return GoalResponse(
        success=True,
        message=f"Modified event '{details.goal_identifier}' with the requested changes",
    )

# --------------------------------------------------------------
# Step 3: Chain the functions together
# --------------------------------------------------------------

def process_goal_request(user_input: str, events: dict) -> Optional[GoalResponse]:
    """Main function implementing the routing workflow"""
    logger.info("Processing goal request")

    # Route the request
    route_result = route_goal_request(user_input)

    # Check confidence threshold
    if route_result.confidence_score < 0.7:
        logger.warning(f"Low confidence score: {route_result.confidence_score}")
        return None

    # Route to appropriate handler
    if route_result.request_type == "new_goal":
        return handle_new_goal(route_result.goal_details, events)
    elif route_result.request_type == "modify_goal":
        return handle_modify_goal(route_result.goal_details, events)
    else:
        logger.warning("Request type not supported")
        return None


# --------------------------------------------------------------
# Step 3: Test with new event
# --------------------------------------------------------------

events = {}

new_goal_input = "I want to write an essay on the care of dogs by next week Wednesday"
result = process_goal_request(new_goal_input, events)
if result:
    print(f"Response: {result.message}")
    print(events)

# --------------------------------------------------------------
# Step 4: Test with modify event
# --------------------------------------------------------------

modify_goal_input = "Can you move the dog care essay deadline to the following Friday instead?"

result = process_goal_request(modify_goal_input, events)
if result:
    print(f"Response: {result.message}")
    print(events)

# --------------------------------------------------------------
# Step 5: Test with invalid request
# --------------------------------------------------------------

invalid_input = "What's the weather like today?"
result = process_goal_request(invalid_input, events)
if not result:
    print("Request not recognized as a goal operation")
print(events)
