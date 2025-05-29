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


class CalendarRequestType(BaseModel):
    """Router LLM call: Determine the type of calendar request"""

    request_type: Literal["new_event", "modify_event", "other"] = Field(
        description="Type of calendar request being made"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")
    details: str = Field(description="Cleaned details of the request")


class NewEventDetails(BaseModel):
    """Details for creating a new event"""

    name: str = Field(description="Name of the event")
    date: datetime = Field(description="Date and time of the event (ISO 8601)")
    duration_minutes: int = Field(description="Duration in minutes")
    participants: list[str] = Field(description="List of participants")


class Change(BaseModel):
    """Details for changing an existing event"""

    field: str = Field(description="Field to change")
    new_value: str = Field(description="New value for the field")


class ModifyEventDetails(BaseModel):
    """Details for modifying an existing event"""

    event_identifier: str = Field(
        description="Description to identify the existing event"
    )
    changes: list[Change] = Field(description="List of changes to make")
    participants_to_add: Optional[list[str]] = Field(description="New participants to add")
    participants_to_remove: Optional[list[str]] = Field(description="Participants to remove")


class CalendarResponse(BaseModel):
    """Final response format"""

    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="User-friendly response message")
    calendar_link: Optional[str] = Field(description="Calendar link if applicable")


# --------------------------------------------------------------
# Step 2: Define the routing and processing functions
# --------------------------------------------------------------

def route_calendar_request(user_input: str) -> CalendarRequestType:
    """Router LLM call to determine the type of calendar request"""
    logger.info("Routing calendar request")
    logger.info(f"Input text: {user_input}")

    completion = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Determine if this is a request to create a new calendar event, modify an existing one, or if it's an irrelevant input",
            },
            {"role": "user", "content": user_input},
        ],
        format=CalendarRequestType.model_json_schema(),
    )
    result = CalendarRequestType.model_validate_json(completion.message.content)
    result.details = user_input
    logger.info(
        f"Request routed as: {result.request_type} with confidence: {result.confidence_score}. Details passed through: {result.details}"
    )
    return result


# Parse details
# f"{date_context} Extract detailed event information. When dates reference 'next Tuesday' or similar relative dates, use this current date as a reference. Assume all events to be in the future, and to start at mentioned times unless otherwise specified. Format dates and times using ISO8601: {ISO8601_info} Ensure you extract the correct date.",

def handle_new_event(description: str, events: dict) -> CalendarResponse:
    """Process a new event request"""
    logger.info("Processing new event request")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %d %B, %Y')}."

    # Get event details
    completion = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"Extract details for creating a new calendar event. {date_context}",
            },
            {"role": "user", "content": description},
        ],
        format=NewEventDetails.model_json_schema(),
    )
    details = NewEventDetails.model_validate_json(completion.message.content)

    logger.info(f"New event: {details.model_dump_json(indent=2)}")
    events.update({"name": details.name, "date": details.date, "duration_minutes": details.duration_minutes, "participants": details.participants})

    # Generate response
    return CalendarResponse(
        success=True,
        message=f"Created new event '{details.name}' for {details.date} with {', '.join(details.participants)}",
        calendar_link=f"calendar://new?event={details.name}",
    )


def handle_modify_event(description: str, events: dict) -> CalendarResponse:
    """Process an event modification request"""
    logger.info("Processing event modification request")


    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %d %B, %Y')}."

    # Get modification details
    completion = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"Extract details for modifying an existing calendar event. {date_context}",
            },
            {"role": "user", "content": description},
        ],
        format=ModifyEventDetails.model_json_schema(),
    )
    details = ModifyEventDetails.model_validate_json(completion.message.content)

    logger.info(f"Modified event: {details.model_dump_json(indent=2)}")
    events.update({details.changes[0].field: details.changes[0].new_value})
    # Generate response
    return CalendarResponse(
        success=True,
        message=f"Modified event '{details.event_identifier}' with the requested changes",
        calendar_link=f"calendar://modify?event={details.event_identifier}",
    )

# --------------------------------------------------------------
# Step 3: Chain the functions together
# --------------------------------------------------------------

def process_calendar_request(user_input: str, events: dict) -> Optional[CalendarResponse]:
    """Main function implementing the routing workflow"""
    logger.info("Processing calendar request")

    # Route the request
    route_result = route_calendar_request(user_input)

    # Check confidence threshold
    if route_result.confidence_score < 0.7:
        logger.warning(f"Low confidence score: {route_result.confidence_score}")
        return None

    # Route to appropriate handler
    if route_result.request_type == "new_event":
        return handle_new_event(route_result.details, events)
    elif route_result.request_type == "modify_event":
        return handle_modify_event(route_result.details, events)
    else:
        logger.warning("Request type not supported")
        return None


# --------------------------------------------------------------
# Step 3: Test with new event
# --------------------------------------------------------------

events = {}

new_event_input = "Let's schedule a team meeting next Tuesday at 2pm with Alice and Bob"
result = process_calendar_request(new_event_input, events)
if result:
    print(f"Response: {result.message}")
    print(events)

# --------------------------------------------------------------
# Step 4: Test with modify event
# --------------------------------------------------------------

modify_event_input = (
    "Can you move the team meeting with Alice and Bob to Wednesday at 3pm instead?"
)
result = process_calendar_request(modify_event_input, events)
if result:
    print(f"Response: {result.message}")
    print(events)

# --------------------------------------------------------------
# Step 5: Test with invalid request
# --------------------------------------------------------------

invalid_input = "What's the weather like today?"
result = process_calendar_request(invalid_input, events)
if not result:
    print("Request not recognized as a calendar operation")
print(events)
