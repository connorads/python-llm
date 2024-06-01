import instructor
from anthropic import AnthropicBedrock
from pydantic import BaseModel, Field
from structlog import get_logger

logger = get_logger()


class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")

client = instructor.from_anthropic(AnthropicBedrock(aws_region="us-east-1"))

# note that client.chat.completions.create will also work
resp = client.messages.create(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Tell me a joke about fintech lending to UK SMEs",
        }
    ],
    response_model=Joke,
)

assert isinstance(resp, Joke)

logger.info("LLM Made prediction", content=resp.model_dump_json())