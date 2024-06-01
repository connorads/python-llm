import instructor
from anthropic import AnthropicBedrock
from pydantic import BaseModel
from structlog import get_logger

logger = get_logger()


class User(BaseModel):
    name: str
    age: int


client = instructor.from_anthropic(AnthropicBedrock(aws_region="us-east-1"))

# note that client.chat.completions.create will also work
resp = client.messages.create(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)

assert isinstance(resp, User)
assert resp.name == "Jason"
assert resp.age == 25

logger.info("LLM Made prediction", content=resp.model_dump_json())