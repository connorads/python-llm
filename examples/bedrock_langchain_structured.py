import boto3
from langchain_aws import ChatBedrock
from langchain_core.pydantic_v1 import BaseModel, Field
from structlog import getLogger

logger = getLogger()


class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")


model = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    client=boto3.client("bedrock-runtime", region_name="us-east-1"),
)

try:
    structured_llm = model.with_structured_output(Joke)
except NotImplementedError as e:
    logger.error("with_structured_output not supported https://bit.ly/3X05wqB")
    raise SystemExit(0) from e

prediction = model.invoke("Tell me a fintech joke")

logger.info("LLM Made prediciton", content=prediction.content)
