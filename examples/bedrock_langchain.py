import boto3
from langchain_aws import ChatBedrock
from structlog import getLogger

logger = getLogger()


model = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    client=boto3.client("bedrock-runtime", region_name="us-east-1"),
)

prediction = model.invoke("I am a happy person")

logger.info("LLM Made prediciton", content=prediction.content)
