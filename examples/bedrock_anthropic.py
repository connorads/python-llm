from anthropic import AnthropicBedrock

from structlog import get_logger

logger = get_logger()

client = AnthropicBedrock(aws_region="us-east-1")

message = client.messages.create(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Hello!",
        }
    ],
    model="anthropic.claude-3-sonnet-20240229-v1:0",
)

logger.info("LLM Made prediction", content=message.content[0].text)