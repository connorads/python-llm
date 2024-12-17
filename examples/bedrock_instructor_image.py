import base64
import instructor
from anthropic import AnthropicBedrock
from pydantic import BaseModel, Field
import requests
from structlog import get_logger

logger = get_logger()


class ImageDescription(BaseModel):
    description: str = Field(description="A description of what the image shows")


client = instructor.from_anthropic(AnthropicBedrock(aws_region="us-east-1"))

IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/London_Skyline_%28125508655%29.jpeg/640px-London_Skyline_%28125508655%29.jpeg"
image_base64 = base64.b64encode(requests.get(IMAGE_URL, timeout=10).content).decode("utf-8")

resp = client.messages.create(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                # Type error: instructor doesn't support anthropic "image" type
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_base64,
                    },
                },
                {"type": "text", "text": "What does this image show?"},
            ],
        }
    ],
    response_model=ImageDescription,
)

assert isinstance(resp, ImageDescription)

logger.info("LLM Made prediction", content=resp.model_dump_json())
