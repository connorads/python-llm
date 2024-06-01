import json

import boto3
from structlog import getLogger

logger = getLogger()


def main():
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime", region_name="us-east-1"
    )

    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    system_prompt = "Please respond only with emoji."
    max_tokens = 1000
    user_message = {"role": "user", "content": "Hello World"}
    messages = [user_message]

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
    )
    response = bedrock_runtime.invoke_model(modelId=model_id, body=body)
    response_body = json.loads(response.get("body").read())

    logger.info("LLM Made prediction", content=response_body["content"][0]["text"])


if __name__ == "__main__":
    main()
