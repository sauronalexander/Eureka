import logging
import copy
import sys

from .llm_client import LLMClientType
from .bedrock_client import BedrockClient, BEDROCK_DEFAULT_REGION

PROVIDER = "Anthropic"
BEDROCK_ALT_REGION = 'us-west-2'

BODY_TEMPLATE = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 4096,
    "system": "",
    "messages": [
        {
            "role": "",
            "content": [
                {
                    "type": "text",
                    "text": ""
                }
            ]
        }
    ],
    "temperature": 1,
    # "top_p": 0.999,
    # "top_k": int,
    # "stop_sequences": [string]
}
DELAY = 5


class ClaudeClient(BedrockClient):
    def __init__(
        self,
        model,
        prompt_dir,
        reward_signature,
        code_output_tip,
        task_obs_code_string,
        task_description
    ):
        region = BEDROCK_DEFAULT_REGION
        if model == "claude-3-sonnet":
            model_name = "Claude 3 Sonnet"
        elif model == "claude-3-haiku":
            model_name = "Claude 3 Haiku"
        elif model == "claude-3-opus":
            model_name = "Claude 3 Opus"
            region = BEDROCK_ALT_REGION
        elif model == "claude-3-5-sonnet":
            model_name = "Claude 3.5 Sonnet"
        else:
            logging.fatal(f"{model} does not exist")
            sys.exit(1)
        super().__init__(
            LLMClientType.CLAUDE,
            prompt_dir,
            reward_signature,
            code_output_tip,
            task_obs_code_string,
            task_description,
            PROVIDER,
            region,
            DELAY
        )
        self.get_model_id(model_name)

    def generate_response(self, iteration, sample_count, temperature):
        body = copy.deepcopy(BODY_TEMPLATE)
        body["system"] = self.system_prompt
        body["temperature"] = temperature
        body["messages"] = self.messages
        responses = self._generate_response(body, sample_count)
        self.responses = [x['content'][0]['text'] for x in responses]
        total_token = 0
        total_completion_token = 0
        prompt_tokens = 0
        for response in responses:
            total_token += response["usage"]["input_tokens"] + response["usage"]["output_tokens"]
            total_completion_token += response["usage"]["output_tokens"]
            prompt_tokens += response["usage"]["input_tokens"]
        logging.info(f"Iteration {iteration}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
