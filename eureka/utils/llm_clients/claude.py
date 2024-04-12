import os
import logging
import time
import json

import copy

import boto3
from .llm_client import LLMClient, LLMClientType

BEDROCK_REGION = 'us-east-1'
CONTENT_TYPE = ACCEPT = 'application/json'
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


class ClaudeClient(LLMClient):
    def __init__(
        self,
        model,
        prompt_dir,
        reward_signature,
        code_output_tip,
        task_obs_code_string,
        task_description
    ):
        super().__init__(
            LLMClientType.CLAUDE,
            prompt_dir,
            reward_signature,
            code_output_tip,
            task_obs_code_string,
            task_description
        )
        if model == "claude-3-sonnet":
            self.model_name = "Claude 3 Sonnet"
        elif model == "claude-3-haiku":
            self.model_name = "Claude 3 Haiku"
        else:
            logging.fatal(f"{model} does not exist")
            assert False
        self.model_id = self.get_model_id(self.model_name)
        self.bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=BEDROCK_REGION
        )
        self.system_prompt = self.initial_system
        self.messages = [
            {"role": "user", "content": self.initial_user}
        ]

    @staticmethod
    def get_model_id(model_name):
        logging.info(f"Finding {model_name} in Bedrock Foundation Models")
        bedrock = boto3.client(service_name="bedrock", region_name=BEDROCK_REGION)
        response = bedrock.list_foundation_models(byProvider='Anthropic')
        for model in response['modelSummaries']:
            if model["modelName"] == model_name:
                logging.info(f"Found model {model_name} with id " + model["modelId"])
                return model["modelId"]
        logging.fatal(f"{model_name} does not exist")
        assert False

    def generate_response(self, iter, sample_count, temperature):
        body = copy.deepcopy(BODY_TEMPLATE)
        body["system"] = self.system_prompt
        body["temperature"] = temperature
        body["messages"] = self.messages
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        prompt_tokens = 0
        while total_samples < sample_count:
            try:
                response = self.bedrock.invoke_model(
                    body=json.dumps(body),
                    modelId=self.model_id,
                    accept=ACCEPT,
                    contentType=CONTENT_TYPE
                )
                response_body = json.loads(response.get('body').read())
                self.responses += [response_body['content'][0]['text']]
                total_samples += 1
                total_token += response_body["usage"]["input_tokens"] + response_body["usage"]["output_tokens"]
                total_completion_token += response_body["usage"]["output_tokens"]
                prompt_tokens += response_body["usage"]["input_tokens"]
                print(f"Prompt {total_samples - 1} completed from {self.model_name}")
            except Exception as e:
                logging.warning("Code generation failed due to ", e)

        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

    def get_messages(self):
        return [{"role": "system", "content": self.system_prompt}] + self.messages