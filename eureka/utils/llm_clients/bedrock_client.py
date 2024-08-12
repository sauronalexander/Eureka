import time

import logging
import json

import boto3
from botocore.config import Config
from .llm_client import LLMClient

from botocore.exceptions import ClientError

CONTENT_TYPE = ACCEPT = 'application/json'
BEDROCK_DEFAULT_REGION = 'us-east-1'


class BedrockClient(LLMClient):
    def __init__(
        self,
        llm_client_type,
        prompt_dir,
        reward_signature,
        code_output_tip,
        task_obs_code_string,
        task_description,
        provider,
        region,
        delay
    ):
        super().__init__(
            llm_client_type,
            prompt_dir,
            reward_signature,
            code_output_tip,
            task_obs_code_string,
            task_description
        )
        self.region = region
        self.provider = provider
        config = Config(
            retries={
                'max_attempts': 0
            }
        )
        self.bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region,
            config=config
        )
        self.system_prompt = self.initial_system
        self.messages = [
            {"role": "user", "content": self.initial_user}
        ]
        self.model_name = ""
        self.model_id = ""
        self.delay = delay

    def get_model_id(self, model_name):
        self.model_name = model_name
        logging.info(f"Finding {model_name} in Bedrock Foundation Models")
        bedrock = boto3.client(service_name="bedrock", region_name=self.region)
        response = bedrock.list_foundation_models(byProvider=self.provider)
        for model in response['modelSummaries']:
            if model["modelName"] == model_name and model['modelId'][-5:] == "-v1:0":
                logging.info(f"Found model {model_name} with id " + model["modelId"])
                self.model_id = model["modelId"]
                return
        logging.fatal(f"{model_name} does not exist")
        assert False

    def _generate_response(self, body, sample_count):
        total_samples = 0
        responses = []
        while total_samples < sample_count:
            try:
                response = self.bedrock.invoke_model(
                    body=json.dumps(body),
                    modelId=self.model_id,
                    accept=ACCEPT,
                    contentType=CONTENT_TYPE
                )
                response_body = json.loads(response.get('body').read())
                responses.append(response_body)
                total_samples += 1
                logging.info(f"Prompt {total_samples - 1} completed from {self.model_name}")
            except ClientError as e:
                if e.response['Error']['Code'] != "ThrottlingException":
                    logging.warning(f"Code generation failed due to {e}")
            except Exception as e:
                logging.warning(f"Code generation failed due to {e}")
            time.sleep(self.delay)

        return responses

    def get_messages(self):
        return [{"role": "system", "content": self.system_prompt}] + self.messages
