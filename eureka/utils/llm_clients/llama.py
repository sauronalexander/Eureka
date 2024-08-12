import logging
import sys
from collections import defaultdict

from .llm_client import LLMClientType
from .bedrock_client import BedrockClient

LLAMA_DEFAULT_REGION = 'us-west-2'
PROVIDER = 'Meta'
PROMPT_FORMAT = """
<|begin_of_text|>
<|start_header_id|>{}<|end_header_id|>
{}
<|eot_id|>"""
BODY_TEMPLATE = {
    "prompt": "",
    "max_gen_len": 0,
    "temperature": 0.0
}
DELAY = 1


class LlamaClient(BedrockClient):
    def __init__(
        self,
        model,
        prompt_dir,
        reward_signature,
        code_output_tip,
        task_obs_code_string,
        task_description,
    ):
        super().__init__(
            LLMClientType.LLAMA,
            prompt_dir,
            reward_signature,
            code_output_tip,
            task_obs_code_string,
            task_description,
            PROVIDER,
            LLAMA_DEFAULT_REGION,
            DELAY
        )
        if model == "llama-3-1-70b":
            model_name = "Llama 3.1 70B Instruct"
            self.max_token = 2048
        elif model == "llama-3-1-405b":
            model_name = "Llama 3.1 405B Instruct"
            self.max_token = 4096
        else:
            logging.fatal(f"{model} does not exist")
            sys.exit(1)
        self.get_model_id(model_name)

    def generate_response(self, iteration, sample_count, temperature):
        body = defaultdict()
        body["temperature"] = temperature
        body["prompt"] = self.translate_to_llama_prompt()
        body["max_gen_len"] = self.max_token
        responses = self._generate_response(body, sample_count)
        self.responses = [x["generation"] for x in responses]
        total_token = 0
        total_completion_token = 0
        prompt_tokens = 0
        for response in responses:
            total_token += response["prompt_token_count"] + response["generation_token_count"]
            total_completion_token += response["generation_token_count"]
            prompt_tokens += response["prompt_token_count"]
        logging.info(f"Iteration {iteration}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

    def translate_to_llama_prompt(self):
        prompts = [PROMPT_FORMAT.format("system", self.system_prompt)]

        for message in self.messages:
            prompts.append(PROMPT_FORMAT.format(message["role"], message["content"]))

        return "\n".join(prompts)
