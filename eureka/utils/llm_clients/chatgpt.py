import os
import logging
import time

from openai import OpenAI
from .llm_client import LLMClient, LLMClientType


class ChatGPTClient(LLMClient):
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
            LLMClientType.CHAT_GPT,
            prompt_dir,
            reward_signature,
            code_output_tip,
            task_obs_code_string,
            task_description
        )
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.system_prompt = self.initial_system
        self.messages = [{"role": "system", "content": self.initial_system}, {"role": "user", "content": self.initial_user}]

    def generate_response(self, iter, sample_count, temperature):
        response_cur = None
        total_samples = 0
        total_token = 0
        prompt_tokens = 0
        total_completion_token = 0
        chunk_size = sample_count if "gpt-3.5" in self.model else 4
        while total_samples < sample_count:
            for attempt in range(1000):
                try:
                    response_cur = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        temperature=temperature,
                        n=chunk_size
                    )
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt + 1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                exit()
            response_cur = response_cur.model_dump()
            self.responses += [response_cur["choices"][i]["message"]["content"] for i in range(chunk_size)]
            prompt_tokens += response_cur["usage"]["prompt_tokens"]
            total_completion_token += response_cur["usage"]["completion_tokens"]
            total_token += response_cur["usage"]["total_tokens"]

        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

    def add_assistant_prompt(self, assistant_prompt, user_prompt):
        if len(self.messages) == 2:
            self.messages += [{"role": "assistant", "content": assistant_prompt}]
            self.messages += [{"role": "user", "content": user_prompt}]
        else:
            assert len(self.messages) == 4
            self.messages[-2] = {"role": "assistant", "content": assistant_prompt}
            self.messages[-1] = {"role": "user", "content": user_prompt}

    def get_messages(self):
        return self.messages
