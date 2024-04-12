from enum import Enum


class LLMClientType(Enum):
    CHAT_GPT = 1
    CLAUDE = 2


class LLMClient:
    def __init__(
        self,
        client_type,
        prompt_dir,
        reward_signature,
        code_output_tip,
        task_obs_code_string,
        task_description
    ):
        self.responses = []
        self.client_type = client_type
        self.initial_user = self.file_to_string(f'{prompt_dir}/gpt_prompts/initial_user.txt')
        self.initial_system = self.file_to_string(f'{prompt_dir}/gpt_prompts/initial_system.txt')
        self.initial_system = self.initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
        self.initial_user = self.initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
        self.messages = []

    @staticmethod
    def file_to_string(filename):
        with open(filename, 'r') as file:
            return file.read()

    def add_assistant_prompt(self, assistant_prompt, user_prompt):
        if len(self.messages) <= 2:
            self.messages += [{"role": "assistant", "content": assistant_prompt}]
            self.messages += [{"role": "user", "content": user_prompt}]
        else:
            self.messages[-2] = {"role": "assistant", "content": assistant_prompt}
            self.messages[-1] = {"role": "user", "content": user_prompt}
