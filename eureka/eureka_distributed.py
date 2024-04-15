import hydra
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import shutil
from tqdm import tqdm

from utils.llm_clients.chatgpt import ChatGPTClient
from utils.llm_clients.claude import ClaudeClient
from utils.misc import *
from utils.file_utils import load_tensorboard_logs, create_env_file
from utils.create_task import create_task
from utils.extract_task_code import *
from utils.sqs import set_message_to_training_queue

EUREKA_ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    env_name = cfg.env.env_name.lower()
    env_parent = 'isaac' if f'{env_name}.py' in os.listdir(f'{EUREKA_ROOT_DIR}/envs/isaac') else 'dexterity'
    task_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}.py'
    task_obs_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py'
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_code_string = file_to_string(task_file)
    task_obs_code_string = file_to_string(task_obs_file)

    # Loading all text prompts
    prompt_dir = f'{EUREKA_ROOT_DIR}/utils/prompts'

    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

    task_code_string = task_code_string.replace(task, task + suffix)
    # Create Task YAML files
    create_task(ISAAC_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None

    if "gpt" in model:
        client = ChatGPTClient(
            model=model,
            prompt_dir=prompt_dir,
            reward_signature=reward_signature,
            code_output_tip=code_output_tip,
            task_obs_code_string=task_obs_code_string,
            task_description=task_description
        )
    else:
        client = ClaudeClient(
            model=model,
            prompt_dir=prompt_dir,
            reward_signature=reward_signature,
            code_output_tip=code_output_tip,
            task_obs_code_string=task_obs_code_string,
            task_description=task_description
        )

    # Eureka generation loop
    for iter in range(cfg.iteration):
        # Get Eureka response
        client.generate_response(iter, cfg.sample, cfg.temperature)
        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + client.responses[0] + "\n")

        code_runs = []
        rl_runs = []
        for response_id in range(cfg.sample):
            response_cur = client.responses[response_id]
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]

            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            # Remove unnecessary imports
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])

            # Add the Eureka Reward Signature to the environment code
            try:
                gpt_reward_signature, input_lst = get_function_signature(code_string)
            except Exception as e:
                logging.info(f"Iteration {iter}: Code Run {response_id} cannot parse function signature!")
                continue

            code_runs.append(code_string)
            reward_signature = [
                f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
                f"self.extras['gpt_reward'] = self.rew_buf.mean()",
                f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
            ]
            indent = " " * 8
            reward_signature = "\n".join([indent + line for line in reward_signature])
            if "def compute_reward(self)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature)
            elif "def compute_reward(self, actions)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self, actions):",
                                                                 "def compute_reward(self, actions):\n" + reward_signature)
            else:
                raise NotImplementedError

            # Save the new environment code when the output contains valid code string!
            cur_env_name = f"{env_name}{suffix.lower()}_iter{iter}_{response_id}"
            output_file = f"{ISAAC_ROOT_DIR}/tasks/{cur_env_name}.py"
            with open(output_file, 'w') as file:
                file.writelines(task_code_string_iter + '\n')
                file.writelines("from typing import Tuple, Dict" + '\n')
                file.writelines("import math" + '\n')
                file.writelines("import torch" + '\n')
                file.writelines("from torch import Tensor" + '\n')
                if "@torch.jit.script" not in code_string:
                    code_string = "@torch.jit.script\n" + code_string
                file.writelines(code_string + '\n')

            with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                file.writelines(code_string + '\n')

            # Copy the generated environment code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")

            # Execute the python file with flags
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            with open(rl_filepath, 'w') as f:
                input_file_path = f"{EUREKA_ROOT_DIR}/env_files/{cur_env_name}.env"
                command = ['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',
                           'hydra/output=subprocess',
                           f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                           f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                           f'headless=True',
                           f'capture_video=False',
                           'force_render=False',
                           f'max_iterations={cfg.max_iterations}']
                command = " ".join(command)
                create_env_file(input_file_path, {
                    "EUREKA_WORKING_DIR": workspace_dir,
                    "EUREKA_TASK": f"{task}{suffix}",
                    "EUREKA_TASK_CODE_MODULE": cur_env_name,
                    "EUREKA_LOG_PATH": os.path.join(workspace_dir, rl_filepath),
                    "EUREKA_CMD": f'"{command}"',
                    "EUREKA_CLEANUP": f'"rm -rf {EUREKA_ROOT_DIR}/isaacgymenvs/isaacgymenvs/tasks/{cur_env_name}.py"'
                })
                logging.info(f"Training input has been written to file {input_file_path}")
                set_message_to_training_queue(input_file_path)
            rl_runs.append((response_id, rl_filepath))

        # Gather RL training results and construct reward reflection
        code_feedbacks = []
        contents = []
        successes = []
        reward_correlations = []
        code_paths = []

        exec_success = False
        done = []

        error_info = [""] * 4

        logging.info(f"Checking progress for Iteration {iter} for {cfg.max_iterations} steps")
        sys.stdout.flush()
        sys.stderr.flush()

        progressbars = [
            tqdm(total=100,
                 file=sys.stdout,
                 position=i + 1,
                 desc=f"Progress of {i}th training sample") for i in range(cfg.sample)]
        sys.stdout.flush()
        while len(done) < cfg.sample:
            for (response_id, rl_filepath) in rl_runs:
                if response_id in done:
                    continue
                rl_log = file_to_string(rl_filepath)
                cur_step = get_current_status(rl_log)
                if cur_step == -1:
                    tqdm.write(f"Training of {response_id}th sample has failed")
                    error_info[response_id] = rl_log
                    done.append(response_id)
                elif cur_step == 100.0:
                    tqdm.write(f"Training of {response_id}th sample succeeded")
                    done.append(response_id)
                elif cur_step > 0:
                    progressbars[response_id].update(cur_step)
            time.sleep(30)

        sys.stdout.flush()
        sys.stderr.flush()

        for i, error in enumerate(error_info):
            if len(error) > 0:
                logging.error(f"Error log for {i}th sample:")
                logging.error(extract_stacktrace(error))

        for (response_id, rl_filepath) in rl_runs:
            code_paths.append(f"env_iter{iter}_response{response_id}.py")

            try:
                with open(rl_filepath, 'r') as f:
                    stdout_str = f.read()
            except:
                content = execution_error_feedback.format(
                    traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                content += code_output_tip
                contents.append(content)
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                response_id += 1
                continue

            content = ''
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == '':
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                lines = stdout_str.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Tensorboard Directory:'):
                        break

                tensorboard_logdir = line.split(':')[-1].strip()
                tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
                epoch_freq = max(int(max_iterations // 10), 1)

                content += policy_feedback.format(epoch_freq=epoch_freq)

                # Compute Correlation between Human-Engineered and GPT Rewards
                if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                    gt_reward = np.array(tensorboard_logs["gt_reward"])
                    gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                    reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                    reward_correlations.append(reward_correlation)

                # Add reward components log to the feedback
                for metric in tensorboard_logs:
                    if "/" not in metric:
                        metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                        metric_cur_max = max(tensorboard_logs[metric])
                        metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                        if "consecutive_successes" == metric:
                            successes.append(metric_cur_max)
                        metric_cur_min = min(tensorboard_logs[metric])
                        if metric != "gt_reward" and metric != "gpt_reward":
                            if metric != "consecutive_successes":
                                metric_name = metric
                            else:
                                metric_name = "task_score"
                            content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                        else:
                            # Provide ground-truth score when success rate not applicable
                            if "consecutive_successes" not in tensorboard_logs:
                                content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                code_feedbacks.append(code_feedback)
                content += code_feedback
            else:
                # Otherwise, provide execution traceback error feedback
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content)

        # Repeat the iteration if all code generation failed
        if not exec_success and cfg.sample != 1:
            execute_rates.append(0.)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
            continue

        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(np.array(successes))
        best_content = contents[best_sample_idx]

        max_success = successes[best_sample_idx]
        max_success_reward_correlation = reward_correlations[best_sample_idx]
        execute_rate = np.sum(np.array(successes) >= 0.) / cfg.sample

        # Update the best Eureka Output
        if max_success > max_success_overall:
            max_success_overall = max_success
            max_success_reward_correlation_overall = max_success_reward_correlation
            max_reward_code_path = code_paths[best_sample_idx]

        execute_rates.append(execute_rate)
        max_successes.append(max_success)
        max_successes_reward_correlation.append(max_success_reward_correlation)
        best_code_paths.append(code_paths[best_sample_idx])

        logging.info(f"Iteration {iter}: Best reward index: {best_sample_idx} with reward code path {max_reward_code_path}")
        logging.info(
            f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {max_success_reward_correlation}")
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(f"Iteration {iter}: GPT Output Content:\n" + client.responses[best_sample_idx] + "\n")
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")

        # Plot the success rate
        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f'{cfg.env.task}')

        x_axis = np.arange(len(max_successes))

        axs[0].plot(x_axis, np.array(max_successes))
        axs[0].set_title("Max Success")
        axs[0].set_xlabel("Iteration")

        axs[1].plot(x_axis, np.array(execute_rates))
        axs[1].set_title("Execute Rate")
        axs[1].set_xlabel("Iteration")

        fig.tight_layout(pad=3.0)
        plt.savefig('summary.png')
        np.savez('summary.npz', max_successes=max_successes, execute_rates=execute_rates, best_code_paths=best_code_paths,
                 max_successes_reward_correlation=max_successes_reward_correlation)

        client.add_assistant_prompt(client.responses[best_sample_idx], best_content)

        # Save dictionary as JSON file
        with open('messages.json', 'w') as file:
            json.dump(client.get_messages(), file, indent=4)

    # Evaluate the best reward code many times
    if max_reward_code_path is None:
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()
    logging.info(
        f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}")
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")
    output_best_reward_file_path = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"
    shutil.copy(max_reward_code_path, output_best_reward_file_path)

    rl_files = []
    for i in range(cfg.num_eval):
        # Execute the python file with flags
        rl_filepath = f"reward_code_eval{i}.txt"

        with open(rl_filepath, 'w') as f:
            input_file_path = f"{EUREKA_ROOT_DIR}/env_files/reward_code_eval{i}.env"
            command = ['python3', '-u', f'{ISAAC_ROOT_DIR}/train.py',
                       'hydra/output=subprocess',
                       f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                       f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                       f'headless=False', f'capture_video=False', 'force_render=True', f'seed={i}']
            command = " ".join(command)
            create_env_file(input_file_path, {
                "EUREKA_WORKING_DIR": workspace_dir,
                "EUREKA_TASK": f"{task}{suffix}",
                "EUREKA_LOG_PATH": os.path.join(workspace_dir, rl_filepath),
                "EUREKA_TASK_CODE_MODULE": f"{env_name}{suffix.lower()}",
                "EUREKA_CMD": f'"{command}"',
                "EUREKA_CLEANUP": ""
            })
            set_message_to_training_queue(input_file_path)
            logging.info(command)
            rl_files.append(rl_filepath)

    reward_code_final_successes = []
    reward_code_correlations_final = []
    done = []
    logging.info(f"Checking progress for Final Training")
    sys.stdout.flush()
    sys.stderr.flush()

    progressbars = [
        tqdm(total=100,
             file=sys.stdout,
             position=i + 1,
             desc=f"Progress of {i}th final training sample") for i in range(len(rl_files))]
    sys.stdout.flush()

    while len(done) < len(rl_files):
        for i, rl_filepath in enumerate(rl_files):
            if i in done:
                continue
            rl_log = file_to_string(rl_filepath)
            cur_step = get_current_status(rl_log)
            if cur_step == -1.0:
                tqdm.write(f"Training of {i}th sample failed")
                tqdm.write(rl_log)
                done.append(i)
            elif cur_step == 100.0:
                tqdm.write(f"Training of {i}th sample succeeded")
                done.append(i)
            elif cur_step > 0:
                progressbars[i].update(cur_step)
        time.sleep(30)

    sys.stdout.flush()
    sys.stderr.flush()

    for i, rl_filepath in enumerate(rl_files):
        with open(rl_filepath, 'r') as f:
            stdout_str = f.read()
        lines = stdout_str.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('Tensorboard Directory:'):
                break
        tensorboard_logdir = line.split(':')[-1].strip()
        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        max_success = max(tensorboard_logs['consecutive_successes'])
        reward_code_final_successes.append(max_success)

        if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
            gt_reward = np.array(tensorboard_logs["gt_reward"])
            gpt_reward = np.array(tensorboard_logs["gpt_reward"])
            reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
            reward_code_correlations_final.append(reward_correlation)

    logging.info(f"Final Success Mean: {np.mean(reward_code_final_successes)}, Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}")
    logging.info(
        f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, Std: {np.std(reward_code_correlations_final)}, Raw: {reward_code_correlations_final}")
    np.savez('final_eval.npz', reward_code_final_successes=reward_code_final_successes, reward_code_correlations_final=reward_code_correlations_final)


if __name__ == "__main__":
    main()
