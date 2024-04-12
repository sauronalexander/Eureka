import subprocess
import os
import json
import logging
import sys
import time
import progressbar

from utils.extract_task_code import file_to_string

def set_freest_gpu():
    freest_gpu = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)

def get_freest_gpu():
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    # Find GPU with most free memory
    freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])

    return freest_gpu['index']

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found


def get_current_status(rl_log):
    if "Traceback" in rl_log:
        return -1
    if "MAX EPOCHS NUM!" in rl_log:
        return 100.0
    if len(rl_log) > 2:
        last_line = rl_log.split('\n')[-2]
        if "fps step" in last_line:
            status = last_line[last_line.find('epoch'):last_line.find('frames')].split(' ')[1].split('/')
            return float(status[0]) / float(status[1]) * 100.0
    return 0


def extract_stacktrace(rl_log):
    start_idx = rl_log.find("RuntimeError:")
    end_idx = rl_log.rfind("Traceback")
    if start_idx > end_idx:
        return rl_log[start_idx:]
    else:
        return rl_log[start_idx:end_idx]


if __name__ == "__main__":
    print(get_freest_gpu())