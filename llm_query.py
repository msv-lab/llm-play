# llm-query is a utility for small-scale experimentation with LLMs in UNIX environment.
# Copyright (C) 2025 Sergey Mechtaev
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from dataclasses import dataclass
from typing import Dict, Tuple

import shlex
import os
import sys
import shutil
import configparser
import argparse
import tempfile
import subprocess
import re
from collections import defaultdict

import yaml
from tqdm import tqdm
import InquirerPy
from openai import OpenAI


ANSWER_FORMAT_DIRECTIVE = "Wrap the final answer with <answer> </answer>."
ANSWER_EXTRACTOR = r"sed -n '0,/<\/answer>/s/.*<answer>\(.*\)<\/answer>.*/\1/p' %%ESCAPED_OUTPUT_FILE%%"
CODE_EXTRACTOR = r"awk '/^```/{if (!found++) { while(getline && $0 !~ /^```/) print; exit}}' %%ESCAPED_OUTPUT_FILE%%"


DEFAULT_CONFIG = fr"""
default:
  model: qwen2.5-72b-instruct
  temperature: 1.0
  extractor: __ID__
  equivalence: __ID__
providers:
  Aliyun:
    API: OpenAI
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    key_env_variable: DASHSCOPE_API_KEY
    support_multiple_responses: False
  DeepSeek:
    API: OpenAI
    base_url: https://api.deepseek.com
    key_env_variable: DEEPSEEK_API_KEY
    support_multiple_responses: False
  CloseAI_OpenAI:
    API: OpenAI
    base_url: https://api.openai-proxy.org/v1
    key_env_variable: CLOSEAI_API_KEY
    support_multiple_responses: True
  CloseAI_Anthropic:
    API: Anthropic
    base_url: https://api.openai-proxy.org/anthropic
    key_env_variable: CLOSEAI_API_KEY
    support_multiple_responses: True
models:
  -
    name: qwen-max-2024-09-19
    provider: Aliyun
  -
    name: qwq-32b-preview
    provider: Aliyun
  -
    name: qwen2.5-72b-instruct
    provider: Aliyun
  -
    name: qwen2.5-7b-instruct
    provider: Aliyun
  -
    name: qwen2.5-coder-32b-instruct
    provider: Aliyun
  -
    name: qwen2.5-coder-7b-instruct
    provider: Aliyun
  -
    name: deepseek-chat
    provider: DeepSeek
  -
    name: o1-2024-12-17
    provider: CloseAI_OpenAI
  -
    name: o1-mini-2024-09-12
    provider: CloseAI_OpenAI
  -
    name: gpt-4o-2024-11-20
    provider: CloseAI_OpenAI
  -
    name: gpt-4o-mini-2024-07-18
    provider: CloseAI_OpenAI
  -
    name: claude-3-5-sonnet-20241022
    provider: CloseAI_Anthropic
extractors:
  - __ID__
  - |-
    {ANSWER_EXTRACTOR}
  - |-
    {CODE_EXTRACTOR}
equivalences:
  - __ID__
  - __TRIMMED_CASE_INSENSITIVE__
  - |-
    llm-query -m qwen2.5-72b-instruct 'Are these two answers equivalent: "%%OUTPUT1%%" and "%%OUTPUT2%%"?' --predicate
"""


UNNAMED_TASK = '__unnamed__'


def get_provider_by_model(model, config):
    for m in config['models']:
        if model == m['name']:
            return m['provider']
    raise ValueError(f"no provider for model {model}")


def get_responses(prompt, model, temperature, n, config):
    provider = get_provider_by_model(model, config)
    client = OpenAI(
        api_key=os.getenv(config['providers'][provider]['key_env_variable']),
        base_url=config['providers'][provider]['base_url'],
    )
    if config['providers'][provider]['support_multiple_responses']:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'user', 'content': prompt}],
            n=n
        )
        return [c.message.content for c in completion.choices]
    else:
        responses = []
        for i in range(n):
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'user', 'content': prompt}],
                n=1
            )
            responses.append(completion.choices[0].message.content)
        return responses


def stream_response(prompt, model, temperature, config):
    provider = get_provider_by_model(model, config)
    client = OpenAI(
        api_key=os.getenv(config['providers'][provider]['key_env_variable']),
        base_url=config['providers'][provider]['base_url'],
    )
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")

    if os.isatty(sys.stdout.fileno()):
        print()


def recreate_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def execute_batch_jobs(inputs, settings, output, config):
    recreate_directory(output)
    num_jobs = len(settings['models']) * len(inputs)
    if num_jobs > 1:
        pbar = tqdm(total=num_jobs, ascii=True)
    for im, model in enumerate(settings['models']):
        model_path = output + "/" + model + "_" + str(settings['temperature'])
        os.makedirs(model_path)
        for name, prompt in inputs:
            with open(f"{model_path}/{name}.md", "w") as file:
                file.write(prompt)
            task_path = model_path + "/" + name
            os.makedirs(task_path)
            for i, response in enumerate(get_responses(prompt, model, settings['temperature'], settings['num_responses'], config)):
                if settings['extractor'] != '__ID__':
                    result = extract(response, prompt, settings['extractor'], name)
                else:
                    result = response
                with open(f"{task_path}/{i}.md", "w") as file:
                    file.write(result)
                if num_jobs > 1:
                    pbar.update()
    if num_jobs > 1:
        pbar.close()


def instantiate_shell_template(t, prompt, prompt_file, outputs, output_files, task):
    assert(len(outputs) == len(output_files))

    def render(value, escape=False):
        return shlex.quote(value) if escape and value is not None else value

    if len(outputs) == 1:
        t = t.replace(f"%%OUTPUT%%", render(outputs[0]))
        t = t.replace(f"%%ESCAPED_OUTPUT%%", render(outputs[0], escape=True))
        t = t.replace(f"%%OUTPUT_FILE%%", render(output_files[0]))
        t = t.replace(f"%%ESCAPED_OUTPUT_FILE%%", render(output_files[0], escape=True))
    else:
        for i in range(len(outputs), 0, -1):
            t = t.replace(f"%%OUTPUT{i}%%", render(outputs[i - 1]))
            t = t.replace(f"%%ESCAPED_OUTPUT{i}%%", render(outputs[i - 1], escape=True))
            t = t.replace(f"%%OUTPUT_FILE{i}%%", render(output_files[i - 1]))
            t = t.replace(f"%%ESCAPED_OUTPUT_FILE{i}%%", render(output_files[i - 1], escape=True))
    t = t.replace(f"%%PROMPT%%", render(prompt, 0))
    t = t.replace(f"%%ESCAPED_PROMPT%%", render(prompt, escape=True))
    t = t.replace(f"%%PROMPT_FILE%%", render(prompt_file))
    t = t.replace(f"%%ESCAPED_PROMPT_FILE%%", render(prompt_file, escape=True))
    t = t.replace(f"%%TASK_ID%%", render(task))
    t = t.replace(f"%%ESCAPED_TASK_ID%%", render(task, escape=True))

    return t


def extract(response, prompt, extractor, task):
    with tempfile.NamedTemporaryFile() as prompt_file:
        prompt_file.write(prompt.encode())
        prompt_file.flush()
        with tempfile.NamedTemporaryFile() as output_file:
            output_file.write(response.encode())
            output_file.flush()
            cmd = instantiate_shell_template(extractor, prompt, prompt_file.name, outputs=[response], output_files=[output_file.name], task=task)
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            return result.stdout


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Query Interface")
    parser.add_argument("query", nargs='?', type=str, help="Query string")
    parser.add_argument("-i", "--inputs", nargs='+', type=str, help="Input files")
    parser.add_argument("-o", "--output", type=str, help="Output directory")
    parser.add_argument("-m", "--models", nargs='+', type=str, help="List of models to query")
    parser.add_argument("-t", "--temperature", type=float, help="Temperature for model generation")
    parser.add_argument("-n", "--num-responses", type=int, help="Number of responses to generate")
    parser.add_argument("-x", "--extract", type=str, help="Extract data in directory")
    parser.add_argument("--extractor", type=str, help="Answer extraction shell command")
    parser.add_argument("--extension", type=str, help="File extension for extracted data")
    parser.add_argument("-a", "--answer", action="store_true", help="Answer question")
    parser.add_argument("-c", "--code", action="store_true", help="Generate code")
    parser.add_argument("-d", "--distribution", type=str, help="Compute distribution of responses in the directory")
    parser.add_argument("--equivalence", type=str, help="Equivalence relation shell command")
    parser.add_argument("-e", "--eval", type=str, help="Evaluate responses in the directory")
    parser.add_argument("--equal", type=str, help="Check equivalence of responses to the specified value")
    parser.add_argument("--evalautor", type=str, help="Evaluator shell command")
    parser.add_argument("-p", "--predicate", action="store_true", help="Evaluate truthfulness of the predicate")
    parser.add_argument("-s", "--setup", action="store_true", help="Set default options")
    return parser.parse_args()


def validate_arguments(arguments):
    if (((arguments.num_responses and arguments.num_responses > 1) or
         (arguments.models and len(arguments.models) > 1) or
         (arguments.inputs and len(arguments.inputs) > 1)) and
        not arguments.output):
        print("for multiple responses/models/inputs, the output directory needs to be specified", file=sys.stderr)
        exit(1)

    if sum([bool(arguments.query),
            bool(arguments.inputs),
            bool(arguments.distribution),
            bool(arguments.eval)]) > 1:
        print("choose only one of (1) query string, (2) input files, (3) response distribution, (4) response evaluation", file=sys.stderr)
        exit(1)

    if ((arguments.answer or arguments.code) and arguments.extractor):
        print("the answer/code options cannot be used with a custom extractor", file=sys.stderr)
        exit(1)

    if (arguments.code and arguments.answer):
        print("the code and the answer options are mutually exclusive", file=sys.stderr)
        exit(1)


def process_input_files(file_list):
    label_content_pairs = []
    seen_labels = set()

    for file_path in file_list:
        base_name = os.path.basename(file_path)
        label = os.path.splitext(base_name)[0]
        if label in seen_labels:
            print(f"duplicate input label: '{label}'", file=sys.stderr)
            sys.exit(1)
        seen_labels.add(label)
        with open(file_path, 'r') as file:
            content = file.read()
            label_content_pairs.append((label, content))

    return label_content_pairs


@dataclass
class DataStore:
    # from task_id to prompt
    prompts: Dict[str, str]
    # from (model, temperature) -> task_id -> (response_id, class_id, content_file)
    responses: Dict[Tuple[str, str], Dict[str, Tuple[str, str, str]]]


def load_data(path_query):
    def collect_response_dirs(path):
        # Returns list of (task id, prompt file path, response dir)
        result = []
        entries = os.listdir(path)
        md_files = { os.path.splitext(entry)[0]: os.path.join(path, entry)
                     for entry in entries if entry.endswith('.md') }
        directories = { entry: os.path.join(path, entry)
                        for entry in entries if os.path.isdir(os.path.join(path, entry)) }
        for name, md_file_path in md_files.items():
            if name in directories:
                result.append((name, md_file_path, directories[name]))
        return result

    def collect_response_files(response_dir):
        response_files = []
        for file_name in os.listdir(response_dir):
            full_path = os.path.join(response_dir, file_name)
            if os.path.isfile(full_path):
                file_parts = file_name.rsplit('.', 1)
                name_parts = file_parts[0].split('_')
                response_id = name_parts[0]
                class_id = name_parts[-1] if len(name_parts) > 1 else None
                response_files.append((response_id, class_id, full_path))
        return response_files

    def load_prompts_and_responses(data):
        # Accepts list of (task_id, prompt_file, response_dir)
        # Returns (task_id -> [(response_id, class_id, content_file)],
        #          task_id -> prompt)
        responses = dict()
        prompts = dict()
        for (task_id, md_file_path, response_dir) in data:
            with open(md_file_path, 'r') as f:
                prompts[task_id] = f.read()
            responses[task_id] = collect_response_files(response_dir)
        return (responses, prompts)

    def pick_directory(path):
        subdirectories = [
            d for d in os.listdir(path) 
            if os.path.isdir(os.path.join(path, d))
        ]
        if not subdirectories:
            return None
        else:
            return os.path.join(path,subdirectories[0])

    data = collect_response_dirs(path_query)
    if len(data) > 0:
        # this is a model directory
        responses, prompts = load_prompts_and_responses(data)
        last_dir = os.path.basename(os.path.normpath(path_query))
        model_name, temperature = tuple(last_dir.rsplit('_', 1))
        return DataStore(prompts=prompts,
                         responses={(model_name, temperature): responses})
    else:
        # this is either a top-level directory, or a response directory
        some_dir = pick_directory(path_query)
        if some_dir:
            data = collect_response_dirs(some_dir)
            if len(data) > 0:
                pass
                # this is a top-level directory
            else:
                # we assume there should be no subdirectories in the response directory:
                print("failed to interpret data path", file=sys.stderr)
                exit(1)
        else:
            # this is a response directory
            task_id = os.path.basename(os.path.normpath(path_query))
            parent_dir = os.path.dirname(os.path.normpath(path_query))
            model_temp = os.path.basename(os.path.normpath(parent_dir))
            model_name, temperature = tuple(model_temp.rsplit('_', 1))
            data = [(task_id, parent_dir + f"/{task_id}.md", path_query)]
            responses, prompts = load_prompts_and_responses(data)
            return DataStore(prompts=prompts,
                         responses={(model_name, temperature): responses})

def main():
    user_config_file = os.path.expanduser("~") + "/.llm_query.yaml"

    if os.path.isfile(user_config_file):
        with open(user_config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = yaml.safe_load(DEFAULT_CONFIG)

    arguments = parse_args()
    validate_arguments(arguments)

    if arguments.setup:
        t = InquirerPy.prompt([
                {
                    "type": "list",
                    "name": "model",
                    "message": "Query model:",
                    "choices": [entry['name'] for entry in config['models']],
                    "default": config['default']['model']
                },
                {
                    "type": "input",
                    "name": "temperature",
                    "message": "Set model temperature:",
                    "default": str(config['default']['temperature'])
                },
                {
                    "type": "list",
                    "name": "extractor",
                    "message": "Extract asnwers with:",
                    "choices": config['extractors'],
                    "default": config['default']['extractor']
                },
                {
                    "type": "list",
                    "name": "equivalence",
                    "message": "Cluster answers with:",
                    "choices": config['equivalences'],
                    "default": config['default']['equivalence']
                },
        ])
        if t:
            config['default'] = t
        with open(user_config_file, 'w') as f:
            yaml.dump(config, f, width=float("inf"))
    else:

        if arguments.query:
            inputs = [(UNNAMED_TASK, arguments.query)]
        elif (not arguments.inputs and
              not arguments.query):
            inputs = [(UNNAMED_TASK, sys.stdin.read())]
        else:
            inputs = process_input_files(arguments.inputs)

        settings = dict()

        if arguments.models:
            settings['models'] = arguments.models
        else:
            settings['models'] = [config['default']['model']]

        if arguments.temperature:
            settings['temperature'] = str(arguments.temperature)
        else:
            settings['temperature'] = config['default']['temperature']

        if arguments.num_responses:
            settings['num_responses'] = arguments.num_responses
        else:
            settings['num_responses'] = 1

        if arguments.extractor:
            settings['extractor'] = arguments.extractor
        else:
            settings['extractor'] = config['default']['extractor']

        if arguments.answer:
            settings['extractor'] = ANSWER_EXTRACTOR
            new_inputs = []
            for label, prompt in inputs:
                new_inputs.append((label, prompt + " " + ANSWER_FORMAT_DIRECTIVE))
            inputs = new_inputs

        if arguments.code:
            settings['extractor'] = CODE_EXTRACTOR

        if arguments.extract:
            print(load_data(arguments.extract))
            exit(1)

        if (settings['num_responses'] <= 1 and
            len(settings['models']) <= 1 and
            inputs[0][0] == UNNAMED_TASK and
            not arguments.output):
            if settings['extractor'] == '__ID__':
                stream_response(inputs[0][1], settings['models'][0], settings['temperature'], config)
            else:
                response = get_responses(inputs[0][1], settings['models'][0], settings['temperature'], 1, config)[0]
                answer = extract(response, inputs[0][1], settings['extractor'], UNNAMED_TASK)
                print(answer, end="")
                if os.isatty(sys.stdout.fileno()):
                    print()
        else:
            execute_batch_jobs(inputs, settings, arguments.output, config)


if __name__ == "__main__":
    main()


# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# # Load the true labels and predicted labels from the CSV files
# ground_truth_file = "ground_truth.csv"  # Path to the ground truth file
# predictions_file = "predictions.csv"  # Path to the predictions file

# # Read the CSV files
# ground_truth = pd.read_csv(ground_truth_file)
# predictions = pd.read_csv(predictions_file)

# # Ensure the column names match the actual structure of your CSV files
# true_labels = ground_truth['true_label']
# predicted_labels = predictions['predicted_label']

# # Compute classification metrics
# accuracy = accuracy_score(true_labels, predicted_labels)
# precision = precision_score(true_labels, predicted_labels, average='weighted')  # Use 'weighted' for multiclass
# recall = recall_score(true_labels, predicted_labels, average='weighted')
# f1 = f1_score(true_labels, predicted_labels, average='weighted')
# conf_matrix = confusion_matrix(true_labels, predicted_labels)

# # Generate a comprehensive classification report
# class_report = classification_report(true_labels, predicted_labels)

# # Print the results
# print("Classification Metrics:")
# print(f"Accuracy: {accuracy:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1:.2f}")
# print("\nConfusion Matrix:")
# print(conf_matrix)

# print("\nClassification Report:")
# print(class_report)
