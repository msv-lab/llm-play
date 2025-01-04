# llm-query is for interactively defining and executing experimental pipelines with LLMs.
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
from typing import Dict, Tuple, List

import shlex
import os
import sys
import shutil
import configparser
import argparse
import tempfile
import subprocess
import re
from collections import deque

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
    support_multiple_responses: True
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


@dataclass
class Task:
    id: str
    prompt: str

    @staticmethod
    def unnamed(prompt):
        return Task('__unnamed__', prompt)

    def is_unnamed(self):
        return self.id == '__unnamed__'


@dataclass
class LLMQuery:
    models: List[str]
    temperature: str
    tasks: List[Task]
    num_responses: int


@dataclass
class StreamItem: 
    model: str
    temperature: str
    task: Task
    response_id: int
    class_id: int
    content: str


def get_provider_by_model(model, config):
    for m in config['models']:
        if model == m['name']:
            return m['provider']
    raise ValueError(f"no provider for model {model}")


class LLMResponseStream:
    def __init__(self, query, config):
        self.temperature = query.temperature
        self.config = config
        self.current_item_index = 0
        self.current_response_index = 0
        self.cache = deque()
        self.execution_plan = deque()
        for m in query.models:
            for t in query.tasks:
                self.execution_plan.append((m, t, query.num_responses))
        self.size = len(query.models) * len(query.tasks) * query.num_responses

    def __iter__(self):
        return self

    def _generate_responses(self, model, prompt, temperature, n):
        provider = get_provider_by_model(model, self.config)
        client = OpenAI(
            api_key=os.getenv(self.config['providers'][provider]['key_env_variable']),
            base_url=self.config['providers'][provider]['base_url'],
        )
        if self.config['providers'][provider]['support_multiple_responses']:
            completion = client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                n=n
            )
            return [c.message.content for c in completion.choices]
        else:
            completion = client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                n=1
            )
            return [completion.choices[0].message.content]

    def __next__(self):
        if self.current_item_index >= len(self):
            raise StopIteration
        self.current_item_index += 1
        if len(self.cache) == 0:
            model, task, n = self.execution_plan.pop()
            responses = self._generate_responses(model, task.prompt, self.temperature, n)
            for r in responses:
                self.cache.append((model, task, self.current_response_index, r))
                self.current_response_index += 1
            if n > len(responses):
                self.execution_plan.append((model, task, n - len(responses)))
            else:
                self.current_response_index = 0
        model, task, response_id, response = self.cache.pop()
        return StreamItem(
            model = model,
            temperature = self.temperature,
            task = task,
            response_id = response_id,
            class_id = response_id,
            content = response)

    def next_batch(self):
        return self.current_index > 0 and len(self.current_response_index) == 0

    def __len__(self):
        return self.size


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


def execute_batch_jobs(query, settings, output, config):
    recreate_directory(output)
    stream = LLMResponseStream(query, config)
    if len(stream) > 1:
        pbar = tqdm(total=len(stream), ascii=True)
    for i in stream:
        model_path = os.path.join(output, f"{i.model}_{i.temperature}")
        os.makedirs(model_path, exist_ok=True)
        prompt_file = os.path.join(model_path, f"{i.task.id}.md")
        if not os.path.exists(prompt_file):
            with open(prompt_file, "w") as file:
                file.write(i.task.prompt)
        task_path = os.path.join(model_path, i.task.id)
        os.makedirs(task_path, exist_ok=True)
        if settings['extractor'] != '__ID__':
            result = extract(i.content, i.task, settings['extractor'])
        else:
            result = i.content
        with open(os.path.join(task_path, f"{i.response_id}.md"), "w") as file:
            file.write(result)
            if len(stream) > 1:
                pbar.update()
    if len(stream) > 1:
        pbar.close()


def instantiate_shell_template(t, task, prompt_file, outputs, output_files):
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
    t = t.replace(f"%%PROMPT%%", render(task.prompt, 0))
    t = t.replace(f"%%ESCAPED_PROMPT%%", render(task.prompt, escape=True))
    t = t.replace(f"%%PROMPT_FILE%%", render(prompt_file))
    t = t.replace(f"%%ESCAPED_PROMPT_FILE%%", render(prompt_file, escape=True))
    t = t.replace(f"%%TASK_ID%%", render(task.id))
    t = t.replace(f"%%ESCAPED_TASK_ID%%", render(task.id, escape=True))

    return t


def extract(response, task, extractor):
    with tempfile.NamedTemporaryFile() as prompt_file:
        prompt_file.write(prompt.encode())
        prompt_file.flush()
        with tempfile.NamedTemporaryFile() as output_file:
            output_file.write(response.encode())
            output_file.flush()
            cmd = instantiate_shell_template(extractor, task, prompt_file.name, outputs=[response], output_files=[output_file.name])
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            return result.stdout


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Query Interface")
    parser.add_argument("query", nargs='?', type=str, help="Query string")
    parser.add_argument("--prompt", nargs='+', type=str, help="Prompt files")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--model", nargs='+', type=str, help="List of models to query")
    parser.add_argument("--temperature", type=float, help="Temperature for model generation")
    parser.add_argument("-n", "--num-responses", type=int, help="Number of responses to generate")
    parser.add_argument("--extract", type=str, help="Extract data in directory")
    parser.add_argument("--extractor", type=str, help="Answer extraction shell command")
    parser.add_argument("--extension", type=str, help="File extension for extracted data")
    parser.add_argument("--answer", action="store_true", help="Answer question")
    parser.add_argument("--code", action="store_true", help="Generate code")
    parser.add_argument("--distribution", type=str, help="Compute distribution of responses in the directory")
    parser.add_argument("--equivalence", type=str, help="Equivalence relation shell command")
    parser.add_argument("--evaluate", type=str, help="Evaluate responses in the directory")
    parser.add_argument("--equal", type=str, help="Check equivalence of responses to the specified value")
    parser.add_argument("--evalautor", type=str, help="Evaluator shell command")
    parser.add_argument("--predicate", action="store_true", help="Evaluate truthfulness of the predicate")
    parser.add_argument("--export", type=str, help="Export data from directory")
    parser.add_argument("--report", type=str, help="Output CSV or JSON file")
    parser.add_argument("-c", "--configure", action="store_true", help="Set default options")
    return parser.parse_args()


def validate_arguments(arguments):
    if (((arguments.num_responses and arguments.num_responses > 1) or
         (arguments.model and len(arguments.model) > 1) or
         (arguments.prompt and len(arguments.prompt) > 1)) and
        not arguments.output):
        print("for multiple responses/models/prompts, the output directory needs to be specified", file=sys.stderr)
        exit(1)

    if sum([bool(arguments.query),
            bool(arguments.prompt),
            bool(arguments.distribution),
            bool(arguments.evaluate)]) > 1:
        print("choose only one of (1) query string, (2) prompt files, (3) response distribution, (4) response evaluation", file=sys.stderr)
        exit(1)

    if ((arguments.answer or arguments.code) and arguments.extractor):
        print("the answer/code options cannot be used with a custom extractor", file=sys.stderr)
        exit(1)

    if (arguments.code and arguments.answer):
        print("the code and the answer options are mutually exclusive", file=sys.stderr)
        exit(1)


def process_prompt_files(file_list):
    tasks = []
    seen_labels = set()

    for file_path in file_list:
        base_name = os.path.basename(file_path)
        label = os.path.splitext(base_name)[0]
        if label in seen_labels:
            print(f"duplicate prompt label: '{label}'", file=sys.stderr)
            sys.exit(1)
        seen_labels.add(label)
        with open(file_path, 'r') as file:
            content = file.read()
            tasks.append(Task(label, content))

    return tasks


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
                subdirectories = [
                    d for d in os.listdir(path_query) 
                    if os.path.isdir(os.path.join(path, d))
                ]
                all_responses = dict()
                prompts = dict()
                for model_dir in subdirectories:                    
                    data = collect_response_dirs(os.path.join(path_query, model_dir))
                    assert(len(data) > 0)
                    model_name, temperature = tuple(last_dir.rsplit('_', 1))
                    responses, prompts = load_prompts_and_responses(data)
                    all_responses[(model_name, temperature)] = responses
                return DataStore(prompts=prompts,
                                 responses=all_responses)
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
            data = [(task_id, os.path.join(parent_dir, f"/{task_id}.md"), path_query)]
            responses, prompts = load_prompts_and_responses(data)
            return DataStore(prompts=prompts,
                             responses={(model_name, temperature): responses})

def main():
    user_config_file = os.path.join(os.path.expanduser("~"), ".llm_query.yaml")

    if os.path.isfile(user_config_file):
        with open(user_config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = yaml.safe_load(DEFAULT_CONFIG)

    arguments = parse_args()
    validate_arguments(arguments)

    if arguments.configure:
        choice = InquirerPy.prompt([
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
                    "message": "Data extractor for --extract:",
                    "choices": config['extractors'],
                    "default": config['default']['extractor']
                },
                {
                    "type": "list",
                    "name": "equivalence",
                    "message": "Relation for --cluster/--diff/--equal:",
                    "choices": config['equivalences'],
                    "default": config['default']['equivalence']
                },
        ])
        if choice:
            config['default'] = choice
        with open(user_config_file, 'w') as f:
            yaml.dump(config, f, width=float("inf"))
    else:

        if arguments.query:
            tasks = [Task.unnamed(arguments.query)]
        elif (not arguments.prompt and not arguments.query):
            tasks = [Task.unnamed(sys.stdin.read())]
        else:
            tasks = process_prompt_files(arguments.prompt)

        query = LLMQuery(
            models = arguments.model if arguments.model else [config['default']['model']],
            temperature = str(arguments.temperature) if arguments.temperature else config['default']['temperature'],
            num_responses = arguments.num_responses if arguments.num_responses else 1,
            tasks = tasks,
        )

        settings = dict()

        if arguments.extractor:
            settings['extractor'] = arguments.extractor
        else:
            settings['extractor'] = config['default']['extractor']

        if arguments.answer:
            settings['extractor'] = ANSWER_EXTRACTOR
            new_tasks = []
            for t in tasks:
                new_tasks.append(Task(t.id, t.prompt + " " + ANSWER_FORMAT_DIRECTIVE))
            tasks = new_tasks

        if arguments.code:
            settings['extractor'] = CODE_EXTRACTOR

        if arguments.extract:
            print(load_data(arguments.extract))
            exit(1)

        if (query.num_responses <= 1 and
            len(query.models) <= 1 and
            tasks[0].is_unnamed() and
            not arguments.output):
            if settings['extractor'] == '__ID__':
                stream_response(tasks[0].prompt, query.models[0], query.temperature, config)
            else:
                i = next(LLMResponseStream(query, config))
                answer = extract(i.content, i.task, settings['extractor'])
                print(answer, end="")
                if os.isatty(sys.stdout.fileno()):
                    print()
        else:
            execute_batch_jobs(query, settings, arguments.output, config)


if __name__ == "__main__":
    main()
