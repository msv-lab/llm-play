# llm-play is a tool that queries LLMs and executes experimental pipelines.
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
import InquirerPy
from openai import OpenAI


ANSWER_FORMAT_DIRECTIVE = "Wrap the final answer with <answer> </answer>."
ANSWER_EXTRACTOR = r"sed -n '0,/<\/answer>/s/.*<answer>\(.*\)<\/answer>.*/\1/p' %%ESCAPED_DATA_FILE%%"
CODE_EXTRACTOR = r"awk '/^```/{if (!found++) { while(getline && $0 !~ /^```/) print; exit}}' %%ESCAPED_DATA_FILE%%"


DEFAULT_CONFIG = fr"""
default:
  models:
    - qwen2.5-72b-instruct
  temperature: 1.0
  extractor: __ID__
  equivalence: __ID__
providers:
  Aliyun:
    API: OpenAI
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    key_env_variable: DASHSCOPE_API_KEY
    support_multiple_samples: True
  DeepSeek:
    API: OpenAI
    base_url: https://api.deepseek.com
    key_env_variable: DEEPSEEK_API_KEY
    support_multiple_samples: False
  CloseAI_OpenAI:
    API: OpenAI
    base_url: https://api.openai-proxy.org/v1
    key_env_variable: CLOSEAI_API_KEY
    support_multiple_samples: True
  CloseAI_Anthropic:
    API: Anthropic
    base_url: https://api.openai-proxy.org/anthropic
    key_env_variable: CLOSEAI_API_KEY
    support_multiple_samples: True
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
    llm-play --model qwen2.5-72b-instruct 'Are these two answers equivalent: "%%DATA1%%" and "%%DATA2%%"?' --predicate
"""


DATA_STREAM_TABLE_HEADER = ['Model', 'Temp.', 'Task', 'Sample', 'Content']

USER_CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".llm_play.yaml")

CSV_TRUNCATE_LENGTH = 30


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
    num_samples: int


@dataclass
class StreamItem: 
    model: str
    temperature: str
    task: Task
    sample_id: int
    class_id: int
    content: str


def get_provider_by_model(model, config):
    for m in config['models']:
        if model == m['name']:
            return m['provider']
    raise ValueError(f"no provider for model {model}")


class LLMStream:
    def __init__(self, query, config):
        self.temperature = query.temperature
        self.config = config
        self.current_item_index = 0
        self.current_sample_index = 0
        self.cache = deque()
        self.execution_plan = deque()
        for m in query.models:
            for t in query.tasks:
                self.execution_plan.append((m, t, query.num_samples))
        self.size = len(query.models) * len(query.tasks) * query.num_samples

        max_model_name_len = max(len(m) for m in query.models)
        max_task_name_len = max(len(t.id) for t in query.tasks)
        self.table_format = [
            (max(max_model_name_len, len(DATA_STREAM_TABLE_HEADER[0])), 'l'),
            (len(DATA_STREAM_TABLE_HEADER[1]), 'r'),
            (min(max(max_task_name_len, len(DATA_STREAM_TABLE_HEADER[2])), 20), 'l'),
            (len(DATA_STREAM_TABLE_HEADER[3]), 'r'),
            (None, 'l'),
        ]

    def __iter__(self):
        return self

    def _sample(self, model, prompt, temperature, n):
        provider = get_provider_by_model(model, self.config)
        client = OpenAI(
            api_key=os.getenv(self.config['providers'][provider]['key_env_variable']),
            base_url=self.config['providers'][provider]['base_url'],
        )
        if self.config['providers'][provider]['support_multiple_samples']:
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
            samples = self._sample(model, task.prompt, self.temperature, n)
            for s in samples:
                self.cache.append((model, task, self.current_sample_index, s))
                self.current_sample_index += 1
            if n > len(samples):
                self.execution_plan.append((model, task, n - len(samples)))
            else:
                self.current_sample_index = 0
        model, task, sample_id, sample = self.cache.pop()
        return StreamItem(
            model = model,
            temperature = self.temperature,
            task = task,
            sample_id = sample_id,
            class_id = sample_id,
            content = sample)

    def next_batch(self):
        return self.current_index > 0 and len(self.current_sample_index) == 0

    def table_format(self):
        return self.table_format

    def table_header(self):
        return DATA_STREAM_TABLE_HEADER

    def __len__(self):
        return self.size


def stream_response_to_stdout(prompt, model, temperature, config):
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


class JSONStream:
    ```
    {
        "prompts":
        "model_id": {
            "temperature": {
            }
            
        },
        ...
    }
    ```
    def __init__(self, data):
        self.current_item_index = 0
        self.current_sample_index = 0

        self.size = 

        max_model_name_len = max(len(m) for m in query.models)
        max_task_name_len = max(len(t.id) for t in query.tasks)
        self.table_format = 

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_item_index >= len(self):
            raise StopIteration
        self.current_item_index += 1
        if len(self.cache) == 0:
            model, task, n = self.execution_plan.pop()
            samples = self._sample(model, task.prompt, self.temperature, n)
            for s in samples:
                self.cache.append((model, task, self.current_sample_index, s))
                self.current_sample_index += 1
            if n > len(samples):
                self.execution_plan.append((model, task, n - len(samples)))
            else:
                self.current_sample_index = 0
        model, task, sample_id, sample = self.cache.pop()
        return StreamItem(
            model = model,
            temperature = self.temperature,
            task = task,
            sample_id = sample_id,
            class_id = sample_id,
            content = sample)

    def next_batch(self):
        return self.current_index > 0 and len(self.current_sample_index) == 0

    def table_format(self):
        return self.table_format

    def table_header(self):
        return DATA_STREAM_TABLE_HEADER

    def __len__(self):
        return self.size
        


def recreate_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def to_single_line(text):
    lines = text.splitlines()
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    result = " ".join(non_empty_lines)
    return result


def execute_batch_jobs(query, settings, output, config):
    recreate_directory(output)
    stream = LLMStream(query, config)
    if len(stream) > 1:
        #TODO: check if terminal
        printer = TablePrinter(stream.table_format(), DATA_STREAM_TABLE_HEADER)
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
        with open(os.path.join(task_path, f"{i.sample_id}.md"), "w") as file:
            file.write(result)
            if len(stream) > 1:
                row = (i.model, i.temperature, i.task.id, i.sample_id, to_single_line(result))
                printer.print_row(row)


def instantiate_shell_template(t, task, prompt_file, data, data_files):
    assert(len(data) == len(data_files))

    def render(value, escape=False):
        return shlex.quote(value) if escape and value is not None else value

    if len(data) == 1:
        t = t.replace(f"%%DATA%%", render(data[0]))
        t = t.replace(f"%%ESCAPED_DATA%%", render(data[0], escape=True))
        t = t.replace(f"%%DATA_FILE%%", render(data_files[0]))
        t = t.replace(f"%%ESCAPED_DATA_FILE%%", render(data_files[0], escape=True))
    else:
        for i in range(len(data), 0, -1):
            t = t.replace(f"%%DATA{i}%%", render(data[i - 1]))
            t = t.replace(f"%%ESCAPED_DATA{i}%%", render(data[i - 1], escape=True))
            t = t.replace(f"%%DATA_FILE{i}%%", render(data_files[i - 1]))
            t = t.replace(f"%%ESCAPED_DATA_FILE{i}%%", render(data_files[i - 1], escape=True))
    t = t.replace(f"%%PROMPT%%", render(task.prompt, 0))
    t = t.replace(f"%%ESCAPED_PROMPT%%", render(task.prompt, escape=True))
    t = t.replace(f"%%PROMPT_FILE%%", render(prompt_file))
    t = t.replace(f"%%ESCAPED_PROMPT_FILE%%", render(prompt_file, escape=True))
    t = t.replace(f"%%TASK_ID%%", render(task.id))
    t = t.replace(f"%%ESCAPED_TASK_ID%%", render(task.id, escape=True))

    return t


def extract(sample, task, extractor):
    with tempfile.NamedTemporaryFile() as prompt_file:
        prompt_file.write(task.prompt.encode())
        prompt_file.flush()
        with tempfile.NamedTemporaryFile() as data_file:
            data_file.write(sample.encode())
            data_file.flush()
            cmd = instantiate_shell_template(extractor, task, prompt_file.name, data=[sample], data_files=[data_file.name])
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            return result.stdout


class TablePrinter:

    def __init__(self, column_widths, headers):
        """column_widths is a list of max widths for columns and
        alignment options ('l' or 'r'). If a column width is None, the
        column will auto-adjust to fill the terminal width.

        """
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        num_columns = len(column_widths)

        fixed_widths = [w[0] for w in column_widths if w[0] is not None]
        fixed_total = sum(fixed_widths) + (num_columns - 1) * 3
        flexible_columns = column_widths.count((None, 'l')) + column_widths.count((None, 'r'))
        #TODO what if overflow?
        if flexible_columns > 0:
            available_width = max(terminal_width - fixed_total, 0)
            flexible_width = available_width // flexible_columns
            column_widths = [w if w[0] is not None else (flexible_width, w[1]) for w in column_widths]
        self.column_widths = column_widths
        self.print_row(headers)
        sep = []
        for (w, _) in column_widths:
            sep.append(u'\u2500'*w)
        print(("" + u'\u2500' +  u'\u253C' + u'\u2500').join(sep))

    def _truncate(self, content, width):
        if len(content) > width:
            return content[:width - 3] + '...'  # Truncate and add ellipsis
        return content

    def print_row(self, row):
        formatted_row = []
        for i, cell in enumerate(row):
            col_width, alignment = self.column_widths[i]
            cell_content = str(cell).splitlines()[0]
            if alignment == 'l':
                formatted_row.append(self._truncate(cell_content, col_width).ljust(col_width))
            else:
                formatted_row.append(self._truncate(cell_content, col_width).rjust(col_width))
        print((" " + u'\u2502' + " ").join(formatted_row))


def parse_args():
    parser = argparse.ArgumentParser(description="llm-play interface")
    parser.add_argument("query", nargs='?', type=str, help="Query string")
    parser.add_argument("--prompt", nargs='+', type=str, help="Prompt files")
    parser.add_argument("--output", type=str, help="Output FS-tree/JSON/CSV")
    parser.add_argument("--update", type=str, help="FS-tree/JSON to update")
    parser.add_argument("--model", nargs='+', type=str, help="List of models to query")
    parser.add_argument("-t", "--temperature", type=float, help="Temperature for model generation")
    parser.add_argument("-n", "--num-samples", type=int, help="Number of samples to generate")
    parser.add_argument("--map", type=str, help="Transform given data")
    parser.add_argument("--transformer", type=str, help="Data transformation shell command")
    parser.add_argument("--extension", type=str, help="File extension for transformed data")
    parser.add_argument("--answer", action="store_true", help="Extract answer")
    parser.add_argument("--code", action="store_true", help="Extract code")
    parser.add_argument("--distribution", type=str, help="Show distribution of samples")
    parser.add_argument("--cluster", type=str, help="Cluster given data")
    parser.add_argument("--equivalence", type=str, help="Equivalence relation shell command")
    parser.add_argument("--equal", type=str, help="Check equivalence of data to the specified value")
    parser.add_argument("--predicate", action="store_true", help="Evaluate truthfulness of the predicate")
    parser.add_argument("--quiet", action="store_true", help="Do not print data on stdout")
    parser.add_argument("--debug", action="store_true", help="Print logs on stderr")
    parser.add_argument("-c", "--configure", action="store_true", help="Set default options")
    return parser.parse_args()


def validate_arguments(arguments):
    if (((arguments.num_samples and arguments.num_samples > 1) or
         (arguments.model and len(arguments.model) > 1) or
         (arguments.prompt and len(arguments.prompt) > 1)) and
        not arguments.output):
        print("for multiple samples/models/prompts, the output directory needs to be specified", file=sys.stderr)
        exit(1)

    if sum([bool(arguments.query),
            bool(arguments.prompt),
            bool(arguments.distribution),
            bool(arguments.evaluate)]) > 1:
        print("choose only one of (1) query string, (2) prompt files, (3) sample distribution, (4) sample evaluation", file=sys.stderr)
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
    # from (model, temperature) -> task_id -> (sample_id, class_id, content_file)
    samples: Dict[Tuple[str, str], Dict[str, Tuple[str, str, str]]]


def load_data(path_query):
    def collect_sample_dirs(path):
        # Returns list of (task id, prompt file path, sample dir)
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

    def collect_sample_files(sample_dir):
        sample_files = []
        for file_name in os.listdir(sample_dir):
            full_path = os.path.join(sample_dir, file_name)
            if os.path.isfile(full_path):
                file_parts = file_name.rsplit('.', 1)
                name_parts = file_parts[0].split('_')
                sample_id = name_parts[0]
                class_id = name_parts[-1] if len(name_parts) > 1 else None
                sample_files.append((sample_id, class_id, full_path))
        return sample_files

    def load_prompts_and_samples(data):
        # Accepts list of (task_id, prompt_file, sample_dir)
        # Returns (task_id -> [(sample_id, class_id, content_file)],
        #          task_id -> prompt)
        samples = dict()
        prompts = dict()
        for (task_id, md_file_path, sample_dir) in data:
            with open(md_file_path, 'r') as f:
                prompts[task_id] = f.read()
            samples[task_id] = collect_sample_files(sample_dir)
        return (samples, prompts)

    def pick_directory(path):
        subdirectories = [
            d for d in os.listdir(path) 
            if os.path.isdir(os.path.join(path, d))
        ]
        if not subdirectories:
            return None
        else:
            return os.path.join(path,subdirectories[0])

    data = collect_sample_dirs(path_query)
    if len(data) > 0:
        # this is a model directory
        samples, prompts = load_prompts_and_samples(data)
        last_dir = os.path.basename(os.path.normpath(path_query))
        model_name, temperature = tuple(last_dir.rsplit('_', 1))
        return DataStore(prompts=prompts,
                         samples={(model_name, temperature): samples})
    else:
        # this is either a top-level directory, or a sample directory
        some_dir = pick_directory(path_query)
        if some_dir:
            data = collect_sample_dirs(some_dir)
            if len(data) > 0:
                subdirectories = [
                    d for d in os.listdir(path_query) 
                    if os.path.isdir(os.path.join(path, d))
                ]
                all_samples = dict()
                prompts = dict()
                for model_dir in subdirectories:                    
                    data = collect_sample_dirs(os.path.join(path_query, model_dir))
                    assert(len(data) > 0)
                    model_name, temperature = tuple(last_dir.rsplit('_', 1))
                    samples, prompts = load_prompts_and_samples(data)
                    all_samples[(model_name, temperature)] = samples
                return DataStore(prompts=prompts,
                                 samples=all_samples)
            else:
                # we assume there should be no subdirectories in the sample directory:
                print("failed to interpret data path", file=sys.stderr)
                exit(1)
        else:
            # this is a sample directory
            task_id = os.path.basename(os.path.normpath(path_query))
            parent_dir = os.path.dirname(os.path.normpath(path_query))
            model_temp = os.path.basename(os.path.normpath(parent_dir))
            model_name, temperature = tuple(model_temp.rsplit('_', 1))
            data = [(task_id, os.path.join(parent_dir, f"/{task_id}.md"), path_query)]
            samples, prompts = load_prompts_and_samples(data)
            return DataStore(prompts=prompts,
                             samples={(model_name, temperature): samples})


def configure(config):
    model_choices = []
    for m in [entry['name'] for entry in config['models']]:
        if m in config['default']['models']:
            model_choices.append(InquirerPy.base.Choice(m, enabled=True))
        else:
            model_choices.append(InquirerPy.base.Choice(m, enabled=False))            
    selected = InquirerPy.prompt([
            {
                "type": "checkbox",
                "name": "models",
                "message": "Models:",
                "choices": model_choices,
                "validate": lambda result: len(result) >= 1,
                "invalid_message": "should be at least 1 selection"
            },
            {
                "type": "input",
                "name": "temperature",
                "message": "Sampling temperature:",
                "default": str(config['default']['temperature'])
            },
            {
                "type": "list",
                "name": "extractor",
                "message": "Transformer for --map:",
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
    if selected:
        config['default'] = selected
    with open(USER_CONFIG_FILE, 'w') as f:
        yaml.dump(config, f, width=float("inf"))
    

def main():
    if os.path.isfile(USER_CONFIG_FILE):
        with open(USER_CONFIG_FILE, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = yaml.safe_load(DEFAULT_CONFIG)

    arguments = parse_args()
    validate_arguments(arguments)

    if arguments.configure:
        configure(config)
    else:

        if arguments.query:
            tasks = [Task.unnamed(arguments.query)]
        elif (not arguments.prompt and not arguments.query):
            tasks = [Task.unnamed(sys.stdin.read())]
        else:
            tasks = process_prompt_files(arguments.prompt)

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

        query = LLMQuery(
            models = arguments.model if arguments.model else config['default']['models'],
            temperature = str(arguments.temperature) if arguments.temperature else config['default']['temperature'],
            num_samples = arguments.num_samples if arguments.num_samples else 1,
            tasks = tasks
        )

        if arguments.extract:
            print(load_data(arguments.extract))
            exit(1)

        if len(query.tasks) * len(query.models) * len(query.num_samples) == 1 and not arguments.output:
            if settings['extractor'] == '__ID__':
                stream_response_to_stdout(tasks[0].prompt, query.models[0], query.temperature, config)
            else:
                i = next(LLMStream(query, config))
                answer = extract(i.content, i.task, settings['extractor'])
                print(answer, end="")
                if os.isatty(sys.stdout.fileno()):
                    print()
        else:
            execute_batch_jobs(query, settings, arguments.output, config)


if __name__ == "__main__":
    main()
