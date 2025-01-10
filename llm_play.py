# llm-play is a tool that queries LLMs, analyzes responses, and
# executes experimental pipelines.
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
import hashlib
from pathlib import Path
import json

import yaml
import InquirerPy
from openai import OpenAI
from wcwidth import wcwidth, wcswidth
import mistletoe


VERSION = "0.0.0"

DEFAULT_MODEL = "qwen2.5-7b-instruct"

ANSWER_DIRECTIVE = "Wrap the final answer with <answer></answer>."

PREDICATE_DIRECTIVE = "Respond Yes or No."

LLM_BASED_AFFIRMATION_CLASSIFIER = rf"llm-play '<answer>'%%CONDENSED_ESCAPED_DATA%%'</answer>. Is this answer affirmative? Respond Yes or No.' --model {DEFAULT_MODEL} --answer"

LLM_BASED_EQUIVALENCE_CHECKER = rf"llm-play 'Are these two answers equivalent: <answer1>'%%CONDENSED_ESCAPED_DATA1%%'</answer1> and <naswer2>'%%CONDENSED_ESCAPED_DATA2%%'</answer2>?' --model {DEFAULT_MODEL} --predicate"

DEFAULT_CONFIG = rf"""
default:
  models:
    - {DEFAULT_MODEL}
  temperature: 1.0
  function: __ID__
  relation: __ID__
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
functions:
  - __ID__
  - __FIRST_TAGGED_ANSWER__
  - __FIRST_MARKDOWN_CODE_BLOCK__
  - |-
    {LLM_BASED_AFFIRMATION_CLASSIFIER}
relations:
  - __ID__
  - __TRIMMED_CASE_INSENSITIVE__
  - |-
    {LLM_BASED_EQUIVALENCE_CHECKER}
"""

USER_CONFIG_FILE = Path.home() / ".llm_play.yaml"

TRUNCATED_CSV_DATA_LENGTH = 50

CONDENSED_SHELL_DATA_LENGTH = 100


@dataclass
class _FunctionFailure:
    pass

FUNCTION_FAILURE = _FunctionFailure()

@dataclass
class FunctionSuccess:
    value: str

FunctionResult = _FunctionFailure | FunctionSuccess


def extract_first_markdown_code_block(content):
    parsed = mistletoe.Document(content)
    for child in parsed.children:
        if child.__class__.__name__ == "CodeFence":
            return FunctionSuccess(child.children[0].content)
    return FUNCTION_FAILURE


BUILTIN_FUNCTIONS = {
    "__ID__": (lambda x: FunctionSuccess(x)),
    "__FIRST_TAGGED_ANSWER__": (
        lambda s: (FunctionSuccess(s.split("<answer>", 1)[1].split("</answer>", 1)[0])
                   if "<answer>" in s and "</answer>" in s and s.index("<answer>") < s.index("</answer>")
                   else FUNCTION_FAILURE)
    ),
    "__FIRST_MARKDOWN_CODE_BLOCK__": (
        lambda s: extract_first_markdown_code_block(s)
    )
}

RELATION_POSITIVE = "Yes"
RELATION_NEGATIVE = "No"

BUILTIN_RELATIONS = {
    "__ID__": (
        lambda x, y: RELATION_POSITIVE if x == y else RELATION_NEGATIVE
    ),
    "__TRIMMED_CASE_INSENSITIVE__": (
        lambda x, y: RELATION_POSITIVE if x.strip().lower() == y.strip().lower() else RELATION_NEGATIVE
    )
}


@dataclass
class Prompt:
    content: str
    label: str
    hash: str

    @staticmethod
    def unlabelled(content):
        return Prompt.labelled(content, '')

    @staticmethod
    def labelled(content, label):
        output_length = 8
        shake = hashlib.shake_128(content.encode("utf-8"))
        return Prompt(content, label, shake.hexdigest(output_length))


@dataclass
class Distribution:
    model: str
    temperature: str

    def id(self):
        return f"{self.model}_{self.temperature}"

    @staticmethod
    def from_id(id):
        m, t = tuple(id.rsplit("_", 1))
        return Distribution(m, t)


@dataclass
class Sample:
    id: int
    class_id: int
    content: str


@dataclass
class LLMQuery:
    distributions: List[Distribution]
    prompts: List[Prompt]
    num_samples: int


@dataclass
class StreamItem:
    distribution: Distribution
    prompt: Prompt
    sample: Sample


def stream_item_table_format(max_model_name_len, max_prompt_label_len):
    return [
        ("Model", max(max_model_name_len, len("Model")), "l"),
        ("Temp.", len("Temp."), "r"),
        ("Label", min(max(max_prompt_label_len, len("Label")), 20), "l"),
        ("Hash", max(len("Hash"), 10), "l"),
        ("Sample", len("Sample"), "r"),
        ("Class", len("Class"), "r"),
        ("Content", None, "l"),
    ]


def get_provider_by_model(model, config):
    for m in config["models"]:
        if model == m["name"]:
            return m["provider"]
    raise ValueError(f"no provider for model {model}")


class LLMSampleStream:
    def __init__(self, query, config):
        self.config = config
        self.current_item_index = 0
        self.current_sample_index = 0
        self.cache = deque()
        self.execution_plan = deque()
        for d in query.distributions:
            for p in query.prompts:
                self.execution_plan.append((d, p, query.num_samples))
        self.size = len(query.distributions) * len(query.prompts) * query.num_samples

        max_model_name_len = max(len(d.model) for d in query.distributions)
        max_prompt_label_len = max(len(p.label) for p in query.prompts)
        self._table_format = stream_item_table_format(max_model_name_len, max_prompt_label_len)

    def __iter__(self):
        return self

    def _sample(self, distribution, prompt, n):
        provider = get_provider_by_model(distribution.model, self.config)
        client = OpenAI(
            api_key=os.getenv(self.config["providers"][provider]["key_env_variable"]),
            base_url=self.config["providers"][provider]["base_url"],
        )
        if self.config["providers"][provider]["support_multiple_samples"]:
            num_responses = n
        else:
            num_responses = 1
        completion = client.chat.completions.create(
            model=distribution.model,
            temperature=float(distribution.temperature),
            messages=[{"role": "user", "content": prompt.content}],
            n=num_responses
        )
        return [c.message.content for c in completion.choices]

    def __next__(self):
        if self.current_item_index >= len(self):
            raise StopIteration
        self.current_item_index += 1
        if len(self.cache) == 0:
            d, p, n = self.execution_plan.pop()
            samples = self._sample(d, p, n)
            for content in samples:
                s = Sample(id=self.current_sample_index,
                           class_id=self.current_sample_index,
                           content = content)
                self.cache.append((d, p, s))
                self.current_sample_index += 1
            if n > len(samples):
                self.execution_plan.append((d, p, n - len(samples)))
            else:
                self.current_sample_index = 0
        d, p, s = self.cache.popleft()
        return StreamItem(
            distribution=d,
            prompt=p,
            sample=s,
        )

    def next_batch(self):
        return self.current_index > 0 and len(self.current_sample_index) == 0

    def table_format(self):
        return self._table_format

    def __len__(self):
        return self.size


def stream_response_to_stdout(prompt, model, temperature, config):
    provider = get_provider_by_model(model, config)
    client = OpenAI(
        api_key=os.getenv(config["providers"][provider]["key_env_variable"]),
        base_url=config["providers"][provider]["base_url"],
    )
    stream = client.chat.completions.create(
        model=model,
        temperature=float(temperature),
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")

    if os.isatty(sys.stdout.fileno()):
        print()


class JSONDataStream:
    def __init__(self, json_data):
        self.current_item_index = 0
        self._next_batch = False
        max_model_name_len = 0
        max_prompt_label_len = 0
        self.cache = deque()
        for distr_id, prompt_to_samples in json_data["data"].items():
            for prompt_hash, samples in prompt_to_samples.items():
                for sample in samples:
                    d = Distribution.from_id(distr_id)
                    if len(d.model) > max_model_name_len:
                        max_model_name_len = len(d.model)
                    p = Prompt(**next(p for p in json_data["prompts"] if p["hash"] == prompt_hash))
                    if len(p.label) > max_prompt_label_len:
                        max_prompt_label_len = len(p.label)
                    s = Sample(**sample)
                    self.cache.append((d, p, s))
        self.size = len(self.cache)
        self._table_format = stream_item_table_format(max_model_name_len, max_prompt_label_len)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_item_index >= len(self):
            raise StopIteration
        self.current_item_index += 1
        self._next_batch = False
        d, p, s = self.cache.popleft()
        if (len(self.cache) > 0 and
            (self.cache[0][0] != d or self.cache[0][1] != p)):
            self._next_batch = True
        return StreamItem(
            distribution=d,
            prompt=p,
            sample=s,
        )

    def next_batch(self):
        return self._next_batch

    def table_format(self):
        return self._table_format

    def __len__(self):
        return self.size


class Map:
    def __init__(self, stream, function):
        self.stream = stream
        self.function = function

    def __iter__(self):
        return self

    def _call_shell_function(self, sample, prompt, function):
        with tempfile.NamedTemporaryFile() as prompt_file:
            prompt_file.write(prompt.content.encode())
            prompt_file.flush()
            with tempfile.NamedTemporaryFile() as data_file:
                data_file.write(sample.encode())
                data_file.flush()
                cmd = instantiate_shell_template(
                    function,
                    prompts=[prompt],
                    prompt_files=[prompt_file.name],
                    data=[sample],
                    data_files=[data_file.name],
                )
                result = subprocess.run(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                if result.returncode == 0:
                    return FunctionSuccess(result.stdout)
                else:
                    return FUNCTION_FAILURE

    def __next__(self):
        while True:
            i = next(self.stream)
            if self.function in BUILTIN_FUNCTIONS:
                result = BUILTIN_FUNCTIONS[self.function](i.sample.content)
            else:
                result = self._call_shell_function(i.sample.content, i.prompt, self.function)
            if result == FUNCTION_FAILURE:
                continue
            return StreamItem(
                distribution=i.distribution,
                prompt=i.prompt,
                sample=Sample(id=i.sample.id,
                              class_id=i.sample.class_id,
                              content=result.value)
            )

    def next_batch(self):
        return self.stream.next_batch

    def table_format(self):
        return self.stream.table_format()

    def __len__(self):
        return len(self.stream)


def get_fs_tree_writer(output):
    def write(i):
        distr_dir = output / i.distribution.id()
        distr_dir.mkdir(exist_ok=True)
        prompt_file = distr_dir / f"{i.prompt.label}_{i.prompt.hash}.md"
        if not prompt_file.exists():
            prompt_file.write_text(i.prompt.content)
        responses_dir = distr_dir / f"{i.prompt.label}_{i.prompt.hash}"
        responses_dir.mkdir(exist_ok=True)
        (responses_dir / f"{i.sample.id}_{i.sample.class_id}.md").write_text(i.sample.content)
    return write


def fs_tree_to_json(path_query):
    """
    JSON format:
    {
        "prompts": [
            {
               "hash": ...
               "label": ...
               "content": ...
            },
            ...
        ],
        "data": {
            "<model_name>_<temperature>": {
                 "<prompt hash>": [
                      {
                          "id": ...
                          "class_id": ...
                          "content": ...
                      },
                 ]
            }
        },
        ...
    }
    FS-tree format:
    <root>
    ├── <model_name>_<temperature>            # distr dir
    │   ├── <prompt_label>_<prompt_hash>.md   # prompt file
    │   └── <prompt_label>_<prompt_hash>      # sample dir
    │       ├── <sample_id>_<class_id>.md     # sample file
    │       ...
    │       └── <sample_id>_<class_id>.md
    └── <model_name>_<temperature>
        ├── ...
        ...
    """
    def parse_distr_dir(distr_dir):
        """
        all entries such that the name of an md file contains _, and there is a corresponding dir
        """
        for prompt_file in distr_dir.iterdir():
            if prompt_file.is_file() and prompt_file.suffix == ".md":
                sample_dir_name = prompt_file.stem
                if "_" in sample_dir_name:
                    prompt_label, prompt_hash = tuple(sample_dir_name.rsplit("_", 1))
                    sample_dir = distr_dir / sample_dir_name
                    if sample_dir.exists() and sample_dir.is_dir():
                        yield (prompt_label, prompt_hash, prompt_file, sample_dir)

    def collect_samples(sample_dir):
        """
        collect all files in sample_dir in the form <sample_id>_<class_id>.<etc>
        """
        for f in sample_dir.iterdir():
            if f.is_file():
                if f.stem.count('_') == 1:
                    sample_id_str, class_id_str = tuple(f.stem.split("_"))
                    yield {
                        "id": int(sample_id_str),
                        "class_id": int(class_id_str),
                        "content": f.read_text()
                    }

    def load_prompts_and_samples(distr_dir_data):
        """
        returns [ { "hash": ..., "label": ..., "content": ... } ]
        and { "<prompt hash>": [
                  { "id": ..., "class_id": ..., "content": ... },
                  ...
              ] }
        """
        prompts = []
        samples = dict()
        for (prompt_label, prompt_hash, prompt_file, sample_dir) in distr_dir_data:
            prompts.append({
                "hash": prompt_hash,
                "label": prompt_label,
                "content": prompt_file.read_text()
            })
            samples[prompt_hash] = list(collect_samples(sample_dir))
        return (prompts, samples)

    def pick_subdir(path):
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        if len(subdirs) == 0:
            return None
        else:
            return subdirs[0]

    distr_dir_data = list(parse_distr_dir(path_query))
    if len(distr_dir_data) > 0: # path_query is a distr dir
        prompts, samples = load_prompts_and_samples(distr_dir_data)
        return {
            "prompts": prompts,
            "data": {
                path_query.name: samples
            }
        }
    else: # path_query is either the root directory, or a sample directory
        subdirs = [d for d in path_query.iterdir() if d.is_dir()]
        if len(subdirs) > 0:
            # assume there are not subdirs in a sample dir, so path_query is the root directory
            all_prompts = []
            all_samples = dict()
            for distr_dir in subdirs:
                distr_dir_data = list(parse_distr_dir(distr_dir))
                if len(distr_dir_data) > 0:
                    prompts, samples = load_prompts_and_samples(distr_dir_data)
                    all_samples[distr_dir.name] = samples
                    for prompt in prompts:
                        if not any(p["hash"] == prompt["hash"] for p in all_prompts):
                            all_prompts.append(prompt)
            return {
                "prompts": all_prompts,
                "data": all_samples
            }
        else:
            # this is a sample directory
            prompt_label, prompt_hash = tuple(path_query.name.rsplit("_", 1))
            prompt_file = path_query.parent / f"{prompt_label}_{prompt_hash}.md"
            distr_dir_data = [(prompt_label, prompt_hash, prompt_file, path_query)]

            prompts, samples = load_prompts_and_samples(distr_dir_data)
            return {
                "prompts": prompts,
                "data": {
                    path_query.parent.name: samples
                }
            }


def instantiate_shell_template(t, prompts, prompt_files, data, data_files):
    assert len(data) == len(data_files)
    assert len(prompts) == len(prompt_files)

    def render(value, escape=False, truncate=False):
        value = truncate_content(value, CONDENSED_SHELL_DATA_LENGTH) if truncate else value
        value = shlex.quote(value) if escape else value
        return value

    for (s, i) in [("", 0), ("1", 0), ("2", 1)]:
        t = t.replace(f"%%RAW_DATA{s}%%", render(data[i-1]))
        t = t.replace(f"%%ESCAPED_DATA{s}%%", render(data[i-1], escape=True))
        t = t.replace(f"%%CONDENSED_DATA{s}%%", render(data[i-1], truncate=True))
        t = t.replace(f"%%CONDENSED_ESCAPED_DATA{s}%%", render(data[i-1], truncate=True, escape=True))
        t = t.replace(f"%%DATA_FILE{s}%%", render(data_files[i-1]))
        t = t.replace(f"%%ESCAPED_DATA_FILE{s}%%", render(data_files[i-1], escape=True))
        t = t.replace(f"%%PROMPT{s}%%", render(prompts[i-1].content, 0))
        t = t.replace(f"%%ESCAPED_PROMPT{s}%%", render(prompts[i-1].content, escape=True))
        t = t.replace(f"%%CONDENSED_PROMPT{s}%%", render(prompts[i-1].content, truncate=True))
        t = t.replace(f"%%CONDENSED_ESCAPED_PROMPT{s}%%", render(prompts[i-1].content, escape=True, truncate=True))
        t = t.replace(f"%%PROMPT_FILE{s}%%", render(prompt_files[i-1]))
        t = t.replace(f"%%ESCAPED_PROMPT_FILE{s}%%", render(prompt_files[i-1], escape=True))
        t = t.replace(f"%%PROMPT_LABEL{s}%%", render(prompts[i-1].label))
        t = t.replace(f"%%ESCAPED_PROMPT_LABEL{s}%%", render(prompts[i-1].label, escape=True))

    return t


def get_stream_item_printer(stream):
    printer = TablePrinter(stream.table_format())
    def print_item(i):
        row = (
            i.distribution.model,
            i.distribution.temperature,
            i.prompt.label,
            i.prompt.hash,
            i.sample.id,
            i.sample.class_id,
            i.sample.content,
        )
        printer.print_row(row)
    return print_item


def to_single_line(text):
    lines = text.splitlines()
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    result = " ".join(non_empty_lines)
    return result


def truncate_content(content, length):
    if len(content) > length:
        current_length = 0
        result = []
        for char in content:
            if current_length + 1 > length - len("..."):
                break
            current_length += 1
            result.append(char)
        return ''.join(result) + "..."
    return content


class TablePrinter:

    def __init__(self, column_specs):
        """column_specs is a list of headers, max widths for columns
        and alignment options ('l' or 'r'). If a column width is None,
        the column will auto-adjust to fill the terminal width.

        """
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        num_columns = len(column_specs)

        fixed_widths = [w[1] for w in column_specs if w[1] is not None]
        fixed_total = sum(fixed_widths) + (num_columns - 1) * 3
        flexible_columns = len([c for c in column_specs if c[1] is None])
        # TODO what if overflow?
        if flexible_columns > 0:
            available_width = max(terminal_width - fixed_total, 0)
            flexible_width = available_width // flexible_columns
            column_specs = [
                c if c[1] is not None else (c[0], flexible_width, c[2]) for c in column_specs
            ]
        self.column_specs = column_specs
        headers = [c[0] for c in column_specs]
        self.print_row(headers)
        sep = []
        for _, w, _ in column_specs:
            sep.append("\u2500" * w)
        print(("\u2500" + "\u253C" + "\u2500").join(sep))

    def _truncate_displayed(self, content, width):
        if wcswidth(content) > width:
            current_width = 0
            result = []
            for char in content:
                char_width = wcwidth(char)
                if char_width == -1:
                    continue
                if current_width + char_width > width - len("..."):
                    break
                current_width += char_width
                result.append(char)
            return ''.join(result) + "..."
        return content

    def _wc_rjust(self, text, length, padding=' '):
        return padding * max(0, (length - wcswidth(text))) + text

    def _wc_ljust(self, text, length, padding=' '):
        return text + padding * max(0, (length - wcswidth(text)))

    # TODO: test with narrow screens and empty responses
    def print_row(self, row):
        formatted_row = []
        for i, cell in enumerate(row):
            _, col_width, alignment = self.column_specs[i]
            processed = self._truncate_displayed(to_single_line(str(cell)), col_width)
            if alignment == "l":
                formatted_row.append(self._wc_ljust(processed, col_width))
            else:
                formatted_row.append(self._wc_rjust(processed, col_width))
        print((" " + "\u2502" + " ").join(formatted_row))


def parse_args():
    parser = argparse.ArgumentParser(description="llm-play interface")
    parser.add_argument("query", nargs="?", type=str, help="Query string")
    parser.add_argument("--prompt", nargs="+", type=str, help="Prompt files")
    parser.add_argument("--output", type=str, help="Output FS-tree/JSON/CSV")
    parser.add_argument("--update", type=str, help="FS-tree/JSON to update")
    parser.add_argument("--model", nargs="+", type=str, help="List of models to query")
    parser.add_argument(
        "-t", "--temperature", type=float, help="Temperature for model generation"
    )
    parser.add_argument(
        "-n", "--num-samples", type=int, help="Number of samples to generate"
    )
    parser.add_argument("--map", type=str, help="Transform given data")
    parser.add_argument(
        "--function", type=str, help="Data transformation shell command"
    )
    parser.add_argument(
        "--extension", type=str, help="File extension for transformed data"
    )
    parser.add_argument("--answer", action="store_true", help="Extract answer")
    parser.add_argument("--code", action="store_true", help="Extract code")
    parser.add_argument("--distribution", type=str, help="Show distribution of samples")
    parser.add_argument(
        "--partition", type=str, help="Partition data into equivalence classes"
    )
    parser.add_argument(
        "--relation", type=str, help="Equivalence relation shell command"
    )
    parser.add_argument(
        "--equal", type=str, help="Check equivalence of data to the specified value"
    )
    parser.add_argument(
        "--predicate",
        action="store_true",
        help="Evaluate truthfulness of the predicate",
    )
    parser.add_argument(
        "--diff", type=str, help="Compute difference between distributions"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Do not print data on stdout"
    )
    parser.add_argument("--debug", action="store_true", help="Print logs on stderr")
    parser.add_argument("--version", action="store_true", help="Print version")
    parser.add_argument(
        "-c", "--configure", action="store_true", help="Set default options"
    )
    return parser.parse_args()


def process_prompt_files(file_list):
    prompts = []
    seen_labels = set()

    for file_path in file_list:
        base_name = os.path.basename(file_path)
        label = os.path.splitext(base_name)[0]
        if label in seen_labels:
            print(f"duplicate prompt label: '{label}'", file=sys.stderr)
            sys.exit(1)
        seen_labels.add(label)
        with open(file_path, "r") as file:
            content = file.read()
            prompts.append(Prompt.labelled(content, label))

    return prompts


def configure(config):
    model_choices = []
    for m in [entry["name"] for entry in config["models"]]:
        if m in config["default"]["models"]:
            model_choices.append(InquirerPy.base.Choice(m, enabled=True))
        else:
            model_choices.append(InquirerPy.base.Choice(m, enabled=False))
    selected = InquirerPy.prompt(
        [
            {
                "type": "checkbox",
                "name": "models",
                "message": "Models:",
                "choices": model_choices,
                "validate": lambda result: len(result) >= 1,
                "invalid_message": "should be at least 1 selection",
            },
            {
                "type": "input",
                "name": "temperature",
                "message": "Sampling temperature:",
                "default": str(config["default"]["temperature"]),
            },
            {
                "type": "list",
                "name": "function",
                "message": "Function for --map:",
                "choices": config["functions"],
                "default": config["default"]["function"],
            },
            {
                "type": "list",
                "name": "relation",
                "message": "Relation for --partition/--diff/--equal:",
                "choices": config["relations"],
                "default": config["default"]["relation"],
            },
        ]
    )
    if selected:
        config["default"] = selected
    with open(USER_CONFIG_FILE, "w") as f:
        yaml.dump(config, f, width=float("inf"))


def load_data_store(store_path):
    if store_path.is_file() and not store_path.suffix == "json":
        print("unsupported input file", file=sys.stderr)
        exit(1)
    elif store_path.is_file():
        with store_path.open("r") as file:
            return json.load(file)
    else:
        return fs_tree_to_json(store_path)


def print_with_newline_if_tty(content):
    print(content, end="")
    if os.isatty(sys.stdout.fileno()) and not content.endswith("\n"):
        print()


def command_dispatch(arguments, config):
    stream = []
    consumers = []

    conflicting_options = [
        bool(arguments.query),
        bool(arguments.prompt),
        bool(arguments.distribution),
        bool(arguments.map),
        bool(arguments.distribution),
        bool(arguments.partition),
        bool(arguments.diff),
    ]

    if (sum(conflicting_options) > 1):
        print("conflicting commands", file=sys.stderr)
        exit(1)

    if (arguments.answer or arguments.code) and \
       (arguments.distribution or arguments.partition or arguments.diff):
        print(
            "--answer/--code can only be used when sampling LLMs",
            file=sys.stderr,
        )
        exit(1)

    if arguments.map:
        json_data = load_data_store(Path(arguments.map))
        stream = JSONDataStream(json_data)
        function=(
            arguments.function
            if arguments.function
            else config["default"]["function"]
        )
        stream = Map(stream, function)
        consumers.append(get_stream_item_printer(stream))
    elif arguments.partition:
        pass
    elif arguments.diff:
        pass
    elif arguments.distribution:
        pass
    else:
        # sampling command

        if arguments.query:
            prompts = [Prompt.unlabelled(arguments.query)]
        elif not arguments.prompt and not arguments.query:
            prompts = [Prompt.unlabelled(sys.stdin.read())]
        else:
            prompts = process_prompt_files(arguments.prompt)

        if (arguments.code or arguments.answer) and arguments.function:
            print("--function is mutually exclusive with --code/--answer", file=sys.stderr)
            exit(1)

        if arguments.code and arguments.answer:
            print("--code  is mutually exclusive with --answer", file=sys.stderr)
            exit(1)

        function = '__ID__'

        if arguments.answer:
            function = '__FIRST_TAGGED_ANSWER__'
            new_prompts = []
            for p in prompts:
                new_prompts.append(Prompt.labelled(p.content + " " + ANSWER_DIRECTIVE, p.label))
            prompts = new_prompts

        if arguments.code:
            function = '__FIRST_MARKDOWN_CODE_BLOCK__'

        if arguments.function:
            function = arguments.function

        temperature=(
            str(arguments.temperature)
            if arguments.temperature
            else config["default"]["temperature"]
        )
        distributions = []
        for m in arguments.model if arguments.model else config["default"]["models"]:
            distributions.append(Distribution(model=m, temperature=temperature))

        query = LLMQuery(
            distributions=distributions,
            num_samples=arguments.num_samples if arguments.num_samples else 1,
            prompts=prompts,
        )

        if (
            len(query.prompts) * len(query.distributions) * query.num_samples == 1
            and not arguments.output
        ):
            if function == "__ID__":
                stream_response_to_stdout(
                    prompts[0].content,
                    query.distributions[0].model,
                    query.distributions[0].temperature,
                    config
                )
            else:
                i = next(Map(LLMSampleStream(query, config), function))
                print_with_newline_if_tty(i.sample.content)
        else:
            stream = LLMSampleStream(query, config)
            if function != '__ID__':
                 stream = Map(stream, function)
            if (
                len(query.prompts) * len(query.distributions) * query.num_samples == 1
            ):
                consumers.append(lambda i: print_with_newline_if_tty(i.sample.content))
            else:
                consumers.append(get_stream_item_printer(stream))

    if arguments.output:
        output_dir = Path(arguments.output)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        consumers.append(get_fs_tree_writer(output_dir))

    for i in stream:
        for c in consumers:
            c(i)


def main():
    arguments = parse_args()

    if arguments.version:
        print(VERSION)
        exit(0)

    if os.path.isfile(USER_CONFIG_FILE):
        with open(USER_CONFIG_FILE, "r") as file:
            config = yaml.safe_load(file)
    else:
        config = yaml.safe_load(DEFAULT_CONFIG)

    if arguments.configure:
        configure(config)
        exit(0)

    command_dispatch(arguments, config)

if __name__ == "__main__":
    main()
