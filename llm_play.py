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
from typing import Dict, Tuple, List, Any, Union
from enum import Enum, auto

import shlex
import os
import sys
import shutil
import configparser
import argparse
import tempfile
import subprocess
import re
from collections import deque, defaultdict
import hashlib
from pathlib import Path
import json
import csv

import yaml
import InquirerPy
from openai import OpenAI
from wcwidth import wcwidth, wcswidth
import mistletoe
from anthropic import Anthropic


VERSION = "0.1.0"

DEFAULT_CONFIG = rf"""
default:
  models: []
  temperature: 1.0
  max_tokens: 1024
  function: __ID__
  relation: __ID__
providers: []
models: []
functions:
  - __ID__
  - |-
    __FIRST_TAGGED_ANSWER__
  - |-
    __FIRST_MARKDOWN_CODE_BLOCK__
relations:
  - __ID__
  - |-
    __TRIMMED_CASE_INSENSITIVE__
answer_directive: |-
  Wrap the final answer with <answer></answer>.
predicate_directive: |-
  Respond Yes or No.
csv_truncate_threshold: 50
shell_truncate_threshold: 100
"""

USER_CONFIG_FILE = Path.home() / ".llm_play.yaml"


@dataclass
class _FunctionFailure:
    pass


FUNCTION_FAILURE = _FunctionFailure()


@dataclass
class FunctionSuccess:
    value: str


FunctionResult = Union[_FunctionFailure, FunctionSuccess]


def extract_first_markdown_code_block(content):
    parsed = mistletoe.Document(content)
    for child in parsed.children:
        if child.__class__.__name__ == "CodeFence":
            return FunctionSuccess(child.children[0].content)
    return FUNCTION_FAILURE


BUILTIN_FUNCTIONS = {
    "__ID__": (lambda x: FunctionSuccess(x)),
    "__FIRST_TAGGED_ANSWER__": (
        lambda s: (
            FunctionSuccess(s.split("<answer>", 1)[1].split("</answer>", 1)[0])
            if "<answer>" in s
            and "</answer>" in s
            and s.index("<answer>") < s.index("</answer>")
            else FUNCTION_FAILURE
        )
    ),
    "__FIRST_MARKDOWN_CODE_BLOCK__": (lambda s: extract_first_markdown_code_block(s)),
}

BUILTIN_RELATIONS = {
    "__ID__": (lambda x, y: x == y),
    "__TRIMMED_CASE_INSENSITIVE__": (
        lambda x, y: x.strip().lower() == y.strip().lower()
    ),
}


def panic(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    exit(1)


@dataclass
class Prompt:
    content: str
    label: str
    hash: str

    @staticmethod
    def unlabelled(content):
        return Prompt.labelled(content, "")

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
    metadata: Dict[str, Any]


class StoreType(Enum):
    FS_TREE = auto()
    JSON = auto()
    CSV = auto()


@dataclass
class Store:
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

    FS_TREE format:
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

    CSV columns:
    1. Model
    2. Temperature
    3. Prompt Label
    4. Prompt Hash
    5. Sample ID
    6. Sample Equivalence Class
    7. Sample Content/Sample Content [Truncated]
    """

    type: StoreType
    path: Path

    @staticmethod
    def from_path(path):
        if path.suffix == ".json":
            return Store(type=StoreType.JSON, path=path)
        if path.suffix == ".csv":
            return Store(type=StoreType.CSV, path=path)
        return Store(type=StoreType.FS_TREE, path=path)

    def load(self):
        if self.type == StoreType.FS_TREE:
            return self._load_fs_tree()
        if self.type == StoreType.JSON:
            with self.path.open("r") as file:
                return json.load(file)
        if self.type == StoreType.CSV:
            return self._load_csv()

    def _load_csv(self):
        prompts = []
        data = defaultdict(lambda: defaultdict(list))
        with open(self.path, mode="r") as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                prompt = {
                    "hash": row["Prompt Hash"],
                    "label": row["Prompt Label"],
                    "content": "",
                }
                if prompt not in prompts:
                    prompts.append(prompt)
                distr_id = f"{row['Model']}_{row['Temperature']}"
                sample = {
                    "id": row["Sample ID"],
                    "class_id": row["Sample Equivalence Class"],
                    "content": row["Sample Content"],
                    "metadata": {"extension": ".md"},
                }
                data[distr_id][row["Prompt Hash"]].append(sample)
        # Convert the defaultdict to a normal dict
        data = {key: dict(value) for key, value in data.items()}
        return {"prompts": prompts, "data": data}

    def _load_fs_tree(self):
        path_query = self.path

        def parse_distr_dir(distr_dir):
            """
            all entries such that the name of an md file contains _, and there is a corresponding dir
            """
            for prompt_file in distr_dir.iterdir():
                if prompt_file.is_file() and prompt_file.suffix == ".md":
                    sample_dir_name = prompt_file.stem
                    if "_" in sample_dir_name:
                        prompt_label, prompt_hash = tuple(
                            sample_dir_name.rsplit("_", 1)
                        )
                        sample_dir = distr_dir / sample_dir_name
                        if sample_dir.exists() and sample_dir.is_dir():
                            yield (prompt_label, prompt_hash, prompt_file, sample_dir)

        def collect_samples(sample_dir):
            """
            collect all files in sample_dir in the form <sample_id>_<class_id>.<etc>
            """
            for f in sample_dir.iterdir():
                if f.is_file():
                    if f.stem.count("_") == 1:
                        sample_id_str, class_id_str = tuple(f.stem.split("_"))
                        yield {
                            "id": int(sample_id_str),
                            "class_id": int(class_id_str),
                            "content": f.read_text(),
                            "metadata": {"extension": f.suffix},
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
            for prompt_label, prompt_hash, prompt_file, sample_dir in distr_dir_data:
                prompts.append(
                    {
                        "hash": prompt_hash,
                        "label": prompt_label,
                        "content": prompt_file.read_text(),
                    }
                )
                samples[prompt_hash] = list(collect_samples(sample_dir))
            return (prompts, samples)

        def pick_subdir(path):
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            if len(subdirs) == 0:
                return None
            else:
                return subdirs[0]

        distr_dir_data = list(parse_distr_dir(path_query))
        if len(distr_dir_data) > 0:  # path_query is a distr dir
            prompts, samples = load_prompts_and_samples(distr_dir_data)
            return {"prompts": prompts, "data": {path_query.name: samples}}
        else:  # path_query is either the root directory, or a sample directory
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
                            if not any(
                                p["hash"] == prompt["hash"] for p in all_prompts
                            ):
                                all_prompts.append(prompt)
                return {"prompts": all_prompts, "data": all_samples}
            else:
                # this is a sample directory
                prompt_label, prompt_hash = tuple(path_query.name.rsplit("_", 1))
                prompt_file = path_query.parent / f"{prompt_label}_{prompt_hash}.md"
                distr_dir_data = [(prompt_label, prompt_hash, prompt_file, path_query)]

                prompts, samples = load_prompts_and_samples(distr_dir_data)
                return {"prompts": prompts, "data": {path_query.parent.name: samples}}

    def get_writer(self, extension, config):
        if self.type == StoreType.FS_TREE:
            return self.FSTreeWriter(self.path, extension, config)
        if self.type == StoreType.JSON:
            return self.JSONWriter(self.path, extension, config)
        if self.type == StoreType.CSV:
            return self.CSVWriter(self.path, extension, config)

    class FSTreeWriter:
        def __init__(self, path, extension, config):
            self.extension = extension
            self.path = path
            path.mkdir(exist_ok=True, parents=True)

        def process(self, i):
            distr_dir = self.path / i.distribution.id()
            distr_dir.mkdir(exist_ok=True, parents=True)
            extension = self.extension if self.extension else i.metadata["extension"]
            prompt_file = distr_dir / f"{i.prompt.label}_{i.prompt.hash}.md"
            if not prompt_file.exists():
                prompt_file.write_text(i.prompt.content)
            responses_dir = distr_dir / f"{i.prompt.label}_{i.prompt.hash}"
            responses_dir.mkdir(exist_ok=True)
            (
                responses_dir / f"{i.sample.id}_{i.sample.class_id}{extension}"
            ).write_text(i.sample.content)

        def flush(self):
            pass

    class JSONWriter:
        def __init__(self, path, extension, config):
            self.extension = extension
            self.path = path
            self.result = {
                "prompts": [],
                "data": defaultdict(lambda: defaultdict(list)),
            }
            self.added_prompts = set()

        def process(self, i):
            if i.prompt.hash not in self.added_prompts:
                self.result["prompts"].append(
                    {
                        "hash": i.prompt.hash,
                        "label": i.prompt.label,
                        "content": i.prompt.content,
                    }
                )
                self.added_prompts.add(i.prompt.hash)
            dist_id = i.distribution.id()
            prompt_hash = i.prompt.hash
            extension = self.extension if self.extension else i.metadata["extension"]
            self.result["data"][dist_id][prompt_hash].append(
                {
                    "id": i.sample.id,
                    "class_id": i.sample.class_id,
                    "content": i.sample.content,
                    "metadata": {"extension": extension},
                }
            )

        def flush(self):
            # convert defaultdict to dict:
            self.result["data"] = {k: dict(v) for k, v in self.result["data"].items()}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self.path.touch()
            with open(self.path, "w") as f:
                json.dump(self.result, f, indent=4)

    class CSVWriter:
        def __init__(self, path, extension, config):
            self.truncate_threshold = config["csv_truncate_threshold"]
            self.extension = extension
            self.path = path
            self.rows = []
            self.truncated = False

        def get_headers(self):
            return (
                "Model",
                "Temperature",
                "Prompt Label",
                "Prompt Hash",
                "Sample ID",
                "Sample Equivalence Class",
                (
                    "Sample Content"
                    if not self.truncated
                    else "Sample Content [Truncated]"
                ),
            )

        def _truncate(self, input_string):
            lines = input_string.splitlines()
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            joined_string = " ".join(non_empty_lines)
            processed_string = joined_string.strip()
            truncated_string = processed_string[:self.truncate_threshold]
            is_modified = truncated_string != input_string
            return truncated_string, is_modified

        def process(self, i):
            content, truncated = self._truncate(i.sample.content)
            self.truncated = self.truncated or truncated
            self.rows.append(
                (
                    i.distribution.model,
                    i.distribution.temperature,
                    i.prompt.label,
                    i.prompt.hash,
                    i.sample.id,
                    i.sample.class_id,
                    content,
                )
            )

        def flush(self):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self.path.touch()
            with self.path.open("w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self.get_headers())
                for row in self.rows:
                    writer.writerow(row)


class PartitioningMode(Enum):
    GLOBAL = auto()
    LOCAL = auto()


def stream_item_table_format(max_model_name_len, max_prompt_label_len):
    return [
        ("Model", max(max_model_name_len, len("Model")), "l"),
        ("Temp.", len("Temp."), "r"),
        ("Label", min(max(max_prompt_label_len, len("Label")), 20), "l"),
        ("Hash", max(len("Hash"), 10), "l"),
        ("ID", max(len("ID"), 4), "r"),
        ("Class", len("Class"), "r"),
        ("Content", None, "l"),
    ]


def get_provider_by_model(model, config):
    for m in config["models"]:
        if model == m["name"]:
            for p in config["providers"]:
                if m["provider"] == p["id"]:
                    return p
    panic(f"no provider for model {model}")


class LLMSampleStream:
    def __init__(self, query, max_tokens, config):
        self.config = config
        self.max_tokens = max_tokens
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
        self._table_format = stream_item_table_format(
            max_model_name_len, max_prompt_label_len
        )

    def __iter__(self):
        return self

    def _sample(self, distribution, prompt, n):
        provider = get_provider_by_model(distribution.model, self.config)
        if provider["support_multiple_samples"]:
            num_responses = n
        else:
            num_responses = 1
        if provider["API"] == "OpenAI":
            client = OpenAI(
                api_key=provider["api_key"],
                base_url=provider["base_url"],
            )
            completion = client.chat.completions.create(
                model=distribution.model,
                max_tokens=self.max_tokens,
                temperature=float(distribution.temperature),
                messages=[{"role": "user", "content": prompt.content}],
                n=num_responses,
            )
            return [c.message.content for c in completion.choices]
        else:
            assert provider["API"] == "Anthropic"
            client = Anthropic(
                api_key=provider["api_key"],
                base_url=provider["base_url"],
            )
            message = client.messages.create(
                model=distribution.model,
                max_tokens=self.max_tokens,
                temperature=float(distribution.temperature),
                messages=[
                    {"role": "user", "content": prompt.content}
                ]
            )
            return [c.text for c in message.content]

    def __next__(self):
        if self.current_item_index >= len(self):
            raise StopIteration
        self.current_item_index += 1
        if len(self.cache) == 0:
            d, p, n = self.execution_plan.pop()
            samples = self._sample(d, p, n)
            for content in samples:
                s = Sample(
                    id=self.current_sample_index,
                    class_id=self.current_sample_index,
                    content=content,
                )
                self.cache.append((d, p, s))
                self.current_sample_index += 1
            if n > len(samples):
                self.execution_plan.append((d, p, n - len(samples)))
            else:
                self.current_sample_index = 0
        d, p, s = self.cache.popleft()
        return StreamItem(
            distribution=d, prompt=p, sample=s, metadata={"extension": ".md"}
        )

    def next_batch(self):
        return self.current_sample_index == 0

    def table_format(self):
        return self._table_format

    def __len__(self):
        return self.size


def stream_response_to_stdout(prompt, model, temperature, max_tokens, config):
    provider = get_provider_by_model(model, config)
    if provider["API"] == "OpenAI":
        client = OpenAI(
            api_key=provider["api_key"],
            base_url=provider["base_url"],
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
    else:
        assert provider["API"] == "Anthropic"
        client = Anthropic(
            api_key=provider["api_key"],
            base_url=provider["base_url"],
        )
        with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=float(temperature),
                messages=[
                    {"role": "user", "content": prompt}
                ]
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
    if os.isatty(sys.stdout.fileno()):
        print()


class JSONDataStream:
    def __init__(self, json_data):
        self.current_item_index = 0
        self._next_batch = True
        max_model_name_len = 0
        max_prompt_label_len = 0
        self.cache = deque()
        for distr_id, prompt_to_samples in json_data["data"].items():
            for prompt_hash, samples in prompt_to_samples.items():
                for sample in samples:
                    d = Distribution.from_id(distr_id)
                    if len(d.model) > max_model_name_len:
                        max_model_name_len = len(d.model)
                    p = Prompt(
                        **next(
                            p for p in json_data["prompts"] if p["hash"] == prompt_hash
                        )
                    )
                    if len(p.label) > max_prompt_label_len:
                        max_prompt_label_len = len(p.label)
                    s = Sample(
                        id=sample["id"],
                        class_id=sample["class_id"],
                        content=sample["content"],
                    )
                    m = sample["metadata"]
                    self.cache.append((d, p, s, m))
        self.size = len(self.cache)
        self._table_format = stream_item_table_format(
            max_model_name_len, max_prompt_label_len
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_item_index >= len(self):
            raise StopIteration
        self.current_item_index += 1
        self._next_batch = False
        d, p, s, m = self.cache.popleft()
        if len(self.cache) > 0 and (self.cache[0][0] != d or self.cache[0][1] != p):
            self._next_batch = True
        return StreamItem(distribution=d, prompt=p, sample=s, metadata=m)

    def next_batch(self):
        return self._next_batch

    def table_format(self):
        return self._table_format

    def __len__(self):
        return self.size


class Map:
    def __init__(self, stream, function, config):
        self.shell_truncate_threshold = config["shell_truncate_threshold"]
        self.stream = stream
        self.function = function

    def __iter__(self):
        return self

    def _call_shell_function(self, function, distribution, prompt, sample):
        with tempfile.NamedTemporaryFile() as prompt_file, \
             tempfile.NamedTemporaryFile() as sample_file:
            prompt_file.write(prompt.content.encode())
            prompt_file.flush()
            sample_file.write(sample.content.encode())
            sample_file.flush()
            cmd = instantiate_shell_template(
                function,
                distributions=[distribution],
                prompts=[prompt],
                prompt_files=[prompt_file.name],
                samples=[sample],
                sample_files=[sample_file.name],
                truncate_threshold=self.shell_truncate_threshold
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
                result = self._call_shell_function(self.function, i.distribution, i.prompt, i.sample)
            if result == FUNCTION_FAILURE:
                continue
            return StreamItem(
                distribution=i.distribution,
                prompt=i.prompt,
                sample=Sample(
                    id=i.sample.id, class_id=i.sample.class_id, content=result.value
                ),
                metadata=i.metadata,
            )

    def next_batch(self):
        return self.stream.next_batch()

    def table_format(self):
        return self.stream.table_format()

    def __len__(self):
        return len(self.stream)


class Partition:
    def __init__(self, stream, relation, mode, config):
        self.shell_truncate_threshold = config["shell_truncate_threshold"]
        self.stream = stream
        self.relation = relation
        self.mode = mode
        self._reset_classes()

    def _reset_classes(self):
        self._next_class = 0
        self._classes = dict()

    def _add_new_class(self, distribution, prompt, sample):
        new_class = self._next_class
        self._classes[new_class] = (distribution, prompt, sample)
        self._next_class += 1
        return new_class

    def __iter__(self):
        return self

    def _call_shell_relation(self, relation, i1, i2):
        distribution1, prompt1, sample1 = i1
        distribution2, prompt2, sample2 = i2
        with tempfile.NamedTemporaryFile() as prompt_file1, \
             tempfile.NamedTemporaryFile() as prompt_file2, \
             tempfile.NamedTemporaryFile() as data_file1, \
             tempfile.NamedTemporaryFile() as data_file2:
            prompt_file1.write(prompt1.content.encode())
            prompt_file1.flush()
            prompt_file2.write(prompt2.content.encode())
            prompt_file2.flush()
            data_file1.write(sample1.content.encode())
            data_file1.flush()
            data_file2.write(sample2.content.encode())
            data_file2.flush()
            cmd = instantiate_shell_template(
                relation,
                distributions=[distribution1, distribution2],
                prompts=[prompt1, prompt2],
                prompt_files=[prompt_file1.name, prompt_file2.name],
                samples=[sample1, sample2],
                sample_files=[data_file1.name, data_file2.name],
                truncate_threshold=self.shell_truncate_threshold
            )
            result = subprocess.run(
                cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return result.returncode == 0

    def __next__(self):
        if self.mode == PartitioningMode.LOCAL and self.stream.next_batch():
            self._reset_classes()
        i = next(self.stream)
        class_id = None
        for id, (distribution, prompt, sample) in self._classes.items():
            if i.sample.content == sample.content:
                class_id = id
                break
            if self.relation != "__ID__":
                if self.relation in BUILTIN_RELATIONS:
                    if BUILTIN_RELATIONS[self.relation](
                        i.sample.content, sample.content
                    ):
                        class_id = id
                        break
                else:
                    if self._call_shell_relation(
                        self.relation,
                        (i.distribution, i.prompt, i.sample),
                        (distribution, prompt, sample)
                    ):
                        class_id = id
                        break
        if class_id == None:
            class_id = self._add_new_class(i.distribution, i.prompt, i.sample)
        return StreamItem(
            distribution=i.distribution,
            prompt=i.prompt,
            sample=Sample(id=i.sample.id, class_id=class_id, content=i.sample.content),
            metadata=i.metadata,
        )

    def next_batch(self):
        return self.stream.next_batch()

    def table_format(self):
        return self.stream.table_format()

    def __len__(self):
        return len(self.stream)


def instantiate_shell_template(t, distributions, prompts, prompt_files, samples, sample_files, truncate_threshold):
    assert len(samples) == len(sample_files)
    assert len(prompts) == len(prompt_files)

    def render(value, escape=False, truncate=False):
        value = (
            truncate_content(value, truncate_threshold) if truncate else value
        )
        value = shlex.quote(value) if escape else value
        return value

    for s, i in [("", 0), ("1", 0), ("2", 1)]:
        data_id = f"{distributions[i - 1].model}_{distributions[i - 1].temperature}_{prompts[i - 1].hash}_{samples[i - 1].id}_{samples[i - 1].class_id}"
        t = t.replace(f"%%RAW_DATA{s}%%", render(samples[i - 1].content))
        t = t.replace(f"%%ESCAPED_DATA{s}%%", render(samples[i - 1].content, escape=True))
        t = t.replace(f"%%CONDENSED_DATA{s}%%", render(samples[i - 1].content, truncate=True))
        t = t.replace(
            f"%%CONDENSED_ESCAPED_DATA{s}%%",
            render(samples[i - 1].content, truncate=True, escape=True),
        )
        t = t.replace(f"%%DATA_FILE{s}%%", render(sample_files[i - 1]))
        t = t.replace(
            f"%%ESCAPED_DATA_FILE{s}%%", render(sample_files[i - 1], escape=True)
        )
        t = t.replace(f"%%DATA_ID{s}%%", data_id)
        t = t.replace(f"%%ESCAPED_DATA_ID{s}%%", shlex.quote(data_id))
        t = t.replace(f"%%PROMPT{s}%%", render(prompts[i - 1].content, 0))
        t = t.replace(
            f"%%ESCAPED_PROMPT{s}%%", render(prompts[i - 1].content, escape=True)
        )
        t = t.replace(
            f"%%CONDENSED_PROMPT{s}%%", render(prompts[i - 1].content, truncate=True)
        )
        t = t.replace(
            f"%%CONDENSED_ESCAPED_PROMPT{s}%%",
            render(prompts[i - 1].content, escape=True, truncate=True),
        )
        t = t.replace(f"%%PROMPT_FILE{s}%%", render(prompt_files[i - 1]))
        t = t.replace(
            f"%%ESCAPED_PROMPT_FILE{s}%%", render(prompt_files[i - 1], escape=True)
        )
        t = t.replace(f"%%PROMPT_LABEL{s}%%", render(prompts[i - 1].label))
        t = t.replace(
            f"%%ESCAPED_PROMPT_LABEL{s}%%", render(prompts[i - 1].label, escape=True)
        )

    return t


class StreamItemPrinter:
    def __init__(self, stream):
        self.printer = TablePrinter(stream.table_format())

    def process(self, i):
        row = (
            i.distribution.model,
            i.distribution.temperature,
            i.prompt.label,
            i.prompt.hash,
            i.sample.id,
            i.sample.class_id,
            i.sample.content,
        )
        self.printer.print_row(row)

    def flush(self):
        pass


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
        return "".join(result) + "..."
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
        if flexible_columns > 0:
            available_width = max(terminal_width - fixed_total, 0)
            flexible_width = available_width // flexible_columns
            column_specs = [
                c if c[1] is not None else (c[0], flexible_width, c[2])
                for c in column_specs
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
            return "".join(result) + "..."
        return content

    def _wc_rjust(self, text, length, padding=" "):
        return padding * max(0, (length - wcswidth(text))) + text

    def _wc_ljust(self, text, length, padding=" "):
        return text + padding * max(0, (length - wcswidth(text)))

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
    parser = argparse.ArgumentParser(description="llm-play CLI")
    parser.add_argument("query", nargs="?", type=str, help="Query string")
    parser.add_argument("--prompt", nargs="+", type=str, help="Prompt files")
    parser.add_argument("--output", nargs="+", type=str, help="Output FS-tree/JSON/CSV")
    parser.add_argument("--model", nargs="+", type=str, help="List of models to query")
    parser.add_argument(
        "-t", "--temperature", type=float, help="Amount of randomness injected into the response"
    )
    parser.add_argument(
        "-n", "--num-samples", type=int, help="Number of samples to generate"
    )
    parser.add_argument(
        "--max-tokens", type=float, help="The maximum number of tokens to generate"
    )
    parser.add_argument("--map", type=str, help="Transform given data", metavar='INPUT')
    parser.add_argument(
        "--function", type=str, help="Builtin function or shell command"
    )
    parser.add_argument(
        "--extension", type=str, help="File extension for transformed data"
    )
    parser.add_argument("--answer", action="store_true", help="Extract answer")
    parser.add_argument("--code", action="store_true", help="Extract code")
    parser.add_argument(
        "--partition-locally",
        type=str,
        help="Locally partition data into equivalence classes",
        metavar='INPUT'
    )
    parser.add_argument(
        "--partition-globally",
        type=str,
        help="Globally partition data into equivalence classes",
        metavar='INPUT'
    )
    parser.add_argument("--relation", type=str, help="Builtin relation shell command")
    parser.add_argument(
        "--predicate",
        action="store_true",
        help="Evaluate truthfulness of the predicate",
    )
    # parser.add_argument("--debug", action="store_true", help="Print logs on stderr")
    parser.add_argument("--version", action="store_true", help="Print version")
    parser.add_argument(
        "-c", "--configure", action="store_true", help="Set default options"
    )
    parser.add_argument("--add-provider", action="store_true", help="Interactively add provider")
    parser.add_argument("--add-model", action="store_true", help="Interactively add model")
    return parser.parse_args()


def process_prompt_files(path_list):
    prompts = []

    def add_file(f):
        label = f.stem
        content = f.read_text()
        prompts.append(Prompt.labelled(content, label))

    for p in path_list:
        path = Path(p)
        if path.is_file():
            add_file(path)
        elif path.is_dir():
            for f in path.iterdir():
                if f.is_file() and f.suffix == ".md":
                    add_file(f)

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
                "type": "number",
                "name": "temperature",
                "float_allowed": True,
                "filter": (lambda result: float(result)),
                "message": "Sampling temperature:",
                "default": config["default"]["temperature"],
            },
            {
                "type": "number",
                "name": "max_tokens",
                "filter": (lambda result: int(result)),
                "message": "Maximum tokens to generate:",
                "default": config["default"]["max_tokens"],
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
                "message": "Relation for --partition-*:",
                "choices": config["relations"],
                "default": config["default"]["relation"],
            },
        ]
    )
    if selected:
        config["default"] = selected
        with open(USER_CONFIG_FILE, "w") as f:
            yaml.dump(config, f, width=float("inf"))
        print(f"changes written to {USER_CONFIG_FILE}")


def add_provider(config):
    selected = InquirerPy.prompt(
        [
            {
                "type": "list",
                "name": "API",
                "message": "Select API type:",
                "choices": ["OpenAI", "Anthropic"],
            },
            {
                "type": "input",
                "name": "id",
                "message": "Choose ID for this provider:",
                "validate": lambda result: result not in (p["id"] for p in config["providers"]),
                "invalid_message": "A provider with this ID already exists",
            },
            {
                "type": "input",
                "name": "base_url",
                "message": "Base URL:",
                "default": (lambda selected: (
                    "https://api.anthropic.com"
                    if selected["API"] == "Anthropic"
                    else "https://api.openai.com/v1"
                ))
            },
            {
                "type": "password",
                "name": "api_key",
                "transformer": lambda _: "[hidden]",
                "message": "Enter your API key:"
            },
            {
                "type": "confirm",
                "name": "support_multiple_samples",
                "message": "Does this provider support arbitrary n > 1?"
            },
        ]
    )
    if selected:
        config["providers"].append(selected)
        with open(USER_CONFIG_FILE, "w") as f:
            yaml.dump(config, f, width=float("inf"))
        print(f"changes written to {USER_CONFIG_FILE}")


def add_model(config):
    selected = InquirerPy.prompt(
        [
            {
                "type": "list",
                "name": "provider",
                "message": "Choose provider:",
                "choices": [p["id"] for p in config["providers"]],
            },
            {
                "type": "input",
                "name": "name",
                "message": "Enter model name:",
                "validate": lambda result: result not in (m["name"] for m in config["models"]),
                "invalid_message": "A model with this name already exists",
            }
        ]
    )
    if selected:
        config["models"].append(selected)
        if len(config["default"]["models"]) == 0:
            config["default"]["models"].append(selected["name"])
        with open(USER_CONFIG_FILE, "w") as f:
            yaml.dump(config, f, width=float("inf"))
        print(f"changes written to {USER_CONFIG_FILE}")


def canonical_float_format(number):
    if number.is_integer():
        return f"{number:.1f}"
    else:
        return str(number)


class SimplePrinter:
    def process(self, i):
        print(i.sample.content, end="")
        if os.isatty(sys.stdout.fileno()) and not i.sample.content.endswith("\n"):
            print()

    def flush(self):
        pass


def delete_path(path: Path):
    if path.exists():
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def command_dispatch(arguments, config):
    stream = []
    consumers = []

    conflicting_options = [
        bool(arguments.query),
        bool(arguments.prompt),
        bool(arguments.map),
        bool(arguments.partition_locally),
        bool(arguments.partition_globally),
    ]

    if sum(conflicting_options) > 1:
        panic("conflicting commands")

    if (arguments.answer or arguments.code) and (
        arguments.partition_globally or arguments.partition_locally
    ):
        panic("--answer/--code can only be used when sampling LLMs")

    extension = f".{arguments.extension}" if arguments.extension else None

    if arguments.map:
        store = Store.from_path(Path(arguments.map))
        stream = JSONDataStream(store.load())
        function = (
            arguments.function if arguments.function else config["default"]["function"]
        )
        stream = Map(stream, function, config)
        consumers.append(StreamItemPrinter(stream))
    elif arguments.partition_globally or arguments.partition_locally:
        if arguments.partition_globally:
            store = Store.from_path(Path(arguments.partition_globally))
            mode = PartitioningMode.GLOBAL
        else:
            store = Store.from_path(Path(arguments.partition_locally))
            mode = PartitioningMode.LOCAL
        stream = JSONDataStream(store.load())
        relation = (
            arguments.relation if arguments.relation else config["default"]["relation"]
        )
        stream = Partition(stream, relation, mode, config)
        consumers.append(StreamItemPrinter(stream))
    else:
        # sampling command

        if arguments.query:
            prompts = [Prompt.unlabelled(arguments.query)]
        elif not arguments.prompt and not arguments.query:
            prompts = [Prompt.unlabelled(sys.stdin.read())]
        else:
            prompts = process_prompt_files(arguments.prompt)

        if (arguments.code or arguments.answer) and arguments.function:
            panic("--function is mutually exclusive with --code/--answer")

        convenience_functions = [
            bool(arguments.code),
            bool(arguments.answer),
            bool(arguments.predicate),
        ]
        if sum(convenience_functions) > 1:
            panic("--code, --answer and --predicate are mutually exclusive")

        function = "__ID__"

        if arguments.answer:
            function = "__FIRST_TAGGED_ANSWER__"
            new_prompts = []
            for p in prompts:
                new_prompts.append(
                    Prompt.labelled(p.content + " " + config["answer_directive"], p.label)
                )
            prompts = new_prompts

        if arguments.code:
            function = "__FIRST_MARKDOWN_CODE_BLOCK__"

        if arguments.predicate:
            function = "__FIRST_TAGGED_ANSWER__"
            new_prompts = []
            pred_prompt = (
                prompts[0].content + " " + config["predicate_directive"] + " " + config["answer_directive"]
            )
            new_prompts.append(Prompt.labelled(pred_prompt, prompts[0].label))
            prompts = new_prompts

        if arguments.function:
            function = arguments.function

        temperature = canonical_float_format(
            float(arguments.temperature)
            if arguments.temperature
            else config["default"]["temperature"]
        )
        max_tokens = (
            int(arguments.max_tokens)
            if arguments.max_tokens
            else config["default"]["max_tokens"]
        )
        distributions = []
        for m in arguments.model if arguments.model else config["default"]["models"]:
            distributions.append(Distribution(model=m, temperature=temperature))

        if len(distributions) == 0:
            panic("select a model")

        query = LLMQuery(
            distributions=distributions,
            num_samples=arguments.num_samples if arguments.num_samples else 1,
            prompts=prompts,
        )

        if (
            arguments.predicate
            and len(query.prompts) * len(query.distributions) * query.num_samples > 1
        ):
            panic("--predicate can only be used with a single model/prompt/response")

        if len(query.prompts) * len(query.distributions) * query.num_samples == 1:
            if arguments.predicate:
                i = next(Map(LLMSampleStream(query, max_tokens, config), function, config))
                if BUILTIN_RELATIONS["__TRIMMED_CASE_INSENSITIVE__"](
                    i.sample.content, "Yes"
                ):
                    exit(0)
                if BUILTIN_RELATIONS["__TRIMMED_CASE_INSENSITIVE__"](
                    i.sample.content, "No"
                ):
                    exit(1)
                exit(2)
            if function == "__ID__":
                stream_response_to_stdout(
                    prompts[0].content,
                    query.distributions[0].model,
                    query.distributions[0].temperature,
                    max_tokens,
                    config,
                )
            else:
                stream = Map(LLMSampleStream(query, max_tokens, config), function, config)
                consumers.append(SimplePrinter())
        else:
            stream = LLMSampleStream(query, max_tokens, config)
            if function != "__ID__":
                stream = Map(stream, function, config)
            stream = Partition(stream, "__ID__", PartitioningMode.GLOBAL, config)
            consumers.append(StreamItemPrinter(stream))

    if arguments.output != None and len(arguments.output) > 0:
        for out in set(arguments.output):
            path = Path(out)
            delete_path(path)
            store = Store.from_path(path)
            consumers.append(store.get_writer(extension, config))

    for i in stream:
        for c in consumers:
            c.process(i)

    for c in consumers:
        c.flush()


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
        if len(config["providers"]) == 0:
            panic("add providers with --add-provider and models with --add-model")
        if len(config["models"]) == 0:
            panic("add models with --add-model")
        configure(config)
        exit(0)

    if arguments.add_model:
        if len(config["providers"]) == 0:
            panic("first, add providers with --add-provider")
        add_model(config)
        exit(0)

    if arguments.add_provider:
        add_provider(config)
        exit(0)

    command_dispatch(arguments, config)


if __name__ == "__main__":
    main()
