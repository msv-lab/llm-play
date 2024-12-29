from enum import Enum

import shlex
import os
import sys
import shutil
import configparser
import argparse
import tempfile
import subprocess

import yaml
from tqdm import tqdm
import InquirerPy
from openai import OpenAI


ZERO_SHOT_COT_DIRECTIVE = "Let's think step by step."
ANSWER_FORMAT_DIRECTIVE = "Wrap the final answer with <answer> </answer>."
ANSWER_EXTRACTOR = r"sed -n '0,/<\/answer>/s/.*<answer>\(.*\)<\/answer>.*/\1/p' %%ESCAPED_FILE%%"
CODE_EXTRACTOR = r"awk '/^```/{if (!found++) { while(getline && $0 !~ /^```/) print; exit}}' %%ESCAPED_FILE%%"


DEFAULT_CONFIG = fr"""
default:
  model: qwen2.5-72b-instruct
  temperature: 1.0
  extractor: ID
  equivalence: TRIMMED_CASE_INSENSITIVE
  directives: []
providers:
  Aliyun:
    API: OpenAI
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    key_env_variable: DASHSCOPE_API_KEY
  DeepSeek:
    API: OpenAI
    base_url: https://api.deepseek.com
    key_env_variable: DEEPSEEK_API_KEY
    
  CloseAI_OpenAI:
    API: OpenAI
    base_url: https://api.openai-proxy.org/v1
    key_env_variable: CLOSEAI_API_KEY

  CloseAI_Anthropic:
    API: Anthropic
    base_url: https://api.openai-proxy.org/anthropic
    key_env_variable: CLOSEAI_API_KEY
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
    provider: Deepseek
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
  - ID
  - |-
    {ANSWER_EXTRACTOR}
  - |-
    {CODE_EXTRACTOR}
directives:
  - |-
    {ZERO_SHOT_COT_DIRECTIVE}
  - |-
    {ANSWER_FORMAT_DIRECTIVE}
equivalences:
  - ID
  - TRIMMED_CASE_INSENSITIVE
  - |-
    llm-query -m qwen2.5-72b-instruct 'Are these two answers equivalent: "%%ANSWER1%%" and "%%ANSWER2%%"?' -p
"""


UNNAMED_TASK = '__unnamed__'


def sanitize_path_to_filename(path: str) -> str:
    return path.replace("/", "__").replace("\\", "__")


def get_provider_by_model(model, config):
    for m in config['models']:
        if model == m['name']:
            return m['provider']
    raise ValueError(f"no provider for model {model}")


def get_response(prompt, model, temperature, config):
    provider = get_provider_by_model(model, config)
    client = OpenAI(
        api_key=os.getenv(config['providers'][provider]['key_env_variable']),
        base_url=config['providers'][provider]['base_url'],
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'user', 'content': prompt}],
    )
    return completion.choices[0].message.content


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
    for name, prompt in inputs:
        with tqdm(total=len(settings['models']) * settings['num_responses']) as pbar:
            for im, model in enumerate(settings['models']):
                model_path = output + "/" + model + "_" + settings['temperature']
                os.makedirs(model_path)
                for i in range(num_responses):
                    response = get_response(prompt, model, settings['temperature'], config)
                    with open(f"{model_path}/{i}.md", "w") as file:
                        file.write(response)
                    pbar.update()


def instantiate_shell_template(template, answers=None, files=None, task=None):
    if isinstance(answers, str) or answers is None:
        answers = [answers]
    if isinstance(files, str) or files is None:
        files = [files]

    def get_replacement(item_list, index, escape=False):
        value = item_list[index]
        return shlex.quote(value) if escape and value is not None else value

    if len(answers) * len(files) == 1:
        template = template.replace(f"%%ANSWER%%", get_replacement(answers, 0))
        template = template.replace(f"%%ESCAPED_ANSWER%%", get_replacement(answers, 0, escape=True))
        template = template.replace(f"%%FILE%%", get_replacement(files, 0))
        template = template.replace(f"%%ESCAPED_FILE%%", get_replacement(files, 0, escape=True))
    else:
        for i in range(max(len(answers), len(files)), 0, -1):
            template = template.replace(f"%%ANSWER{i}%%", get_replacement(answers, i - 1))
            template = template.replace(f"%%ESCAPED_ANSWER{i}%%", get_replacement(answers, i - 1, escape=True))
            template = template.replace(f"%%FILE{i}%%", get_replacement(files, i - 1))
            template = template.replace(f"%%ESCAPED_FILE{i}%%", get_replacement(files, i - 1, escape=True))
    template = template.replace(f"%%TASK%%", get_replacement([task], 0))
    template = template.replace(f"%%ESCAPED_TASK%%", get_replacement([task], 0, escape=True))

    return template


def extract(response, extractor, task):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(response.encode())
        temp_file.flush()
        cmd = instantiate_shell_template(extractor, answers=response, files=temp_file.name, task=task)
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
    parser.add_argument("--extractor", type=str, help="Answer extraction shell command")
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

        try:
            with open(file_path, 'r') as file:
                content = file.read()
                label_content_pairs.append((label, content))
        except Exception as e:
            print(f"error reading file '{file_path}': {e}", file=sys.stderr)
            sys.exit(1)

    return label_content_pairs


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
                    "type": "checkbox",
                    "name": "directives",
                    "message": "Append directives to the prompt:",
                    "choices": config['directives']
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

        settings['directives'] = config['default']['directives']

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
            if ANSWER_FORMAT_DIRECTIVE not in settings['directives']:
                settings['directives'].append(ANSWER_FORMAT_DIRECTIVE)

        if arguments.code:
            settings['extractor'] = CODE_EXTRACTOR

        if len(settings['directives']) > 0:
            new_inputs = []
            for label, prompt in inputs:
                new_prompt = prompt
                for d in settings['directives']:
                    new_prompt += " " + d
                new_inputs.append((label, new_prompt))
            inputs = new_inputs

        if (settings['num_responses'] <= 1 and
            len(settings['models']) <= 1 and
            inputs[0][0] == UNNAMED_TASK and
            not arguments.output):
            if settings['extractor'] == 'ID':
                stream_response(inputs[0][1], settings['models'][0], settings['temperature'], config)
            else:
                response = get_response(inputs[0][1], settings['models'][0], settings['temperature'], config)
                answer = extract(response, settings['extractor'], UNNAMED_TASK)
                print(answer, end="")
                if os.isatty(sys.stdout.fileno()):
                    print()
        else:
            execute_batch_jobs(inputs, settings, arguments.output, config)


if __name__ == "__main__":
    main()
