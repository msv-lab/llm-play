from enum import Enum

import os
import sys
import shutil
import configparser
import argparse

import yaml
from tqdm import tqdm
import inquirer
from openai import OpenAI

DEFAULT_CONFIG = r"""
default:
  model: qwen2.5-72b-instruct
  temperature: 1.0
  extractor: ID
  equivalence: TRIMMED_CASE_INSENSITIVE

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
  - |
    sed -n '0,/<\/answer>/s/.*<answer>\(.*\)<\/answer>.*/\1/p' %%SINGLEQUOTED_FILE%%
  - |
    awk '/^```/{if (!found++) { while(getline && $0 !~ /^```/) print; exit}}' %%SINGLEQUOTED_FILE%%

equivalences:
  - ID
  - TRIMMED_CASE_INSENSITIVE
  - |
    llm-query -m qwen2.5-72b-instruct 'Are these two answers equivalent: %%DOUBLEQUOTED_ANSWER1%% and %%DOUBLEQUOTED_ANSWER2%%?' -p
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


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Query Interface")
    parser.add_argument("query", nargs='?', type=str, help="Query string")
    parser.add_argument("-i", "--input", nargs='+', type=str, help="Input files")
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
    parser.add_argument("-s", "--set", action="store_true", help="Set default options")
    return parser.parse_args()


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


def main():
    user_config_file = os.path.expanduser("~") + "/.llm_query.yaml"

    if not os.path.isfile(user_config_file):
        config = yaml.safe_load(DEFAULT_CONFIG)
    else:
        with open(user_config_file, 'r') as file:
            config = yaml.safe_load(file)

    arguments = parse_args()

    if (((arguments.num_responses and arguments.num_responses > 1) or
         (arguments.models and len(arguments.models) > 1) or
         (arguments.input and len(arguments.input) > 1)) and
        not arguments.output):
        print("the output directory needs to be specified with -o/--output", file=sys.stderr)
        exit(1)

    if arguments.query and arguments.input:
        print("specify either the query string, or the input files with -i/-input, not both", file=sys.stderr)
        exit(1)

    if arguments.set:
        questions = [
            inquirer.List('model',
                          message="Set model",
                          choices=[entry['name'] for entry in config['models']],
                          default=config['default']['model']
                          ),
            inquirer.Text('temperature',
                          message="Set model temperature",
                          default=config['default']['temperature']
                          ),
            inquirer.List('extractor',
                          message="Set answer extractor",
                          choices=config['extractors'],
                          default=config['default']['extractor']
                          ),
            inquirer.List('equivalence',
                          message="Set equivalence relation",
                          choices=config['equivalences'],
                          default=config['default']['equivalence']
                          ),
        ]
        config['selected'] = inquirer.prompt(questions)
        with open(user_config_file, 'w') as f:
            yaml.dump(config, f)
    else:
        settings = dict()

        if arguments.query:
            inputs = [(UNNAMED_TASK, arguments.query)]
        elif (not arguments.input and
              not arguments.query):
            inputs = [(UNNAMED_TASK, sys.stdin.read())]
        else:
            inputs = [('label', 'content')]

        if arguments.models:
            settings['models'] = arguments.models
        else:
            settings['models'] = [config['default']['model']]

        if arguments.temperature:
            settings['temperature'] = str(arguments.temperature)
        else:
            settings['temperature'] = [config['default']['temperature']]

        if arguments.num_responses:
            settings['num_responses'] = arguments.num_responses
        else:
            settings['num_responses'] = 1

        if (settings['num_responses'] <= 1 and
            len(settings['models']) <= 1 and
            inputs[0][0] == UNNAMED_TASK and
            not arguments.output):
            stream_response(inputs[0][1], settings['models'][0], settings['temperature'], config)
        else:
            execute_batch_jobs(inputs, settings, arguments.output, config)


if __name__ == "__main__":
    main()
