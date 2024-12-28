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
selected:
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
  - sed -n '0,/<\/answer>/s/.*<answer>\(.*\)<\/answer>.*/\1/p' %%SINGLEQUOTED_FILE%%
  - awk '/^```/{if (!found++) { while(getline && $0 !~ /^```/) print; exit}}' %%SINGLEQUOTED_FILE%%

equivalences:
  - ID
  - TRIMMED_CASE_INSENSITIVE
  - llm-query -m qwen2.5-72b-instruct 'Are these two answers equivalent: <answer>%%ANSWER1%%</answer> and <answer>%%ANSWER2%%</answer>?' -p
"""


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
    parser.add_argument("-o", "--output", type=str, help="Output directory or file")
    parser.add_argument("-m", "--model", nargs='+', type=str, help="List of models to query")
    parser.add_argument("-t", "--temperature", type=float, help="Temperature for model generation")
    parser.add_argument("-n", "--num-responses", type=int, help="Number of responses to generate")
    parser.add_argument("-c", "--configure", action="store_true", help="Configure default model and temperature") 
    return parser.parse_args()


def execute_jobs(arguments, config):
    if arguments.query:
        inputs = arguments.query
    elif (not arguments.input and
        not arguments.query):
        inputs = sys.stdin.read()
    else:
        inputs = [('label', 'content')]

    if arguments.model:
        models = arguments.model
    else:
        models = [config['selected']['model']]

    if arguments.temperature:
        temperature = str(arguments.temperature)
    else:
        temperature = [config['selected']['temperature']]

    if arguments.num_responses:
        num_responses = arguments.num_responses
    else:
        num_responses = 1
    
    if (num_responses and
        len(models) <= 1 and
        isinstance(inputs, str) and
        not arguments.output):
        stream_response(inputs, models[0], temperature, config)
    else:
        execute_batch_jobs(inputs, models, num_responses, temperature, arguments.output)


def recreate_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def execute_batch_jobs(inputs, models, num_responses, temperature, output):
    recreate_directory(output)
    if isinstance(inputs, str):
        with tqdm(total=len(models) * num_responses) as pbar:
            for im, model in enumerate(models):
                model_path = output + "/" + model + "_" + temperature
                os.makedirs(model_path)
                for i in range(num_responses):
                    response = get_response(inputs, model, temperature, config)
                    with open(f"{model_path}/{i}.md", "w") as file:
                        file.write(response)
                    pbar.update()
    else:
        pass


def main():
    user_config_file = os.path.expanduser("~") + "/.llm_query.yaml"

    if not os.path.isfile(user_config_file):
        config = yaml.safe_load(DEFAULT_CONFIG)
    else:
        with open(user_config_file, 'r') as file:
            config = yaml.safe_load(file)

    arguments = parse_args()

    if (((arguments.num_responses and arguments.num_responses > 1) or
         (arguments.model and len(arguments.model) > 1) or
         (arguments.input and len(arguments.input) > 1)) and
        not arguments.output):
        print("the output directory needs to be specified with -o/--output", file=sys.stderr)
        exit(1)

    if arguments.query and arguments.input:
        print("specify either the query string, or the input files with -i/-input, not both", file=sys.stderr)
        exit(1)

    if arguments.configure:
        questions = [
            inquirer.List('model',
                          message="Set model",
                          choices=[entry['name'] for entry in config['models']],
                          default=config['selected']['model']
                          ),
            inquirer.Text('temperature',
                          message="Set model temperature",
                          default=config['selected']['temperature']
                          ),
            inquirer.List('extractor',
                          message="Set answer extractor",
                          choices=config['extractors'],
                          default=config['selected']['extractor']
                          ),
            inquirer.List('equivalence',
                          message="Set equivalence relation",
                          choices=config['equivalences'],
                          default=config['selected']['equivalence']
                          ),
        ]
        config['selected'] = inquirer.prompt(questions)
        with open(user_config_file, 'w') as f:
            yaml.dump(config, f)
    else:
        execute_jobs(arguments, config)


if __name__ == "__main__":
    main()
