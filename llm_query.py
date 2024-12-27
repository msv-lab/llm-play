from enum import Enum

import os
import sys
import shutil
import configparser
import argparse
import fileinput


from tqdm import tqdm
import inquirer
from openai import OpenAI


class Provider(Enum):
    ALIYUN = 1
    DEEPSEEK = 2
    CLOSEAI_OPENAI = 3
    CLOSEAI_ANTHROPIC = 4


PROVIDER_MODELS = [
    (Provider.ALIYUN, 'qwen-max-2024-09-19'),
    (Provider.ALIYUN, 'qwq-32b-preview'),
    (Provider.ALIYUN, 'qwen2.5-72b-instruct'),
    (Provider.ALIYUN, 'qwen2.5-7b-instruct'),
    (Provider.ALIYUN, 'qwen2.5-coder-32b-instruct'),
    (Provider.ALIYUN, 'qwen2.5-coder-7b-instruct'),
   
    (Provider.DEEPSEEK, 'deepseek-chat'),

    (Provider.CLOSEAI_OPENAI, 'o1-2024-12-17'),
    (Provider.CLOSEAI_OPENAI, 'o1-mini-2024-09-12'),
    (Provider.CLOSEAI_OPENAI, 'gpt-4o-2024-11-20'),
    (Provider.CLOSEAI_OPENAI, 'gpt-4o-mini-2024-07-18'),
]


provider_base_url = {
    Provider.ALIYUN: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    Provider.DEEPSEEK: "https://api.deepseek.com",
    Provider.CLOSEAI_OPENAI: "https://api.openai-proxy.org/v1"
}


provider_key_env_var = {
    Provider.ALIYUN: "DASHSCOPE_API_KEY",
    Provider.DEEPSEEK: "DEEPSEEK_API_KEY",
    Provider.CLOSEAI_OPENAI: "CLOSEAI_API_KEY"
}


provider_name = {
    Provider.ALIYUN: "AliCloud",
    Provider.DEEPSEEK: "DeepSeek",
    Provider.CLOSEAI_OPENAI: "CloseAI"
}


def get_provider_by_model(model):
    for (provider, m) in PROVIDER_MODELS:
        if model == m:
            return provider
    raise ValueError(f"no provider for model {model}")


def get_response(prompt, model, temperature):
    provider = get_provider_by_model(model)
    client = OpenAI(
        api_key=os.getenv(provider_key_env_var[provider]),
        base_url=provider_base_url[provider],
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'user', 'content': prompt}],
    )
    return completion.choices[0].message.content


def stream_response(prompt, model, temperature):
    provider = get_provider_by_model(model)
    client = OpenAI(
        api_key=os.getenv(provider_key_env_var[provider]),
        base_url=provider_base_url[provider],
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
        models = [config['llm']['model']]

    if arguments.temperature:
        temperature = str(arguments.temperature)
    else:
        temperature = [config['llm']['temperature']]

    if arguments.num_responses:
        num_responses = arguments.num_responses
    else:
        num_responses = 1
    
    if (num_responses and
        len(models) <= 1 and
        isinstance(inputs, str) and
        not arguments.output):
        stream_response(inputs, models[0], temperature)
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
                    response = get_response(inputs, model, temperature)
                    with open(f"{model_path}/{i}.md", "w") as file:
                        file.write(response)
                    pbar.update()
    else:
        pass


def main():
    config = configparser.ConfigParser()
    
    user_config_file = os.path.expanduser("~") + "/.llm_query.ini"
    if os.path.isfile(user_config_file):
        config.read(user_config_file)
    else:
        config['llm'] = {
            'model': 'qwen2.5-72b-instruct',
            'temperature': '1.0'
        }

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
                          choices=[m for (p, m) in PROVIDER_MODELS],
                          default=config['llm']['model']
                          ),
            inquirer.Text('temperature',
                          message="Set model temperature",
                          default=config['llm']['temperature']
                          ),
        ]
        config['llm'] = inquirer.prompt(questions)
        with open(user_config_file, 'w') as f:
            config.write(f)
    else:
        execute_jobs(arguments, config)


if __name__ == "__main__":
    main()
