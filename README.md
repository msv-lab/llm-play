# llm-query

An utility for experimenting with LLMs.

## Installation

Define some of `DASHSCOPE_API_KEY`, `DEEPSEEK_API_KEY`, `CLOSEAI_API_KEY`.

Install with `pip install .`

## Usage

Simple query:

    llm-query "What is the capital of China?"
    
Query from file `prompt.md`:

    llm-query < prompt.md

Write answer to `output.md`:

    llm-query "What is the capital of China?" > output.md
    
Configure default model and temperature (written to `~/.llm_query.ini`):

    llm-query -c
    
Query two models with the temparature 0.5, generate 10 responses, write into the directory `output`:

    llm-query "What is the capital of China?" -m qwen2.5-7b-instruct qwen2.5-coder-7b-instruct -t 0.5 -n 10 -o output
    
The resulting directory structure:

    output
    ├── qwen2.5-7b-instruct_1.0
    │   ├── 0.md
    ...
    │   └── 9.md
    └── qwen2.5-coder-7b-instruct_1.0
        ├── 0.md
        ...
        └── 9.md

Query a model on all prompts in the files `a.md` and `b.md` in the current repository, write responses into the directory `output`:

    llm-query -i *.md -o output
    
The resulting directory structure (assuming `qwen2.5-7b-instruct` and `1.0` as defaults):
    
    output
    ├── a
    │   └── qwen2.5-7b-instruct_1.0
    │       ├── 0.md
    ...
    │       └── 9.md
    └── b
        └── qwen2.5-7b-instruct_1.0
            ├── 0.md
            ...
            └── 9.md
    
The name of the output directory is required if any of the following is higher than 1:

- number of models
- number of responses
- number of input files
    

