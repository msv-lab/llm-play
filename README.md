# llm-query

An utility for small-scale experimentation with LLMs. It supports querying models, extracting answers from responses, executing evals, comparing outputs.

## Installation

Define some of `DASHSCOPE_API_KEY`, `DEEPSEEK_API_KEY`, `CLOSEAI_API_KEY`.

Install with `pip install .`

## Basic Usage

Simple query:

    llm-query "What is the capital of China?"
    
Query from file `prompt.md`:

    llm-query < prompt.md

Write answer to `output.md`:

    llm-query "What is the capital of China?" > output.md
    
Set default model, temperature, etc (written to `~/.llm_query.yaml`):

    llm-query -s
    
## Batch Processing
    
Query two models with the temparature 0.5, generate 10 responses, write into the directory `output`:

    llm-query "What is the capital of China?" -m qwen2.5-7b-instruct qwen2.5-coder-7b-instruct -t 0.5 -n 10 -o output
    
The resulting directory structure (`unnamed` is the name of the task):

    output
    └── unnamed
        ├─── qwen2.5-7b-instruct_1.0
        │   ├── 0.md
        ...
        │   └── 9.md
        │
        └── qwen2.5-coder-7b-instruct_1.0
            ├── 0.md
            ...
            └── 9.md

Query a model on all prompts in the files `a.md` and `b.md` in the current repository, write responses into the directory `output`:

    llm-query -i *.md -o output
    
The resulting directory structure (assuming `qwen2.5-7b-instruct` and `1.0` as defaults):
    
    output
    ├── a.md
    │   └── qwen2.5-7b-instruct_1.0
    │       ├── 0.md
    ...
    │       └── 9.md
    └── b.md
        └── qwen2.5-7b-instruct_1.0
            ├── 0.md
            ...
            └── 9.md
    
The name of the output directory is required if any of the following is higher than 1:

- number of models
- number of responses
- number of input files

## Extracting Answers

Extractors are shell commands that extract answers from responses. They are defined using the shell template langauge described below. The special extractor `ID` just returns the entire response.

Assume that the extractor `sed -n '0,/<\/answer>/s/.*<answer>\(.*\)<\/answer>.*/\1/p' %%SINGLEQUOTED_FILE%%` was selected using `llm-query -s`. The following command will print only the name of the city:

    llm-query "What is the capital of China? Wrap the final answer with <answer> </answer>"
    
A custom extractor can be specified as the argument of the option `--extractor`.

There are helper functions for answers and code that automatically augment the prompt and extract the relevant parts of the response:

    llm-query "What is the capital of China?" -a
    llm-query "Write a Python function f(n: int) -> int that computes the n-th Catalan number" -c
    
## Examining Answers

To view the distribution of responses (for one or more models/inputs):

    llm-query -d output/

An answer extraction is performed before computing the distribution.

The discribution is computed over equivalence classes of answers. The equivalence relation is defined via shell command that terminates with the zero exit code iff two answers are equivalent. It can either be set using `llm-query -s`, or specified using the option `--equivalence`.
    
## Evaluating Results

Evaluation is enabled using the options `-e` and `--equals`/`--evaluator`.

This command will output an evaluation table for previously computed responses, checking if they are equal to `Beijing`:

    llm-query -e output --equals Beijing
    
This command will print the evaluation table for only one response, and will also terminate with the zero exit code if the answer is correct:

    llm-query "What is the capital of China?" --equals Beijing
    
The special evaluator `--equals VALUE` checks if answer is equivalent to `VALUE` wrt the specified equivalence relation.

Custom evaluators are shell commands that terminate with the zero exit code iff the answer passes evaluation. A custom evaluator is specified using the option `--evaluator COMMAND` instead of `--equals`.

This helper function acts as a predicate over `$ANSWER`:

    llm-query "Is $ANSWER the capital of China?" -p
    
It is equivalent to the following:

    llm-query "Is $ANSWER the capital of China? Respond Yes or No." -a --equals Yes >/dev/null

## Shell Template Language

## Case Study: Executing MBPP

