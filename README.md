# llm-play

A tool that queries LLMs and executes experimental pipelines.

```mermaid
flowchart LR
    A["`**Prompting LLMs:**
    - Multiple models
    - Multiple prompts
    - Multiple samples`"] --> B["`**Data Extraction:**
    - Answer extraction
    - Code extraction
    - Custom extractors`"]
    B --> C["`**Analysis:**
    - Comparing distributions
    - Uncertainty measures
    - Semantic clusters`"]
    B --> D["`**Evaluation:**
    - Custom evaluators
    - CSV/JSON export`"]
```

## Installation

Set some of the following API keys as environment variables, depending on the services you plan to use:

- `DASHSCOPE_API_KEY`
- `DEEPSEEK_API_KEY`
- `CLOSEAI_API_KEY`

Install the tool by running the command `python -m pip install .`

## Basic Usage

Run the following command to ask a question directly:

    llm-play "What is the capital of China?"

You can input a query stored in a file, such as `prompt.md`:

    llm-play < prompt.md
    
or

    llm-play --prompt prompt.md

Write the response to a file (e.g., `output.md`):

    llm-play "What is the capital of China?" > output.md

For convenience, default settings such as the model and its temperature can be configured globally using the option `-c/--configure`. These settings are saved in `~/.llm_play.yaml`:

    llm-play -c

Command-line options take precedence over the default settings.

## Batch Processing

To query two models (`qwen2.5-7b-instruct` and `qwen2.5-coder-7b-instruct`) with a temperature of 0.5, sample 10 responses, and save the results into the directory `samples`, use the command:

    llm-play "What is the capital of China?" \
             --model qwen2.5-7b-instruct qwen2.5-coder-7b-instruct \
             --temperature 0.5 \
             -n 10 \
             --output samples

The samples will be organized as follows (`__unnamed__` is the prompt id, `__unnamed__.md` contains the prompt, `0.md`, ..., `9.md` are the samples):

    samples
    ├── qwen2.5-7b-instruct_1.0
    │   ├── __unnamed__.md
    │   └── __unnamed__
    │       ├── 0.md
    │       ...
    │       └── 9.md
    └── qwen2.5-coder-7b-instruct_1.0
        ├── __unnamed__.md
        └── __unnamed__
            ├── 0.md
            ...
            └── 9.md

To query a model with prompts contained in all files matching `*.md` in the current directory, use the command:

    llm-play --prompt *.md --output samples

When a query is supplied through stdin or as a command-line argument, the prompt is automatically assigned the identifier `__unnamed__`. However, if the query originates from a file, the prompt will adopt the file's name (excluding the extension) as its identifier. In cases where multiple files are provided, ensure that their names are unique to avoid conflicts.

## Data Extraction

Extractors are shell commands used to extract relevant information from string data, such as generated samples or from data extracted in earlier stages. These commands are defined using a shell template language (described below). The special extractor `__ID__` simply returns the entire string without modification.

This is to extract text within the tag `<answer> ... </answer>` from all samples in `samples`, and save the results into the directory `extracted`:

    llm-play --extract samples \
             --output extracted \
             --extractor "sed -n '0,/<\/answer>/s/.*<answer>\(.*\)<\/answer>.*/\1/p' %%ESCAPED_DATA_FILE%%"

The above extractor searches for text wrapped within `<answer>` and `</answer>` tags and prints only the content inside the tags.

A datum is treated as not containing any relevant information iff these two condition hold: (1) the extractor terminates with a non-zero exit code, and (2) its stdout is empty. In this case, the datum is ignored.

By default, the extracted data is saved into "txt" files. The file extension can be specified using the `--extension` options, e.g. `--extension py` resulting in:

    extracted
    └── qwen2.5-7b-instruct_1.0
        ├── __unnamed__.md
        └── __unnamed__
            ├── 0.py
            ├── 1.py
            ...
            └── 9.py

### On-the-fly Extraction

Data can be extracted on-the-fly while querying LLMs if `--extractor` is explicitly provided:

    llm-play "What is the capital of China? Wrap the final answer with <answer> </answer>" \
             --extractor "sed -n '0,/<\/answer>/s/.*<answer>\(.*\)<\/answer>.*/\1/p' %%ESCAPED_DATA_FILE%%"

There are built-in helper functions to simplify extracting answers or code when performed on-the-fly. These helpers automatically augment the prompt and apply the necessary extractors to extract the relevant parts of the sample (the default extactor and equivalence options are ignored).

    llm-play "What is the capital of China?" --answer
    llm-play "Write a Python function f(n: int) -> int that computes the n-th Catalan number" --code
    
## Clustering

To group answers into equivalence classes based qwen2.5's judgement, use the following command:

    llm-play --cluster data \
             --output clusters \
             --equivalence "llm-play --model qwen2.5-72b-instruct 'Are these two answers equivalent: \"%%DATA1%%\" and \"%%DATA2%%\"?' --predicate"

Clustering can be performed for a subset of data:

    llm-play --cluster data/qwen2.5-7b-instruct_1.0/a/ \
             --output clsuters \
             --equivalence "$EQUIVALENCE"
    
The equivalence class identifiers will be added to end of output file names, after the underscore:

    clusters
    └── qwen2.5-7b-instruct_1.0
        ├── __unnamed__.md
        └── __unnamed__
            ├── 0_0.md
            ├── 1_0.md
            ...
            └── 9_3.md

This equivalence is defined via a shell command that exits with the zero status code when two answers are equivalent. The classes are computed using the [disjoint-set algorithm](https://en.wikipedia.org/wiki/Disjoint-set_data_structure).

Equivalence relations can be composed by repeated clustering:

    llm-play --cluster data --output clusters1 --equivalence "$EQUIVALENCE1"
    llm-play --cluster clusters1 --output clusters2 --equivalence "$EQUIVALENCE2"
    
The equivalence relation can be configured:

- Using the `-c` option to select a predefined equivalence command.
- Or, specifying a custom equivalence command using the `--equivalence` option.

Clustering can also be performed on-the-fly while querying models if any non-trivial equivalence relations is specified explicitly with `--equivalence`. The trivial relation `__ID__` means syntactic identity and effectively disables clustering.

## Data Analysis

To show the distribution of equivalence classes of outputs (across one or more models and/or prompts), use the following command:

    llm-play --distribution data

A distribution can be analyzed for a subset of outputs:

    llm-play --distribution data/a.md/qwen2.5-7b-instruct_1.0
    
This will compute and visualise

- [empirical probability](https://en.wikipedia.org/wiki/Empirical_probability) of clusters;
- semantic uncertainty (semantic entropy) computed over the equivalence classes

Related work on semantic uncertainty:

- Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation<br>
  Lorenz Kuhn, Yarin Gal, Sebastian Farquhar<br>
  ICLR 2023

Note that `--distribution` does not itself perform any data extraction or clustering.

### Comparing Distributions

To analyse difference between distributions of clusters, e.g. for different model temperatures, use the following command:

    llm-play --diff data/qwen2.5-7b-instruct_1.0/a data/qwen2.5-7b-instruct_0.5/a
    
This command aligns the cluster labels between these two distributions w.r.t. the specified equivalence relation, as well as computes some useful statistics:

- [Wasserstein metric](https://en.wikipedia.org/wiki/Wasserstein_metric)
- [Permutation test](https://en.wikipedia.org/wiki/Permutation_test) based on the Wasserstein metric
- [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index) over supports
- Differences between supports

## Evaluation

The evaluation mode is enabled with the `--evaluate` option and evaluates previously computed outputs against specified criteria.

To evaluate data by checking if each datum is equal to a specific value, i.e. `Beijing`, use:

    llm-play --evaluate data --equal Beijing

Evalation can be done for a subset of outputs:

    llm-play --evaluate data/qwen2.5-7b-instruct_1.0/a --equal Beijing
    
The evaluator `--equal VALUE` checks if the answer is equivalent to `VALUE` wrt the equivalence relations specified with `--equivalence` or the default one selected with `-c`.

You can specify a custom evaluator using the `--evaluator` option. A custom evaluator is a shell command that terminates with the zero exit code if the datum passes evaluation.

    llm-play --evaluate data --evaluator 'wc -w <<< %%ESCAPED_DATA%% | grep -q ^1$'

This example evaluates whether each datum contains exactly one word.

### On-the-fly Evaluation

When a single sample is generated from an single model, it can be evaluated on-the-fly.

    llm-play "What is the capital of China?" --answer --equal Beijing
    
This helper option acts as a predicate over `$CITY`:

    llm-play "Is $CITY the capital of China?" --predicate

It is equivalent to the following (plus, the command will terminate with the zero exit code iff it passes the evaluation):

    llm-play "Is $CITY the capital of China? Respond Yes or No." \
              --answer \
              --equal Yes \
              --equivalence __TRIMMED_CASE_INSENSITIVE__ \
              >/dev/null
    
## Data Export

Data can be exported to a format suitable for further analysis, such as JSON or CVS using the options `--export` and `--report`.

This will export data as an CSV table (the file extension determines the format):

    llm-play --export data --report data.csv
    
The option `--report` can be added to other formats for on-the-fly reporting, e.g.

    llm-play --distribution data --report distribution.json

## Shell Template Language

The shell template language allows dynamic substitution of specific placeholders with runtime values before executing a shell command. These placeholders are instantiated and replaced with their corresponding values before the command is executed by the system shell.

Available placeholders:

- `%%DATA%%` - replaced with the raw output (sample or extracted information).
- `%%DATA_FILE%%` - replaced with a path to a temporary file containing the output.
- `%%PROMPT%%` - replaced with the raw input prompt.
- `%%PROMPT_FILE%%` - replaced with a path to a temporary file containing the input prompt.
- `%%PROMPT_ID%%` - replaced with the prompt id.

For commands that require multiple outputs, indexed placeholders are provided, e.g. `%%DATA1%%`, `%%DATA2%%`.

Variants of shell-escaped placeholders are available for safety when handling special characters, e.g. `%%ESCAPED_DATA%%`.

## Troubleshooting

The `--debug` option prints detailed logs on stderr.
