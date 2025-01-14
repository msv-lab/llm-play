# llm-play

llm-play is a tool that queries LLMs, analyzes responses, and executes experimental pipelines. First, it simplifies querying multiple LLMs with multiple prompts and generating multiple samples. Second, it provides data transformation capabilities, including answer extraction, code extraction, and customizable extractors to parse LLM outputs. Finally, it facilitates data analysis through semantic partitioning of responses, custom evaluators, and exporting results to CSV or JSON formats.

## Installation & Setup

Install the tool from PyPI:

    pip install llm-play

Configure API providers and models interactively (with settings editable in `~/.llm_play.yaml`):

    llm-play --add-provider
    llm-play --add-model

## Basic Usage

An LLM can be queried via an argument, a specified prompt file, or stdin:

    llm-play "What is the capital of China?"
    llm-play --prompt prompt.md
    llm-play < prompt.md

In all these cases, the response is printed on stdout, and can be redirected to a file:

    llm-play "What is the capital of China?" > output.md

Default settings such as the model and its temperature can be configured interactively with `-c/--configure` (with settings editable in `~/.llm_play.yaml`):

    llm-play -c

Command-line options take precedence over the default settings. `--version` prints the version; `--help` print the help message.

## Batch Processing

When the number of models or prompts or responses exceeds one, the tool operates in batch mode. For example, to sample 10 responses from two models (`qwen2.5-7b-instruct` and `qwen2.5-coder-7b-instruct`) with a temperature of 0.5, use the command:

    llm-play --prompt prompts/question1.md \
             --model qwen2.5-72b-instruct qwen2.5-7b-instruct \
             -t 0.5 \
             -n 10

In batch mode, a short summary of responses will be printed on stdout:

    Model                │ Temp. │ Label     │ Hash       │   ID │ Class │ Content
    ─────────────────────┼───────┼───────────┼────────────┼──────┼───────┼────────
    qwen2.5-72b-instruct │   0.5 │ question1 │ 4ae91f5... │    0 │     0 │ "It ...
    qwen2.5-72b-instruct │   0.5 │ question1 │ 4ae91f5... │    1 │     1 │ "It ...
    qwen2.5-72b-instruct │   0.5 │ question1 │ 4ae91f5... │    2 │     2 │ "It ...
    ...

In this table, `question1` is the prompt label, `4ae91f5bd6090fb6` is its SHAKE128 length=8 hash. Prompts with repeating hashes are skipped. The `Class` column displays the IDs of equivalence classes of responses (see [Partitioning](#partitioning)).

To store results, the output needs to be specified with `--output`. For example, `--output samples` will save the results in the following filesystem tree:

    samples
    ├── qwen2.5-7b-instruct_0.5
    │   ├── question1_4ae91f5bd6090fb6.md
    │   └── question1_4ae91f5bd6090fb6
    │       ├── 0_0.md
    │       ...
    │       └── 9_9.md
    └── qwen2.5-coder-7b-instruct_0.5
        ├── question1_4ae91f5bd6090fb6.md
        └── question1_4ae91f5bd6090fb6
            ├── 0_0.md
            ...
            └── 9_9.md

In this tree, `question1_4ae91f5bd6090fb6.md` contains the prompt; `0_0.md`, ..., `9_9.md` are the samples. In `5_3.md`, `5` is the sample identifier, and `3` is the identifier of its equivalence class. The sample file extension can be specified using the `--extension` options, e.g. `--extension py`.

The data can also be stored in CSV and JSON formats (see [Data Formats](#data-formats)).

Multiple prompt files can be specified as inputs, e.g. using all `*.md` files in the current directory:

    llm-play --prompt *.md --output samples

When the argument of `--prompt` is a directory, all `*.md` files are loaded from this directory non-recursively. If the query originates from a file, the prompt will adopt the file's name (excluding the extension) as its label. When a query is supplied through stdin or as a command-line argument, the label is empty.

Multiple outputs can be specified at the same time, e.g.

    --output samples samples.json

## Data Transformation

Data transformation can be used, for example, to extract relevant information from the generated samples or from data extracted in earlier stages. This is to extract text within the tag `<answer> ... </answer>` from all samples in `samples`, and save the results into the directory `extracted`:

    llm-play --map samples \
             --function __FIRST_TAGGED_ANSWER__ \
             --output extracted

The above function searches for text wrapped with `<answer>` and `</answer>` and prints only the content inside the tags.

Transformation is performed by either builtin functions or shell commands. The builtin function `__ID__` simply returns the entire string without modification. The builtin function `__FIRST_TAGGED_ANSWER__` returns the first occurence of a string wrapped into the tag `<answer></answer>`. The builtin function `__FIRST_MARKDOWN_CODE_BLOCK__` extract the content of the first Markdown code block.

Function defined through shell commands should use the [shell template language](#shell-template-language). For example, this is to count the number of characters in each response:

    --function 'wc -m < %%ESCAPED_DATA_FILE%%'

A transformation of a datum fails iff the function terminates with a non-zero exit code; in this case, the datum is ignored. Thus, shell commands can also be used for data filtering. For example, this is to filter out responses longer than 50 characters:

    --function '(( $(wc -m < %%ESCAPED_DATA_FILE%%) <= 50 )) && cat %%ESCAPED_DATA_FILE%%' \

Answers can also be extracted by LLMs. For example, this function checks if a prevously received response is affirmative:

    --function "llm-play '<answer>'%%CONDENSED_ESCAPED_DATA%%'</answer>. Is this answer affirmative? Respond Yes or No.' --model qwen2.5-72b-instruct --answer"

### On-the-fly Transformation

Data can be extracted on-the-fly while querying LLMs if `--function` is explicitly provided:

    llm-play "Name a city in China. Your answer should be formatted like **CITY NAME**" \
             --function "grep -o '\*\*[^*]*\*\*' %%ESCAPED_DATA_FILE%% | head -n 1 | sed 's/\*\*//g'"

There are convenience options to simplify extracting answers or code. The option `--answer` automatically augment the prompt and apply the necessary transformation to extract the relevant parts of the response:

    llm-play "${QUESTION}" --answer

is equivalent to

    llm-play "${QUESTION} Wrap the final answer with <answer></answer>."" --function __FIRST_TAGGED_ANSWER__

The option `--code` extracts a code block from Markdown formatting.

    llm-play "Write a Python function that computes the n-th Catalan number" --code

is equivalent to

    llm-play "Write a Python function that computes the n-th Catalan number" --function __FIRST_MARKDOWN_CODE_BLOCK__

In on-the-fly mode, the transformation options selected with `-c` are ignored.

## Partitioning

Responses can be grouped into equivalence classes based on a specified binary relation. The equivalence relation used for partitioning can be customized via the option `--relation`. An equivalence is defined via a builtin function or a shell command. The builtin relation `__ID__` checks if two answers are syntactically identical. The builtin relation `__TRIMMED_CASE_INSENSITIVE__` ignores trailing whitespaces and is case-insensitive. A relation defined via a shell command holds iff the command exits with the zero status code. For example, this is to group answers into equivalence classes based on a judgement from the `qwen2.5-72b-instruct` model:

    --relation "llm-play 'Are these two answers equivalent: <answer1>'%%CONDENSED_ESCAPED_DATA1%%'</answer1> and <answer2>'%%CONDENSED_ESCAPED_DATA2%%'</answer2>?' --model qwen2.5-72b-instruct --predicate"

Paritioning can be performed either locally - for responses associated with the same (model, prompt) pair - using the option  `--partition-locally`, or globally - across all responses - using the option `--partition-globally`. For example, this is to partition using a custom relation defined in a Python script:

    llm-play --partition-globally data \
             --relation `python custom_equivalence.py %%ESCAPED_DATA_FILE1%% %%ESCAPED_DATA_FILE2%%` \
             --output classes

When partitioning is performed, the existing equivalence classes are ignored.

Additionally, the option `-c` can be used to select a predefined relation when using the options `--partition-*`.

A global partitioning w.r.t. the relation `__ID__` is performed on-the-fly during LLM sampling.

## Predicates

Predicates are special on-the-fly boolean evaluators. For example, this command acts as a predicate over `$CITY`:

    llm-play "Is $CITY the capital of China?" --predicate

It first extracts the answer to this question with

    llm-play "Is $CITY the capital of China? Respond Yes or No." --answer

If the answer is equivalent to `Yes` w.r.t. `__TRIMMED_CASE_INSENSITIVE__`, then it exits with the zero status code. If the answer is equivalent to `No`, it exits with the code `1`. If the answer is neither `Yes` or `No`, it exits with the code `2`.

The output of a command with `--predicate` cannot be exported with `--output`. Predicates can only be applied to commands with a single model/prompt/response.

## Data Formats

Data can be written using the `--output` and `--update` options, or read using the `--map` and `--partition-*` options in the following three formats: `FS_TREE` (filesystem tree), `JSON` and `CSV`. The format is determined by the argument of the above options, which is treated as a directory path unless it ends with `.json` or `.csv`. Here is a comparison table between these formats.

|   | `FS_TREE` | `JSON` | `CSV` |
| - | --------- | ------ | ----- |
| Intended use | Manual inspection | Storage and sharing | Data analysis |
| Store prompts? | Yes | Yes | Truncated |
| Store responses? | Yes | Yes | Truncated |
| Store metadata? | File extension | File extension | No |

`FS_TREE` enables running commands for a subset of data, e.g.

    llm-play --partition-locally data/qwen2.5-7b-instruct_1.0/a_4ae91f5bd6090fb6 \
             --relation __TRIMMED_CASE_INSENSITIVE__ \
             --output classes

When data exported into CSV is truncated, the corresponding column name is changed from `Sample Content` to `Sample Content [Truncated]`. A CSV with `Sample Content [Truncated]` cannot be used as an input to `--map` and `--partition-*`.

To convert between different formats, a transformation with an identity function can used:

    llm-play --map data --function __ID__ --relation __ID__ --output data.json

## Shell Template Language

The shell template language allows dynamic substitution of specific placeholders with runtime values before executing a shell command. These placeholders are instantiated and replaced with their corresponding values before the command is executed by the system shell.

Available placeholders for data:

- `%%CONDENSED_ESCAPED_DATA%%` - the single-lined, stripped, truncated and shell-escaped text.
- `%%ESCAPED_DATA%%` - the shell-escaped text.
- `%%CONDENSED_DATA%%` - the single-lined, stripped, truncated text.
- `%%RAW_DATA%%` - the original text.

Similarly, `RAW_`, `ESCAPED_`, `CONDENCED_` and `CONDENSED_ESCAPED_` variants are provided for the following variables:

- `%%PROMPT%%` - the prompt content.

The `ESCAPED_` variants are provided for the following variables:

- `%%DATA_FILE%%` - a path to a temporary file containing the data.
- `%%DATA_ID%%` - a unique ID associated with the datum, i.e. `<model>_<temperature>_<prompt hash>_<sample id>_<class_id>`.
- `%%PROMPT_FILE%%` - a path to a temporary file containing the prompt.
- `%%PROMPT_LABEL%%` - the prompt label.

For equivalence relation commands, which require multiple arguments, the data and prompt placeholders are indexed, e.g. `%%RAW_DATA1%%` and `%%PROMPT_LABEL2%%`.

## Planned Improvements

[WIP] To update an existing store, use the option `--update` instead of `--output`:

    llm-play --prompt *.md --update samples

In case of collisions, i.e. when samples for the same (model, temperature, prompt) tuple already exist in the store, the prompt labels with matching hashes will be updated, and the old responses will be removed.

[WIP] To continue an interrupted experiment, use `--continue` instead of `--output` or `--update`.

    llm-play --prompt *.md --continue samples

It will skip all tasks for which there is already an output file in the store.

[WIP] The option `--debug` prints detailed logs on stderr.

