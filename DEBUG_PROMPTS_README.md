# Debug Prompt Logging for BFCL Evaluation

This document explains how to debug input formatting issues in the BFCL evaluation.

## Overview

The Gemma FC handler has been instrumented with detailed logging to help diagnose issues where the model generates gibberish or unexpected outputs.

## What's Been Added

### 1. Console Logging

The `gemma_fc.py` handler now prints detailed debug information to the console:

- **Before formatting**: Raw messages and functions received
- **After formatting**: The complete formatted prompt sent to the model
- **Model response**: The raw response text and its length

### 2. Debug Log File

All prompts and responses are saved to `gemma_fc_debug_prompts.jsonl` in the working directory.

Each entry contains:
- `prompt_id`: Sequential ID for matching prompts with responses
- `test_id`: The BFCL test case ID
- `formatted_prompt`: The exact string sent to the model
- `model_response`: The raw model output
- `input_tokens` / `output_tokens`: Token counts

### 3. Analysis Script

Run `./analyze_debug_prompts.py` to interactively view the logged data:

```bash
cd gemma_tool_use/evaluation/bfcl
python analyze_debug_prompts.py
```

The script will:
- Match prompts with their corresponding responses
- Show formatted prompts and model outputs side-by-side
- Provide summary statistics
- Highlight potential issues (e.g., unusually long responses)

## How to Use

### 1. Run the Evaluation

Run your BFCL evaluation as normal:

```bash
cd gemma_tool_use/evaluation/bfcl/berkeley-function-call-leaderboard

# Example command (adjust model name and parameters as needed)
python openfunctions_evaluation.py \
    --model your-gemma-model-name \
    --test-category simple \
    --num-gpus 1
```

As it runs, you'll see debug output printed to the console for each test case.

### 2. Analyze the Debug Log

After running (or while it's running), analyze the debug log:

```bash
cd gemma_tool_use/evaluation/bfcl
python analyze_debug_prompts.py
```

This will show you:
- The exact formatted prompts being sent
- The exact responses received
- Any patterns in problematic outputs

### 3. Inspect Specific Test Cases

The debug log file (`gemma_fc_debug_prompts.jsonl`) is in JSON Lines format, so you can also process it with standard tools:

```bash
# View all prompts for a specific test ID
grep '"test_id": "simple_javascript_48"' gemma_fc_debug_prompts.jsonl | jq .

# Count total prompts
grep '"formatted_prompt"' gemma_fc_debug_prompts.jsonl | wc -l

# Find responses longer than 2000 characters (potential gibberish)
jq 'select(.response_length > 2000)' gemma_fc_debug_prompts.jsonl
```

## Common Issues to Look For

### 1. Malformed Prompts

Check if the formatted prompt has the correct structure:
- Should start with `<bos><start_of_turn>user`
- Tools should be in `<tools>...</tools>` tags
- Should end with `<start_of_turn>model\n`

### 2. Missing or Incorrect Special Tokens

Verify that special tokens match your training data:
- `<bos>` at the start
- `<start_of_turn>` and `<end_of_turn>` for each turn
- `<tools>`, `<tool_call>`, `<tool_response>` tags

### 3. Tool Format Mismatch

Ensure the tool definitions in the prompt match the format your model was trained on.

### 4. Context Length Issues

If input_tokens is close to the model's max context length, the model might not have enough space to generate a proper response.

## Troubleshooting

### The debug log file is empty

Make sure:
1. The evaluation is using the `GemmaFCHandler` class
2. The model name in your evaluation command matches a registered Gemma model
3. The evaluation is actually running (check for errors)

### I see console output but no file

Check:
1. You have write permissions in the working directory
2. The `debug_log_path` in `gemma_fc.py` is accessible
3. No exceptions are being raised during file writing

### The analysis script shows mismatched prompts/responses

This can happen if:
1. The evaluation was interrupted and restarted
2. Multiple evaluations are writing to the same log file

Solution: Delete `gemma_fc_debug_prompts.jsonl` before starting a new evaluation run.

## Cleaning Up

To start fresh:

```bash
rm gemma_fc_debug_prompts.jsonl
```

The file will be recreated on the next evaluation run.
