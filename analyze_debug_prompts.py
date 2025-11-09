#!/usr/bin/env python3
"""
Analyze debug prompts from BFCL evaluation.

This script reads the gemma_fc_debug_prompts.jsonl file and helps you
inspect the exact prompts being sent to the model and the responses received.
"""

import json
import sys
from pathlib import Path


def main():
    debug_file = Path("gemma_fc_debug_prompts.jsonl")

    if not debug_file.exists():
        print(f"Error: {debug_file} not found.")
        print("Run the BFCL evaluation first to generate the debug log.")
        sys.exit(1)

    print(f"Reading from {debug_file}...")
    print("=" * 80)

    prompts = {}
    responses = {}

    with open(debug_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            prompt_id = entry.get("prompt_id")
            if "formatted_prompt" in entry:
                prompts[prompt_id] = entry
            elif "model_response" in entry:
                responses[prompt_id] = entry

    print(f"\nFound {len(prompts)} prompts and {len(responses)} responses")
    print("=" * 80)

    # Match prompts with responses by prompt_id
    matched_pairs = []
    for prompt_id in sorted(prompts.keys()):
        if prompt_id in responses:
            matched_pairs.append((prompts[prompt_id], responses[prompt_id]))
        else:
            print(f"⚠️  Warning: No response found for prompt_id {prompt_id}")

    print(f"Matched {len(matched_pairs)} prompt-response pairs")
    print("=" * 80)

    # Display matched pairs
    for i, (prompt, response) in enumerate(matched_pairs):

        print(f"\n{'=' * 80}")
        print(f"Test ID: {prompt.get('test_id', 'unknown')}")
        print(f"Prompt ID: {prompt.get('prompt_id', 'unknown')}")
        print(f"{'=' * 80}")

        print("\n--- FORMATTED PROMPT ---")
        print(prompt["formatted_prompt"])

        print("\n--- MODEL RESPONSE ---")
        print(f"Length: {response['response_length']} characters")
        print(f"Input tokens: {response['input_tokens']}")
        print(f"Output tokens: {response['output_tokens']}")
        print(f"\nResponse text:")
        print(response["model_response"])

        print("\n" + "=" * 80)

        # Ask if user wants to continue
        if i < len(matched_pairs) - 1:
            user_input = input("\nPress Enter to see next example, 'q' to quit, or a number to jump to that test: ")
            if user_input.lower() == 'q':
                break
            elif user_input.isdigit():
                target_idx = int(user_input)
                if 0 <= target_idx < len(matched_pairs):
                    i = target_idx - 1  # Will be incremented in next iteration
                    continue

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if matched_pairs:
        response_list = [r for _, r in matched_pairs]
        total_input_tokens = sum(r["input_tokens"] for r in response_list)
        total_output_tokens = sum(r["output_tokens"] for r in response_list)
        avg_input_tokens = total_input_tokens / len(response_list)
        avg_output_tokens = total_output_tokens / len(response_list)
        avg_response_length = sum(r["response_length"] for r in response_list) / len(response_list)

        print(f"Total prompts: {len(prompts)}")
        print(f"Total responses: {len(responses)}")
        print(f"Matched pairs: {len(matched_pairs)}")
        print(f"Average input tokens: {avg_input_tokens:.1f}")
        print(f"Average output tokens: {avg_output_tokens:.1f}")
        print(f"Average response length: {avg_response_length:.1f} characters")

        # Find potential issues
        long_responses = [r for r in response_list if r["response_length"] > 2000]
        if long_responses:
            print(f"\n⚠️  Found {len(long_responses)} responses longer than 2000 characters")
            print("These might indicate gibberish generation:")
            for r in long_responses[:5]:  # Show first 5
                print(f"  - Test ID: {r['test_id']}, Length: {r['response_length']}")


if __name__ == "__main__":
    main()
