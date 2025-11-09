#!/usr/bin/env python3
"""
Extract examples where the model generated gibberish (unusually long outputs).

This helps quickly identify and analyze problematic test cases.
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

    # Threshold for considering a response as gibberish
    # Adjust this based on your observations
    GIBBERISH_THRESHOLD = 1500  # characters

    print(f"Reading from {debug_file}...")
    print(f"Looking for responses longer than {GIBBERISH_THRESHOLD} characters...\n")

    prompts = {}
    gibberish_responses = []

    with open(debug_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            if "formatted_prompt" in entry:
                prompt_id = entry.get("prompt_id")
                prompts[prompt_id] = entry
            elif "model_response" in entry:
                if entry.get("response_length", 0) > GIBBERISH_THRESHOLD:
                    gibberish_responses.append(entry)

    if not gibberish_responses:
        print("✅ No gibberish responses found!")
        print(f"   All responses are under {GIBBERISH_THRESHOLD} characters.")
        return

    print(f"⚠️  Found {len(gibberish_responses)} responses with potential gibberish\n")
    print("=" * 80)

    for i, response in enumerate(gibberish_responses, 1):
        prompt_id = response.get("prompt_id")
        test_id = response.get("test_id", "unknown")
        response_length = response.get("response_length", 0)

        print(f"\n{i}. Test ID: {test_id}")
        print(f"   Prompt ID: {prompt_id}")
        print(f"   Response length: {response_length} characters")
        print(f"   Output tokens: {response.get('output_tokens', 'unknown')}")

        # Get the corresponding prompt
        if prompt_id in prompts:
            prompt = prompts[prompt_id]
            print(f"\n   --- PROMPT ---")
            prompt_text = prompt["formatted_prompt"]
            # Show first and last 500 chars of prompt
            if len(prompt_text) > 1000:
                print(f"   {prompt_text[:500]}")
                print(f"   ... ({len(prompt_text) - 1000} more chars) ...")
                print(f"   {prompt_text[-500:]}")
            else:
                print(f"   {prompt_text}")

        print(f"\n   --- RESPONSE (first 500 chars) ---")
        response_text = response["model_response"]
        print(f"   {response_text[:500]}")
        if len(response_text) > 500:
            print(f"   ... ({len(response_text) - 500} more chars) ...")

        print("\n" + "=" * 80)

    # Save problematic test IDs to a file
    output_file = Path("gibberish_test_ids.txt")
    with open(output_file, "w") as f:
        for response in gibberish_responses:
            f.write(f"{response.get('test_id', 'unknown')}\n")

    print(f"\n✅ Saved {len(gibberish_responses)} test IDs to {output_file}")
    print(f"\nTo re-run just these test cases, you can use the --run-ids flag")
    print(f"after setting up the test_ids_to_generate.txt file in the BFCL config.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            GIBBERISH_THRESHOLD = int(sys.argv[1])
            print(f"Using custom threshold: {GIBBERISH_THRESHOLD} characters")
        except ValueError:
            print(f"Invalid threshold: {sys.argv[1]}")
            print("Usage: python extract_gibberish_examples.py [threshold_in_chars]")
            sys.exit(1)

    main()
