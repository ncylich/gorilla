#!/usr/bin/env python3
"""
Analyze debug log where each entry contains both prompt and response.
Much simpler than the old two-entry format!
"""

import json
import sys
from pathlib import Path


def main():
    debug_file = Path("oss_model_debug_prompts.jsonl")

    if not debug_file.exists():
        print(f"Error: {debug_file} not found.")
        print("Run the BFCL evaluation first to generate the debug log.")
        print("The file should be in the directory where you ran the BFCL command.")
        sys.exit(1)

    print(f"Reading from {debug_file}...")
    print("=" * 80)

    entries = []
    with open(debug_file, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")

    print(f"\nTotal entries: {len(entries)}")
    print("=" * 80)

    # Display entries interactively
    for i, entry in enumerate(entries):
        print(f"\n{'=' * 80}")
        print(f"Entry {i+1}/{len(entries)}")
        print(f"Test ID: {entry.get('test_id', 'unknown')}")
        print(f"{'=' * 80}")

        print("\n--- FORMATTED PROMPT ---")
        print(entry.get("formatted_prompt", "N/A"))

        print("\n--- MODEL RESPONSE ---")
        print(f"Length: {entry.get('response_length', 0)} characters")
        print(f"Input tokens: {entry.get('input_tokens', 0)}")
        print(f"Output tokens: {entry.get('output_tokens', 0)}")
        print(f"\nResponse text:")
        print(entry.get("model_response", "N/A"))

        print("\n" + "=" * 80)

        # Ask if user wants to continue
        if i < len(entries) - 1:
            user_input = input(
                "\nPress Enter to see next example, 'q' to quit, or a number to jump to that entry: "
            )
            if user_input.lower() == "q":
                break
            elif user_input.isdigit():
                target_idx = int(user_input) - 1  # User sees 1-indexed
                if 0 <= target_idx < len(entries):
                    # Jump to target (will be incremented in next iteration)
                    for _ in range(target_idx - i - 1):
                        next(iter(entries), None)
                    continue

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if entries:
        total_input_tokens = sum(e.get("input_tokens", 0) for e in entries)
        total_output_tokens = sum(e.get("output_tokens", 0) for e in entries)
        avg_input_tokens = total_input_tokens / len(entries)
        avg_output_tokens = total_output_tokens / len(entries)
        avg_response_length = sum(e.get("response_length", 0) for e in entries) / len(
            entries
        )

        print(f"Total entries: {len(entries)}")
        print(f"Average input tokens: {avg_input_tokens:.1f}")
        print(f"Average output tokens: {avg_output_tokens:.1f}")
        print(f"Average response length: {avg_response_length:.1f} characters")

        # Check for empty responses
        empty_responses = [e for e in entries if not e.get("model_response", "").strip()]
        empty_percentage = 100 * len(empty_responses) / len(entries)
        print(f"\nEmpty responses: {len(empty_responses)} ({empty_percentage:.1f}%)")
        if empty_responses:
            print("⚠️  Sample test IDs with empty responses:")
            for e in empty_responses[:5]:
                print(f"  - {e.get('test_id', 'unknown')}")

        # Find potential issues
        long_responses = [e for e in entries if e.get("response_length", 0) > 2000]
        if long_responses:
            print(
                f"\n⚠️  Found {len(long_responses)} responses longer than 2000 characters"
            )
            print("These might indicate gibberish generation:")
            for e in long_responses[:5]:
                print(
                    f"  - Test ID: {e.get('test_id', 'unknown')}, Length: {e.get('response_length', 0)}"
                )

        # Check for tool calls
        responses_with_tool_calls = [
            e for e in entries if "<tool_call>" in e.get("model_response", "") or "```tool_call" in e.get("model_response", "")
        ]
        print(
            f"\n✓ {len(responses_with_tool_calls)} responses contain tool calls ({100*len(responses_with_tool_calls)/len(entries):.1f}%)"
        )


if __name__ == "__main__":
    main()
