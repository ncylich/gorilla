import json
import re
import threading
from typing import Any
from pathlib import Path

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override


class GemmaFCHandler(OSSHandler):
    """
    Handler for fine-tuned Gemma models with function calling support.
    Uses Qwen-style <tools>, <tool_call>, and <tool_response> XML tags
    with Gemma's chat template (<start_of_turn>...<end_of_turn>).
    """

    SYSTEM_PROMPT = """To get data you don't have access to, you may use the appropriate tools:
1. Call the tool with <tool_call>{"name": "...", "parameters": {...}}</tool_call>
2. You will receive tool results as: <tool_response>{"name": "...", "result": ...}</tool_response>
3. Use the result to answer the user's question.
You can make multiple tool calls, but only use tools when necessary.
"""

    # Class-level shared state for thread-safe file writing
    _debug_lock = threading.Lock()
    _debug_log_path = Path("gemma_fc_debug_prompts.jsonl")

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_name_huggingface = model_name

        # Instance-level storage for current inference
        self._current_log_entry = {}

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        # Model response format:
        # "<tool_call>\n{\"name\": \"function_name\", \"parameters\": {...}}\n</tool_call>"
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [
            {call["name"]: {k: v for k, v in call["parameters"].items()}}
            for call in tool_calls
        ]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        decoded_result = []
        for item in tool_calls:
            decoded_result.append({item["name"]: item["parameters"]})
        return convert_to_function_call(decoded_result)

    @override
    def _format_prompt(self, messages, function):
        """
        Format prompt using Gemma's chat template with function calling.

        Format:
        - Tools: <tools>[list of tool dicts]</tools> prepended to first user message
        - Tool calls: <tool_call>{"name": "...", "parameters": {...}}</tool_call>
        - Tool responses: <tool_response>{"name": "...", "result": {...}}</tool_response>

        Note: Gemma 3 does NOT support the 'system' role. System instructions must be
        prepended to the first user message as per official Gemma 3 documentation.
        """

        formatted_prompt = "<bos>"

        # Extract system message if present (will be prepended to first user message)
        system_content = ""
        if len(messages) > 0 and messages[0]["role"] == "system":
            system_content = messages[0]["content"]
            messages = messages[1:]  # Remove system message from list

        # Process conversation messages
        is_first_user_message = True
        for message in messages:
            role = message["role"]
            content = message["content"]

            # Map assistant to model for Gemma
            if role == "assistant":
                role = "model"

            if role in ["user", "model"]:
                formatted_prompt += f"<start_of_turn>{role}\n"

                # For first user message: prepend system instructions and tools
                if role == "user" and is_first_user_message:
                    is_first_user_message = False

                    # Add system instructions if present
                    if system_content:
                        formatted_prompt += system_content + "\n\n"

                    # Add tools definition if functions are provided
                    if len(function) > 0:
                        # formatted_prompt += self.SYSTEM_PROMPT
                        formatted_prompt += "Here are the available tools that you can use:\n"
                        formatted_prompt += "<tools>\n"
                        formatted_prompt += '\n'.join(json.dumps(tool, separators=(',', ':')) for tool in function)
                        formatted_prompt += "\n</tools>\n\n"

                # Add regular content
                if content:
                    formatted_prompt += content

                # Add tool calls if present
                if "tool_calls" in message and message["tool_calls"]:
                    if content:
                        formatted_prompt += "\n"
                    for tool_call in message["tool_calls"]:
                        if "function" in tool_call:
                            tool_call = tool_call["function"]

                        formatted_prompt += '<tool_call>\n'
                        formatted_prompt += json.dumps({
                            "name": tool_call["name"],
                            "parameters": tool_call.get("parameters", tool_call.get("arguments", tool_call.get("args", {})))
                        }, separators=(',', ':'))
                        formatted_prompt += '\n</tool_call>\n'
                    assert formatted_prompt.endswith('\n')
                    formatted_prompt = formatted_prompt[:-1]  # Remove last newline
                
                formatted_prompt += "<end_of_turn>\n"

            elif role == "tool":
                # Tool responses come as user messages with JSON format
                formatted_prompt += "<start_of_turn>user\n"

                # Parse the execution result (content) - it's already a string
                # Try to parse it as JSON, otherwise use it as-is
                try:
                    result_data = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    result_data = content

                # Format tool response with name and result fields per PLAN.md
                tool_response = {
                    "name": message.get("name", ""),
                    "result": result_data
                }

                formatted_prompt += "<tool_response>\n"
                formatted_prompt += json.dumps(tool_response, separators=(',', ':'))
                formatted_prompt += "\n</tool_response>"
                formatted_prompt += "<end_of_turn>\n"

        # Add generation prompt
        formatted_prompt += "<start_of_turn>model\n"

        # Store prompt data for later combination with response
        self._current_log_entry = {
            "test_id": getattr(self, "current_test_id", "unknown"),
            "messages": messages,
            "functions": function,
            "formatted_prompt": formatted_prompt
        }

        return formatted_prompt

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        # Store test_entry ID for debugging
        self.current_test_id = test_entry.get("id", "unknown")
        # FC models use their own system prompt, so no need to add messages
        return {"message": [], "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_response = api_response.choices[0].text

        # Combine prompt and response data
        complete_entry = {
            **self._current_log_entry,  # Contains test_id, messages, functions, formatted_prompt
            "model_response": model_response,
            "response_length": len(model_response),
            "input_tokens": api_response.usage.prompt_tokens,
            "output_tokens": api_response.usage.completion_tokens
        }

        # Thread-safe file write: lock only protects the file I/O
        with self._debug_lock:
            with open(self._debug_log_path, "a") as f:
                f.write(json.dumps(complete_entry) + "\n")

        extracted_tool_calls = self._extract_tool_calls(model_response)

        # Remove tool call XML from display text
        cleaned_response = model_response
        for match in re.finditer(r"<tool_call>.*?</tool_call>", model_response, re.DOTALL):
            cleaned_response = cleaned_response.replace(match.group(0), "").strip()

        if len(extracted_tool_calls) > 0:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": cleaned_response if cleaned_response else "",
                "tool_calls": extracted_tool_calls,
            }
        else:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": cleaned_response,
            }

        return {
            "model_responses": cleaned_response,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"],
        )
        return inference_data

    @staticmethod
    def _extract_tool_calls(input_string):
        """Extract tool calls from <tool_call>...</tool_call> XML tags."""
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)

        result = []
        for match in matches:
            try:
                tool_call = json.loads(match)
                result.append(tool_call)
            except Exception:
                # Skip malformed JSON
                pass
        return result
