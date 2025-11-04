import json
import re
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override


class GemmaFCHandler(OSSHandler):
    """
    Handler for fine-tuned Gemma models with function calling support.
    Uses Qwen-style <tools>, <tool_call>, and <tool_response> XML tags
    with Gemma's chat template (<start_of_turn>...<end_of_turn>).
    """

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

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        # Model response format:
        # "<tool_call>\n{\"name\": \"function_name\", \"args\": {...}}\n</tool_call>"
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [
            {call["name"]: {k: v for k, v in call["args"].items()}}
            for call in tool_calls
        ]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        decoded_result = []
        for item in tool_calls:
            if type(item) == str:
                item = eval(item)
            decoded_result.append({item["name"]: item["args"]})
        return convert_to_function_call(decoded_result)

    @override
    def _format_prompt(self, messages, function):
        """
        Format prompt using Gemma's chat template with function calling.

        Format:
        - Tools: <tools>[list of tool dicts]</tools> in system prompt
        - Tool calls: <tool_call>{"name": "...", "args": {...}}</tool_call>
        - Tool responses: <tool_response>{"name": "...", "result": {...}}</tool_response>
        """
        formatted_prompt = "<bos>"

        # Add system message with tools if present
        if len(function) > 0:
            formatted_prompt += "<start_of_turn>system\n"

            # Add existing system message if present
            if messages[0]["role"] == "system":
                formatted_prompt += messages[0]["content"] + "\n\n"

            # Add tools definition
            formatted_prompt += "<tools>\n"
            formatted_prompt += json.dumps(function, indent=2)
            formatted_prompt += "\n</tools><end_of_turn>\n"

            # Skip system message in main loop
            messages = messages[1:] if messages[0]["role"] == "system" else messages
        else:
            # No tools, just add system message if present
            if messages[0]["role"] == "system":
                formatted_prompt += f"<start_of_turn>system\n{messages[0]['content']}<end_of_turn>\n"
                messages = messages[1:]

        # Process conversation messages
        for message in messages:
            role = message["role"]
            content = message["content"]

            # Map assistant to model for Gemma
            if role == "assistant":
                role = "model"

            if role in ["user", "model"]:
                formatted_prompt += f"<start_of_turn>{role}\n"

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
                            "args": tool_call.get("arguments", tool_call.get("args", {}))
                        })
                        formatted_prompt += '\n</tool_call>\n'

                formatted_prompt += "<end_of_turn>\n"

            elif role == "tool":
                # Tool responses come as user messages
                formatted_prompt += f"<start_of_turn>user\n"
                formatted_prompt += f"<tool_response>\n{content}\n</tool_response>"
                formatted_prompt += "<end_of_turn>\n"

        # Add generation prompt
        formatted_prompt += "<start_of_turn>model\n"
        return formatted_prompt

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        # FC models use their own system prompt, so no need to add messages
        return {"message": [], "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_response = api_response.choices[0].text
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
            except Exception as e:
                pass
        return result
