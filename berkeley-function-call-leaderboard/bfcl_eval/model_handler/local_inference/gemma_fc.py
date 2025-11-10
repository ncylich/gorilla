import ast
import json
import re
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override


class GemmaFCHandler(OSSHandler):
    """
    Handler for fine-tuned Gemma models with function calling support.
    Uses philschmid's Python-style format with ```tool_code and ```tool_output blocks
    with Gemma's chat template (<start_of_turn>...<end_of_turn>).

    Reference: https://www.philschmid.de/gemma-function-calling
    """

    SYSTEM_PROMPT = """At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```. \
The python methods described below are imported and available, you can only use defined methods. \
The generated code should be readable and efficient. \
The response to a method will be wrapped in ```tool_output``` use it to call more tools or generate a helpful, friendly response. \
When using a ```tool_call``` think step by step why and how it should be used.

The following Python methods are available:

"""

    @staticmethod
    def _format_tools_as_python(tools):
        """
        Format tools as Python function signatures with docstrings.
        Following philschmid's format from https://www.philschmid.de/gemma-function-calling

        Args:
            tools: List of tool definitions in BFCL/OpenAI format

        Returns:
            Python code block with function signatures and docstrings
        """
        python_functions = []

        for tool in tools:
            # Convert to nested format if needed
            if "type" not in tool:
                tool = {"type": "function", "function": tool}

            func = tool['function']
            func_name = func['name']
            description = func.get('description', '')
            parameters = func.get('parameters', {})
            properties = parameters.get('properties', {})

            # Build function signature with type hints
            params = []
            for param_name, param_info in properties.items():
                param_type = param_info.get('type', 'any')

                # Map JSON schema types to Python types
                type_map = {
                    'string': 'str',
                    'number': 'float',
                    'integer': 'int',
                    'boolean': 'bool',
                    'array': 'list',
                    'object': 'dict',
                    'any': 'Any'
                }
                python_type = type_map.get(param_type, 'Any')

                params.append(f"{param_name}: {python_type}")

            # Replace hyphens with underscores for valid Python identifiers
            func_name_safe = func_name.replace('-', '_')
            signature = f"def {func_name_safe}({', '.join(params)}):"

            # Build docstring
            docstring_lines = []
            if description:
                # Split description into lines and indent each line properly
                desc_lines = description.split('\n')
                docstring_lines.append(f'  """{desc_lines[0]}')
                for line in desc_lines[1:]:
                    # Add proper indentation (2 spaces) to each line of the description
                    if line.strip():  # Only indent non-empty lines
                        docstring_lines.append(f'  {line}')
                    else:
                        docstring_lines.append('')
            else:
                docstring_lines.append('  """')

            # Add Args section if there are parameters
            if properties:
                docstring_lines.append('')
                docstring_lines.append('  Args:')
                for param_name, param_info in properties.items():
                    param_desc = param_info.get('description', 'No description')
                    docstring_lines.append(f'    {param_name}: {param_desc}')

            docstring_lines.append('  """')

            # Combine signature and docstring
            function_def = signature + '\n' + '\n'.join(docstring_lines)
            python_functions.append(function_def)

        # Wrap all functions in a Python code block
        all_functions = '\n\n'.join(python_functions)
        return f'```python\n{all_functions}\n```'

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

    def _validate_and_extract_tool_calls(self, result):
        """Extract and validate tool calls from model response."""
        tool_calls = self._extract_tool_calls(result)
        if not isinstance(tool_calls, list) or any(not isinstance(item, dict) for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return tool_calls

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        tool_calls = self._validate_and_extract_tool_calls(result)
        return [{call["name"]: call["arguments"]} for call in tool_calls]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = self._validate_and_extract_tool_calls(result)
        return convert_to_function_call([{call["name"]: call["arguments"]} for call in tool_calls])

    @override
    def _format_prompt(self, messages, function):
        """
        Format prompt using Gemma's chat template with philschmid-style function calling.

        Format:
        - Tools: Python functions with docstrings in ```python blocks
        - Tool calls: ```tool_code\nfunction_name(arg1=value1)\n```
        - Tool responses: ```tool_output\nresult\n```

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
                        formatted_prompt += self.SYSTEM_PROMPT
                        formatted_prompt += self._format_tools_as_python(function)
                        formatted_prompt += "\n\n"

                # Add regular content
                if content:
                    formatted_prompt += content

                # Add tool calls if present (convert to philschmid format)
                if "tool_calls" in message and message["tool_calls"]:
                    if content:
                        formatted_prompt += "\n"
                    for tool_call in message["tool_calls"]:
                        if "function" in tool_call:
                            tool_call = tool_call["function"]

                        # Convert to philschmid's format: function_name(arg1=value1, arg2=value2)
                        func_name = tool_call["name"].replace('-', '_')
                        args = tool_call.get("parameters", tool_call.get("arguments", tool_call.get("args", {})))
                        args_str = ', '.join([f"{k}={json.dumps(v)}" for k, v in args.items()])

                        formatted_prompt += f'```tool_code\n{func_name}({args_str})\n```\n'

                    assert formatted_prompt.endswith('\n')
                    formatted_prompt = formatted_prompt[:-1]  # Remove last newline

                formatted_prompt += "<end_of_turn>\n"

            elif role == "tool":
                # Tool responses come as user messages with ```tool_output format
                formatted_prompt += "<start_of_turn>user\n"

                # Parse the execution result (content) - it's already a string
                # Try to parse it as JSON, otherwise use it as-is
                try:
                    result_data = json.loads(content)
                    # If it's JSON, use it as-is
                    result_str = json.dumps(result_data) if isinstance(result_data, (dict, list)) else str(result_data)
                except (json.JSONDecodeError, TypeError):
                    result_str = content

                formatted_prompt += f'```tool_output\n{result_str}\n```'
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

        if len(extracted_tool_calls) > 0:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": model_response if model_response else "",
                "tool_calls": extracted_tool_calls,
            }
        else:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": model_response,
            }

        # Return RAW response for BFCL evaluation, not cleaned version
        return {
            "model_responses": model_response,  # Raw response with tool calls intact
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
        """
        Extract tool calls from ```tool_code blocks following philschmid's format.

        Expected format: ```tool_code\nfunction_name(arg1=value1, arg2=value2)\n```

        Returns list of dicts with format: {"name": "function_name", "arguments": {...}}

        Uses Python's ast module for safe and reliable parsing.
        """

        pattern = r"```tool_code\s*(.*?)\s*```"
        matches = re.findall(pattern, input_string, re.DOTALL)

        result = []
        for match in matches:
            try:
                # Parse the function call using ast
                # This safely handles all Python syntax including nested structures
                code = match.strip()

                # Parse as Python expression
                tree = ast.parse(code, mode='eval')

                # The tree should be an Expr containing a Call
                if not isinstance(tree.body, ast.Call):
                    continue

                call_node = tree.body

                # Extract function name
                if isinstance(call_node.func, ast.Name):
                    func_name = call_node.func.id
                elif isinstance(call_node.func, ast.Attribute):
                    # Handle cases like module.function_name
                    func_name = call_node.func.attr
                else:
                    continue

                # Extract keyword arguments (our format only uses keyword args)
                arguments = {
                    keyword.arg: ast.literal_eval(keyword.value)
                    for keyword in call_node.keywords
                }

                result.append({
                    "name": func_name,
                    "arguments": arguments
                })

            except (SyntaxError, ValueError):
                # Skip malformed calls (invalid syntax or non-literal values)
                continue

        return result
