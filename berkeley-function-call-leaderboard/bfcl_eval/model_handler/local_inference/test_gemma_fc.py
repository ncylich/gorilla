"""
Unit tests for GemmaFCHandler.

Tests the function calling handler for fine-tuned Gemma models, verifying:
- Tool call extraction from XML tags
- Prompt formatting with Gemma's chat template
- Decoding of AST and execution formats
- Handling of various response formats (correct and malformed)
"""

import json
import unittest
from typing import Any
from unittest.mock import MagicMock, Mock, patch

from bfcl_eval.model_handler.local_inference.gemma_fc import GemmaFCHandler


class MockAPIResponse:
    """Mock object for OpenAI API response."""

    def __init__(self, text: str, prompt_tokens: int = 100, completion_tokens: int = 50):
        self.choices = [Mock(text=text)]
        self.usage = Mock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class TestGemmaFCHandler(unittest.TestCase):
    """Test suite for GemmaFCHandler."""

    def setUp(self):
        """Set up test handler instance."""
        # Mock the parent class initialization to avoid server setup
        with patch.object(GemmaFCHandler, '__init__', lambda x, *args, **kwargs: None):
            self.handler = GemmaFCHandler(
                model_name="google/gemma-3-270m-it",
                temperature=0.0,
                registry_name="google/gemma-3-270m-it-FC",
                is_fc_model=True
            )
            # Manually set required attributes
            self.handler.model_name_huggingface = "google/gemma-3-270m-it"

    # ========================================================================
    # Test _extract_tool_calls (static method)
    # ========================================================================

    def test_extract_single_tool_call(self):
        """Test extraction of a single valid tool call."""
        response = '''<tool_call>
{"name": "get_weather", "args": {"location": "San Francisco", "unit": "celsius"}}
</tool_call>'''

        result = GemmaFCHandler._extract_tool_calls(response)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "get_weather")
        self.assertEqual(result[0]["args"]["location"], "San Francisco")
        self.assertEqual(result[0]["args"]["unit"], "celsius")

    def test_extract_multiple_tool_calls(self):
        """Test extraction of multiple tool calls."""
        response = '''<tool_call>
{"name": "play_song", "args": {"artist": "Taylor Swift", "duration": 20}}
</tool_call>
Some text in between
<tool_call>
{"name": "play_song", "args": {"artist": "Maroon 5", "duration": 15}}
</tool_call>'''

        result = GemmaFCHandler._extract_tool_calls(response)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "play_song")
        self.assertEqual(result[0]["args"]["artist"], "Taylor Swift")
        self.assertEqual(result[1]["name"], "play_song")
        self.assertEqual(result[1]["args"]["artist"], "Maroon 5")

    def test_extract_tool_call_with_whitespace(self):
        """Test extraction with extra whitespace."""
        response = '''<tool_call>

{"name": "calculator", "args": {"operation": "add", "a": 5, "b": 3}}

</tool_call>'''

        result = GemmaFCHandler._extract_tool_calls(response)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "calculator")
        self.assertEqual(result[0]["args"]["a"], 5)

    def test_extract_no_tool_calls(self):
        """Test extraction when no tool calls are present."""
        response = "Just a regular text response without any tool calls."

        result = GemmaFCHandler._extract_tool_calls(response)

        self.assertEqual(len(result), 0)

    def test_extract_malformed_json(self):
        """Test extraction with malformed JSON (should be skipped)."""
        response = '''<tool_call>
{"name": "bad_function", "args": {invalid json}}
</tool_call>
<tool_call>
{"name": "good_function", "args": {"param": "value"}}
</tool_call>'''

        result = GemmaFCHandler._extract_tool_calls(response)

        # Should only extract the valid one
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "good_function")

    def test_extract_empty_tool_call_tags(self):
        """Test extraction with empty tool call tags."""
        response = '<tool_call></tool_call>'

        result = GemmaFCHandler._extract_tool_calls(response)

        self.assertEqual(len(result), 0)

    def test_extract_nested_objects(self):
        """Test extraction with nested object arguments."""
        response = '''<tool_call>
{"name": "create_event", "args": {"title": "Meeting", "attendees": ["alice@example.com", "bob@example.com"], "details": {"room": "A1", "duration": 60}}}
</tool_call>'''

        result = GemmaFCHandler._extract_tool_calls(response)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "create_event")
        self.assertIsInstance(result[0]["args"]["attendees"], list)
        self.assertIsInstance(result[0]["args"]["details"], dict)
        self.assertEqual(result[0]["args"]["details"]["room"], "A1")

    # ========================================================================
    # Test decode_ast
    # ========================================================================

    def test_decode_ast_single_call(self):
        """Test decode_ast with a single function call."""
        response = '''<tool_call>
{"name": "get_stock_price", "args": {"symbol": "AAPL"}}
</tool_call>'''

        result = self.handler.decode_ast(response, language="python", has_tool_call_tag=False)

        self.assertEqual(len(result), 1)
        self.assertIn("get_stock_price", result[0])
        self.assertEqual(result[0]["get_stock_price"]["symbol"], "AAPL")

    def test_decode_ast_multiple_calls(self):
        """Test decode_ast with multiple function calls."""
        response = '''<tool_call>
{"name": "func1", "args": {"x": 1}}
</tool_call>
<tool_call>
{"name": "func2", "args": {"y": 2}}
</tool_call>'''

        result = self.handler.decode_ast(response, language="python", has_tool_call_tag=False)

        self.assertEqual(len(result), 2)
        self.assertIn("func1", result[0])
        self.assertIn("func2", result[1])

    def test_decode_ast_invalid_response(self):
        """Test decode_ast with invalid response (returns empty list)."""
        response = "Not a valid tool call"

        # When there are no tool calls, _extract_tool_calls returns []
        # The current implementation doesn't raise ValueError for empty list
        result = self.handler.decode_ast(response, language="python", has_tool_call_tag=False)

        # Should return empty list when no tool calls found
        self.assertEqual(result, [])

    def test_decode_ast_empty_args(self):
        """Test decode_ast with empty arguments."""
        response = '''<tool_call>
{"name": "no_params_func", "args": {}}
</tool_call>'''

        result = self.handler.decode_ast(response, language="python", has_tool_call_tag=False)

        self.assertEqual(len(result), 1)
        self.assertIn("no_params_func", result[0])
        self.assertEqual(result[0]["no_params_func"], {})

    # ========================================================================
    # Test decode_execute
    # ========================================================================

    def test_decode_execute_single_call(self):
        """Test decode_execute with a single function call."""
        response = '''<tool_call>
{"name": "add", "args": {"a": 5, "b": 3}}
</tool_call>'''

        result = self.handler.decode_execute(response, has_tool_call_tag=False)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "add(a=5,b=3)")

    def test_decode_execute_multiple_calls(self):
        """Test decode_execute with multiple function calls."""
        response = '''<tool_call>
{"name": "multiply", "args": {"x": 4, "y": 2}}
</tool_call>
<tool_call>
{"name": "subtract", "args": {"x": 10, "y": 3}}
</tool_call>'''

        result = self.handler.decode_execute(response, has_tool_call_tag=False)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "multiply(x=4,y=2)")
        self.assertEqual(result[1], "subtract(x=10,y=3)")

    def test_decode_execute_string_args(self):
        """Test decode_execute with string arguments."""
        response = '''<tool_call>
{"name": "send_email", "args": {"to": "user@example.com", "subject": "Hello", "body": "Test message"}}
</tool_call>'''

        result = self.handler.decode_execute(response, has_tool_call_tag=False)

        self.assertEqual(len(result), 1)
        self.assertIn("send_email", result[0])
        self.assertIn("to='user@example.com'", result[0])
        self.assertIn("subject='Hello'", result[0])

    def test_decode_execute_invalid_response(self):
        """Test decode_execute with invalid response (returns empty list)."""
        response = "Invalid response format"

        # When there are no tool calls, _extract_tool_calls returns []
        # convert_to_function_call([]) returns []
        result = self.handler.decode_execute(response, has_tool_call_tag=False)

        # Should return empty list when no tool calls found
        self.assertEqual(result, [])

    # ========================================================================
    # Test _format_prompt
    # ========================================================================

    def test_format_prompt_with_tools(self):
        """Test prompt formatting with tools definition (Gemma 3 style: no system role)."""
        messages = [
            {"role": "user", "content": "What's the weather in NYC?"}
        ]
        functions = [
            {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    }
                }
            }
        ]

        result = self.handler._format_prompt(messages, functions)

        # Check structure - Gemma 3 does NOT support system role
        self.assertIn("<bos>", result)
        self.assertNotIn("<start_of_turn>system", result)  # Should NOT have system role
        self.assertIn("<start_of_turn>user", result)
        self.assertIn("<tools>", result)
        self.assertIn("get_weather", result)
        self.assertIn("</tools>", result)
        self.assertIn("What's the weather in NYC?", result)
        self.assertIn("<end_of_turn>", result)
        self.assertIn("<start_of_turn>model", result)

        # Tools should come before user query in the user message
        user_message_start = result.find("<start_of_turn>user")
        tools_start = result.find("<tools>")
        user_content_start = result.find("What's the weather")
        self.assertTrue(user_message_start < tools_start < user_content_start)

    def test_format_prompt_without_tools(self):
        """Test prompt formatting without tools."""
        messages = [
            {"role": "user", "content": "Hello!"}
        ]
        functions = []

        result = self.handler._format_prompt(messages, functions)

        self.assertIn("<bos>", result)
        self.assertNotIn("<tools>", result)
        self.assertIn("<start_of_turn>user", result)
        self.assertIn("Hello!", result)
        self.assertIn("<start_of_turn>model", result)

    def test_format_prompt_with_system_message(self):
        """Test prompt formatting with system message and tools (Gemma 3 style)."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"}
        ]
        functions = [{"name": "test_func", "parameters": {}}]

        result = self.handler._format_prompt(messages, functions)

        # System content should be prepended to first user message, not in separate system turn
        self.assertNotIn("<start_of_turn>system", result)
        self.assertIn("You are a helpful assistant.", result)
        self.assertIn("<tools>", result)
        self.assertIn("test_func", result)
        self.assertIn("Hi", result)

        # Verify order: user turn contains system instructions, then tools, then user content
        user_turn_start = result.find("<start_of_turn>user")
        system_content_pos = result.find("You are a helpful assistant.")
        tools_pos = result.find("<tools>")
        user_content_pos = result.find("Hi")
        self.assertTrue(user_turn_start < system_content_pos < tools_pos < user_content_pos)

    def test_format_prompt_with_system_message_no_tools(self):
        """Test prompt formatting with system message but no tools."""
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"}
        ]
        functions = []

        result = self.handler._format_prompt(messages, functions)

        # System should be prepended to first user message even without tools
        self.assertNotIn("<start_of_turn>system", result)
        self.assertIn("Be concise.", result)
        self.assertIn("Hello", result)

        # Verify order
        user_turn_start = result.find("<start_of_turn>user")
        system_pos = result.find("Be concise.")
        user_pos = result.find("Hello")
        self.assertTrue(user_turn_start < system_pos < user_pos)

    def test_format_prompt_with_assistant_message(self):
        """Test prompt formatting with assistant message (mapped to model)."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        functions = []

        result = self.handler._format_prompt(messages, functions)

        # Assistant should be mapped to model
        self.assertIn("<start_of_turn>model", result)
        self.assertIn("Hi there!", result)
        self.assertNotIn("<start_of_turn>assistant", result)

    def test_format_prompt_with_tool_calls(self):
        """Test prompt formatting with tool calls in assistant message."""
        messages = [
            {"role": "user", "content": "Get weather for NYC"},
            {
                "role": "assistant",
                "content": "I'll check that for you.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "NYC"}
                        }
                    }
                ]
            }
        ]
        functions = [{"name": "get_weather", "parameters": {}}]

        result = self.handler._format_prompt(messages, functions)

        self.assertIn("<tool_call>", result)
        self.assertIn("get_weather", result)
        self.assertIn('"name": "get_weather"', result)
        self.assertIn('"args":', result)
        self.assertIn("NYC", result)

    def test_format_prompt_with_tool_response(self):
        """Test prompt formatting with tool response."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "get_weather", "arguments": {"location": "NYC"}}}]
            },
            {
                "role": "tool",
                "name": "get_weather",
                "content": '{"temperature": 72, "condition": "sunny"}'
            }
        ]
        functions = [{"name": "get_weather", "parameters": {}}]

        result = self.handler._format_prompt(messages, functions)

        self.assertIn("<tool_response>", result)
        self.assertIn("get_weather", result)
        self.assertIn('"result":', result)
        self.assertIn("temperature", result)

    def test_format_prompt_tool_calls_without_function_wrapper(self):
        """Test prompt formatting when tool_calls don't have 'function' wrapper."""
        messages = [
            {"role": "user", "content": "Test"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "test_func",
                        "args": {"param": "value"}
                    }
                ]
            }
        ]
        functions = [{"name": "test_func", "parameters": {}}]

        result = self.handler._format_prompt(messages, functions)

        self.assertIn("<tool_call>", result)
        self.assertIn("test_func", result)
        self.assertIn('"args":', result)

    def test_format_prompt_multiple_tool_calls(self):
        """Test prompt formatting with multiple tool calls."""
        messages = [
            {"role": "user", "content": "Play two songs"},
            {
                "role": "assistant",
                "content": "Playing songs",
                "tool_calls": [
                    {"function": {"name": "play", "arguments": {"song": "A"}}},
                    {"function": {"name": "play", "arguments": {"song": "B"}}}
                ]
            }
        ]
        functions = [{"name": "play", "parameters": {}}]

        result = self.handler._format_prompt(messages, functions)

        # Should have two tool_call tags
        self.assertEqual(result.count("<tool_call>"), 2)
        self.assertIn('"song": "A"', result)
        self.assertIn('"song": "B"', result)

    # ========================================================================
    # Test _pre_query_processing_prompting
    # ========================================================================

    def test_pre_query_processing(self):
        """Test pre-query processing returns empty messages with functions."""
        test_entry = {
            "function": [
                {"name": "func1", "parameters": {}},
                {"name": "func2", "parameters": {}}
            ]
        }

        result = self.handler._pre_query_processing_prompting(test_entry)

        self.assertEqual(result["message"], [])
        self.assertEqual(len(result["function"]), 2)
        self.assertEqual(result["function"][0]["name"], "func1")

    # ========================================================================
    # Test _parse_query_response_prompting
    # ========================================================================

    def test_parse_query_response_with_tool_calls(self):
        """Test parsing response with tool calls."""
        api_response = MockAPIResponse(
            text='Let me help you.\n<tool_call>\n{"name": "search", "args": {"query": "test"}}\n</tool_call>',
            prompt_tokens=100,
            completion_tokens=30
        )

        result = self.handler._parse_query_response_prompting(api_response)

        self.assertIn("model_responses", result)
        self.assertIn("model_responses_message_for_chat_history", result)
        self.assertEqual(result["input_token"], 100)
        self.assertEqual(result["output_token"], 30)

        # Check that tool call XML is removed from display text
        self.assertNotIn("<tool_call>", result["model_responses"])
        self.assertIn("Let me help you.", result["model_responses"])

        # Check message format
        msg = result["model_responses_message_for_chat_history"]
        self.assertEqual(msg["role"], "assistant")
        self.assertIn("tool_calls", msg)
        self.assertEqual(len(msg["tool_calls"]), 1)
        self.assertEqual(msg["tool_calls"][0]["name"], "search")

    def test_parse_query_response_without_tool_calls(self):
        """Test parsing response without tool calls."""
        api_response = MockAPIResponse(
            text='This is a regular response.',
            prompt_tokens=50,
            completion_tokens=10
        )

        result = self.handler._parse_query_response_prompting(api_response)

        self.assertEqual(result["model_responses"], "This is a regular response.")

        # Check message format
        msg = result["model_responses_message_for_chat_history"]
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "This is a regular response.")
        self.assertNotIn("tool_calls", msg)

    def test_parse_query_response_tool_call_only(self):
        """Test parsing response with only tool call, no text."""
        api_response = MockAPIResponse(
            text='<tool_call>\n{"name": "calculator", "args": {"op": "add"}}\n</tool_call>'
        )

        result = self.handler._parse_query_response_prompting(api_response)

        # Cleaned response should be empty or minimal
        self.assertEqual(result["model_responses"].strip(), "")

        # But should have tool calls
        msg = result["model_responses_message_for_chat_history"]
        self.assertEqual(len(msg["tool_calls"]), 1)
        self.assertEqual(msg["content"], "")

    def test_parse_query_response_multiple_tool_calls(self):
        """Test parsing response with multiple tool calls."""
        api_response = MockAPIResponse(
            text='<tool_call>\n{"name": "func1", "args": {"x": 1}}\n</tool_call>\n<tool_call>\n{"name": "func2", "args": {"y": 2}}\n</tool_call>'
        )

        result = self.handler._parse_query_response_prompting(api_response)

        msg = result["model_responses_message_for_chat_history"]
        self.assertEqual(len(msg["tool_calls"]), 2)

    # ========================================================================
    # Test _add_assistant_message_prompting
    # ========================================================================

    def test_add_assistant_message(self):
        """Test adding assistant message to inference data."""
        inference_data = {"message": [{"role": "user", "content": "Hello"}]}
        model_response_data = {
            "model_responses_message_for_chat_history": {
                "role": "assistant",
                "content": "Hi there!"
            }
        }

        result = self.handler._add_assistant_message_prompting(
            inference_data, model_response_data
        )

        self.assertEqual(len(result["message"]), 2)
        self.assertEqual(result["message"][1]["role"], "assistant")
        self.assertEqual(result["message"][1]["content"], "Hi there!")

    # ========================================================================
    # Edge cases and error handling
    # ========================================================================

    def test_extract_tool_calls_unicode(self):
        """Test extraction with unicode characters."""
        response = '''<tool_call>
{"name": "translate", "args": {"text": "Hello 世界 🌍"}}
</tool_call>'''

        result = GemmaFCHandler._extract_tool_calls(response)

        self.assertEqual(len(result), 1)
        self.assertIn("世界", result[0]["args"]["text"])
        self.assertIn("🌍", result[0]["args"]["text"])

    def test_decode_ast_with_special_characters(self):
        """Test decode_ast with special characters in arguments."""
        response = '''<tool_call>
{"name": "echo", "args": {"message": "Line1\\nLine2\\tTabbed"}}
</tool_call>'''

        result = self.handler.decode_ast(response, language="python", has_tool_call_tag=False)

        self.assertEqual(len(result), 1)
        self.assertIn("echo", result[0])
        # The escaped characters should be preserved
        self.assertIn("Line1", result[0]["echo"]["message"])

    def test_format_prompt_empty_content(self):
        """Test prompt formatting with empty content."""
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""}
        ]
        functions = []

        result = self.handler._format_prompt(messages, functions)

        # Should still produce valid structure
        self.assertIn("<start_of_turn>user", result)
        self.assertIn("<start_of_turn>model", result)

    def test_decode_execute_complex_types(self):
        """Test decode_execute with complex argument types."""
        response = '''<tool_call>
{"name": "process_data", "args": {"values": [1, 2, 3], "config": {"verbose": true, "threshold": 0.5}}}
</tool_call>'''

        result = self.handler.decode_execute(response, has_tool_call_tag=False)

        self.assertEqual(len(result), 1)
        # Check that lists and dicts are properly formatted in the execution string
        self.assertIn("process_data", result[0])
        self.assertIn("values=[1, 2, 3]", result[0])
        self.assertIn("config={'verbose': True, 'threshold': 0.5}", result[0])


if __name__ == "__main__":
    unittest.main()
