from unittest.mock import Mock, patch

import pytest

from mem0.configs.llms.ollama import OllamaConfig
from mem0.llms.ollama import OllamaLLM


@pytest.fixture
def mock_ollama_client():
    with patch("mem0.llms.ollama.Client") as mock_ollama:
        mock_client = Mock()
        mock_client.list.return_value = {"models": [{"name": "llama3.1:70b"}]}
        mock_ollama.return_value = mock_client
        yield mock_client


def test_generate_response_without_tools(mock_ollama_client):
    config = OllamaConfig(model="llama3.1:70b", temperature=0.7, max_tokens=100, top_p=1.0)
    llm = OllamaLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    mock_response = {"message": {"content": "I'm doing well, thank you for asking!"}}
    mock_ollama_client.chat.return_value = mock_response

    response = llm.generate_response(messages)

    mock_ollama_client.chat.assert_called_once_with(
        model="llama3.1:70b", messages=messages, options={"temperature": 0.7, "num_predict": 100, "top_p": 1.0}
    )
    assert response == "I'm doing well, thank you for asking!"


def test_generate_response_with_tools_passes_tools_to_client(mock_ollama_client):
    """Tools should be forwarded to ollama client.chat()."""
    config = OllamaConfig(model="llama3.1:70b", temperature=0.1, max_tokens=100, top_p=1.0)
    llm = OllamaLLM(config)
    messages = [{"role": "user", "content": "Extract entities from: Alice works at UCSD"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_entities",
                "description": "Extract entities",
                "parameters": {"type": "object", "properties": {"entities": {"type": "array"}}},
            },
        }
    ]

    mock_response = {
        "message": {
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "extract_entities",
                        "arguments": {"entities": [{"name": "Alice"}, {"name": "UCSD"}]},
                    }
                }
            ],
        }
    }
    mock_ollama_client.chat.return_value = mock_response

    response = llm.generate_response(messages, tools=tools)

    # Verify tools were passed to client.chat
    call_kwargs = mock_ollama_client.chat.call_args
    assert "tools" in call_kwargs.kwargs or (len(call_kwargs.args) > 0 and "tools" in call_kwargs[1])
    assert call_kwargs[1]["tools"] == tools

    # Verify tool_calls were parsed correctly
    assert response["tool_calls"] == [
        {"name": "extract_entities", "arguments": {"entities": [{"name": "Alice"}, {"name": "UCSD"}]}}
    ]


def test_generate_response_with_tools_no_tool_calls_in_response(mock_ollama_client):
    """When model returns content without tool_calls, tool_calls should be empty list."""
    config = OllamaConfig(model="llama3.1:70b", temperature=0.1, max_tokens=100, top_p=1.0)
    llm = OllamaLLM(config)
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "noop", "parameters": {}}}]

    mock_response = {"message": {"content": "I cannot use tools for this.", "tool_calls": []}}
    mock_ollama_client.chat.return_value = mock_response

    response = llm.generate_response(messages, tools=tools)

    assert response["content"] == "I cannot use tools for this."
    assert response["tool_calls"] == []


def test_generate_response_with_tools_string_arguments(mock_ollama_client):
    """When tool_call arguments come as JSON string, they should be parsed."""
    config = OllamaConfig(model="llama3.1:70b", temperature=0.1, max_tokens=100, top_p=1.0)
    llm = OllamaLLM(config)
    messages = [{"role": "user", "content": "test"}]
    tools = [{"type": "function", "function": {"name": "test_fn", "parameters": {}}}]

    mock_response = {
        "message": {
            "content": "",
            "tool_calls": [
                {"function": {"name": "test_fn", "arguments": '{"key": "value"}'}}
            ],
        }
    }
    mock_ollama_client.chat.return_value = mock_response

    response = llm.generate_response(messages, tools=tools)

    assert response["tool_calls"] == [{"name": "test_fn", "arguments": {"key": "value"}}]


def test_parse_response_with_tools_object_style(mock_ollama_client):
    """Test _parse_response with object-style response (non-dict)."""
    config = OllamaConfig(model="llama3.1:70b")
    llm = OllamaLLM(config)

    # Simulate object-style response
    mock_fn = Mock()
    mock_fn.name = "extract"
    mock_fn.arguments = {"entities": ["Alice"]}

    mock_tool_call = Mock()
    mock_tool_call.function = mock_fn

    mock_message = Mock()
    mock_message.content = ""
    mock_message.tool_calls = [mock_tool_call]

    mock_response = Mock()
    mock_response.message = mock_message

    tools = [{"type": "function", "function": {"name": "extract"}}]
    result = llm._parse_response(mock_response, tools)

    assert result["tool_calls"] == [{"name": "extract", "arguments": {"entities": ["Alice"]}}]


# ---------------------------------------------------------------------------
# Vision: OpenAI image_url -> Ollama images translation + vision_model routing
# (added for the ollama-native vision fix). The pure normalizer is table-tested;
# generate_response is checked against a mocked Client for wire shape + routing.
# ---------------------------------------------------------------------------
import base64

from mem0.llms.ollama import _extract_ollama_image, _ollama_messages

_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\nHELLO").decode()
_DATA_URI = f"data:image/png;base64,{_PNG_B64}"


def _img_msg(url):
    return {"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": url}},
    ]}


def test_normalizer_data_uri_stripped_to_raw_base64():
    out, has = _ollama_messages([_img_msg(_DATA_URI)])
    assert has is True
    assert out[0]["content"] == "describe"
    # the "data:...;base64," prefix must be gone (Ollama b64decodes the value)
    assert out[0]["images"] == [_PNG_B64]


def test_normalizer_local_path_passthrough():
    out, has = _ollama_messages([_img_msg("/tmp/pic.png")])
    assert has is True
    assert out[0]["images"] == ["/tmp/pic.png"]


def test_normalizer_multiple_images_and_text_order():
    msg = {"role": "user", "content": [
        {"type": "text", "text": "a"},
        {"type": "image_url", "image_url": {"url": _PNG_B64}},
        {"type": "text", "text": "b"},
        {"type": "image_url", "image_url": {"url": "/tmp/x.png"}},
    ]}
    out, _ = _ollama_messages([msg])
    assert out[0]["content"] == "a b"            # text order preserved
    assert out[0]["images"] == [_PNG_B64, "/tmp/x.png"]  # every image kept


def test_normalizer_image_only_message():
    msg = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": _PNG_B64}}]}
    out, has = _ollama_messages([msg])
    assert has is True
    assert out[0]["content"] == ""
    assert out[0]["images"] == [_PNG_B64]


def test_normalizer_unknown_part_ignored():
    msg = {"role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "input_audio", "audio": "..."},  # unknown -> dropped, no crash
    ]}
    out, has = _ollama_messages([msg])
    assert has is False
    assert out[0]["content"] == "hi"
    assert "images" not in out[0]


def test_normalizer_plain_message_passthrough_and_immutability():
    original = [{"role": "user", "content": [{"type": "text", "text": "x"}]},
                {"role": "system", "content": "sys"}]
    out, has = _ollama_messages(original)
    # input not mutated
    assert isinstance(original[0]["content"], list)
    assert out[1] == {"role": "system", "content": "sys"}


def test_extract_image_rejects_http():
    for bad in ("http://h/x.png", "https://h/x.png"):
        with pytest.raises(ValueError, match="http"):
            _extract_ollama_image({"url": bad})


def test_extract_image_rejects_malformed_data_uri():
    for bad in ("data:image/png,notb64", "data:image/png;base64,@@@@", "data:x;base64,"):
        with pytest.raises(ValueError):
            _extract_ollama_image({"url": bad})


def test_generate_response_translates_image_to_ollama_images(mock_ollama_client):
    config = OllamaConfig(model="qwen-text", enable_vision=True)
    llm = OllamaLLM(config)
    mock_ollama_client.chat.return_value = {"message": {"content": "a red square"}}

    llm.generate_response(messages=[_img_msg(_DATA_URI)])

    _, kwargs = mock_ollama_client.chat.call_args
    sent = kwargs["messages"][0]
    assert sent["images"] == [_PNG_B64]          # image reached `images`
    assert sent["content"] == "describe"
    assert not isinstance(sent["content"], list)  # never an image_url content part
    assert kwargs["model"] == "qwen-text"         # no vision_model -> uses model


def test_generate_response_routes_to_vision_model_when_image_present(mock_ollama_client):
    config = OllamaConfig(model="qwen-text", vision_model="qwen3-vl", enable_vision=True)
    llm = OllamaLLM(config)
    mock_ollama_client.chat.return_value = {"message": {"content": "ok"}}

    llm.generate_response(messages=[_img_msg(_DATA_URI)])
    assert mock_ollama_client.chat.call_args.kwargs["model"] == "qwen3-vl"


def test_generate_response_text_only_ignores_vision_model(mock_ollama_client):
    config = OllamaConfig(model="qwen-text", vision_model="qwen3-vl")
    llm = OllamaLLM(config)
    mock_ollama_client.chat.return_value = {"message": {"content": "ok"}}

    llm.generate_response(messages=[{"role": "user", "content": "no image here"}])
    # no image -> vision_model must NOT hijack the call
    assert mock_ollama_client.chat.call_args.kwargs["model"] == "qwen-text"


def test_vision_model_config_treats_empty_string_as_unset():
    assert OllamaConfig(model="m", vision_model="").vision_model is None
    assert OllamaConfig(model="m").vision_model is None
    assert OllamaConfig(model="m", vision_model="vlm").vision_model == "vlm"


def test_generate_response_image_plus_json_format(mock_ollama_client):
    """Regression: the json_object block appends to content; it must run AFTER
    multimodal normalization (else content is a list and += raises TypeError)."""
    config = OllamaConfig(model="qwen-text", vision_model="qwen3-vl")
    llm = OllamaLLM(config)
    mock_ollama_client.chat.return_value = {"message": {"content": "{}"}}

    llm.generate_response(messages=[_img_msg(_DATA_URI)],
                          response_format={"type": "json_object"})

    kwargs = mock_ollama_client.chat.call_args.kwargs
    assert kwargs["format"] == "json"
    assert kwargs["model"] == "qwen3-vl"
    last = kwargs["messages"][-1]
    assert "Please respond with valid JSON only." in last["content"]
    assert last["images"] == [_PNG_B64]


def test_generate_response_text_only_unchanged_regression(mock_ollama_client):
    """A plain text call must produce the exact same client.chat args as before
    the vision change (byte-for-byte guard)."""
    config = OllamaConfig(model="llama3.1:70b", temperature=0.7, max_tokens=100, top_p=1.0)
    llm = OllamaLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]
    mock_ollama_client.chat.return_value = {"message": {"content": "hi"}}
    llm.generate_response(messages)
    mock_ollama_client.chat.assert_called_once_with(
        model="llama3.1:70b", messages=messages,
        options={"temperature": 0.7, "num_predict": 100, "top_p": 1.0},
    )
