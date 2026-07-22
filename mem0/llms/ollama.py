import base64
import binascii
import json
from typing import Dict, List, Optional, Tuple, Union

try:
    from ollama import Client
except ImportError:
    raise ImportError("The 'ollama' library is required. Please install it using 'pip install ollama'.")

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.ollama import OllamaConfig
from mem0.llms.base import LLMBase
from mem0.memory.utils import extract_json

# Cap on a decoded data-URI image so a hostile/huge payload can't exhaust memory.
_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB


def _extract_ollama_image(image_url) -> str:
    """Turn an OpenAI-style ``image_url`` value into what Ollama's ``images``
    list accepts (raw base64 string, or a local file path — Ollama's ``Image``
    serializer reads/encodes either). Never emits an ``http`` URL: Ollama would
    reject it with an opaque error, so we reject it here with a clear one.

    Accepts either the OpenAI object ``{"url": ...}`` or a bare string.
    """
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not isinstance(url, str) or not url:
        raise ValueError("ollama vision: image_url has no usable 'url' string")

    if url.startswith("data:"):
        # data:[<mediatype>][;base64],<payload>  -> Ollama needs the bare base64,
        # NOT the "data:...;base64," prefix (b64decode of the prefix fails).
        header, _, payload = url.partition(",")
        if not _ or ";base64" not in header:
            raise ValueError("ollama vision: only base64 data URIs are supported")
        payload = payload.strip()
        if not payload:
            raise ValueError("ollama vision: empty base64 payload in data URI")
        try:
            decoded = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("ollama vision: malformed base64 in data URI")
        if len(decoded) > _MAX_IMAGE_BYTES:
            raise ValueError(
                f"ollama vision: image exceeds {_MAX_IMAGE_BYTES} bytes"
            )
        return payload

    if url.startswith(("http://", "https://")):
        raise ValueError(
            "ollama vision: http(s) image URLs are not supported — pass a base64 "
            "data URI or a local file path (download the image first)"
        )

    # Otherwise a local file path or a raw base64 string: Ollama's Image handles
    # both. A path is read from the local filesystem — treat it as trusted input.
    return url


def _ollama_messages(messages: List[Dict]) -> Tuple[List[Dict], bool]:
    """Rewrite OpenAI-style multimodal messages into Ollama's shape.

    OpenAI puts images inside a ``content`` list as ``{"type": "image_url", ...}``
    parts; Ollama wants a single string ``content`` plus a message-level
    ``images: [...]`` list. This joins every text part (in order) and collects
    every image part. Non-multimodal messages pass through untouched. The input
    is never mutated (messages are copied). Returns ``(messages, has_image)``.

    Note: the exact text/image interleaving cannot be preserved (Ollama has one
    content string + a flat images list); text order is kept, images are appended.
    """
    out: List[Dict] = []
    has_image = False
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        texts: List[str] = []
        images: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                texts.append(part.get("text", ""))
            elif ptype == "image_url":
                images.append(_extract_ollama_image(part.get("image_url")))
        new_msg = dict(msg)
        new_msg["content"] = " ".join(t for t in texts if t)
        if images:
            new_msg["images"] = images
            has_image = True
        out.append(new_msg)
    return out, has_image


class OllamaLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, OllamaConfig, Dict]] = None):
        # Convert to OllamaConfig if needed
        if config is None:
            config = OllamaConfig()
        elif isinstance(config, dict):
            config = OllamaConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, OllamaConfig):
            # Convert BaseLlmConfig to OllamaConfig
            config = OllamaConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                http_client_proxies=config.http_client_proxies,
            )

        super().__init__(config)

        if not self.config.model:
            self.config.model = "llama3.1:70b"

        self.client = Client(host=self.config.ollama_base_url)

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        # Get the content from response
        if isinstance(response, dict):
            content = response["message"]["content"]
        else:
            content = response.message.content

        if tools:
            processed_response = {
                "content": content,
                "tool_calls": [],
            }

            if isinstance(response, dict):
                raw_calls = response.get("message", {}).get("tool_calls") or []
            else:
                raw_calls = getattr(response.message, "tool_calls", None) or []

            for tool_call in raw_calls:
                if isinstance(tool_call, dict):
                    fn = tool_call.get("function", {})
                    name = fn.get("name", "")
                    arguments = fn.get("arguments", {})
                else:
                    fn = getattr(tool_call, "function", None)
                    name = getattr(fn, "name", "") if fn else ""
                    arguments = getattr(fn, "arguments", {}) if fn else {}

                if isinstance(arguments, str):
                    arguments = json.loads(extract_json(arguments))

                processed_response["tool_calls"].append(
                    {"name": name, "arguments": arguments}
                )

            return processed_response
        else:
            return content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a response based on the given messages using Ollama.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional Ollama-specific parameters.

        Returns:
            str: The generated response.
        """
        # Normalize OpenAI-style multimodal messages into Ollama's shape FIRST,
        # so image content becomes a message-level `images` list and `content`
        # is a plain string (the JSON-format block below assumes string content).
        messages, has_image = _ollama_messages(messages)

        # Route image-bearing calls to the vision model when configured, keeping
        # `model` as the text model (they often can't co-fit in GPU memory).
        vision_model = getattr(self.config, "vision_model", None)
        model = vision_model if (has_image and vision_model) else self.config.model

        # Build parameters for Ollama
        params = {
            "model": model,
            "messages": messages,
        }

        # Handle JSON response format by using Ollama's native format parameter
        if response_format and response_format.get("type") == "json_object":
            params["format"] = "json"
            messages = [dict(m) for m in messages]
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\nPlease respond with valid JSON only."
            else:
                messages.append({"role": "user", "content": "Please respond with valid JSON only."})
            params["messages"] = messages

        # Add options for Ollama (temperature, num_predict, top_p)
        options = {
            "temperature": self.config.temperature,
            "num_predict": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        params["options"] = options

        # Remove OpenAI-specific parameters that Ollama doesn't support
        params.pop("max_tokens", None)  # Ollama uses different parameter names

        if tools:
            params["tools"] = tools

        response = self.client.chat(**params)
        return self._parse_response(response, tools)
