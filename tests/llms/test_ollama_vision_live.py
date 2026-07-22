"""Live smoke: prove that an image actually reaches a real Ollama VLM through
mem0's native vision path (parse_vision_messages -> get_image_description ->
OllamaLLM.generate_response with the image_url->images translation).

A mocked Client can only prove we emit the structure we *expect*; it cannot
prove Ollama consumes it. This runs the whole path against a real local Ollama
VLM and asserts the model describes the image content. If the translation were
wrong (image never reaches the model), the description would not mention the
color.

GATED: skipped unless a local Ollama is reachable AND the vision model is
installed AND Pillow is importable. Set MEM0_LIVE_OLLAMA=0 to force-skip.
Point at a different model with MEM0_LIVE_VLM (default qwen3-vl:4b-instruct).
"""
import base64
import io
import os

import pytest

VLM = os.environ.get("MEM0_LIVE_VLM", "qwen3-vl:4b-instruct")
OLLAMA_URL = os.environ.get("MEM0_LLM_URL", "http://localhost:11434")


def _skip_reason():
    if os.environ.get("MEM0_LIVE_OLLAMA") == "0":
        return "MEM0_LIVE_OLLAMA=0 (force-skip)"
    try:
        from ollama import Client
    except ImportError:
        return "ollama client not installed"
    try:
        import PIL  # noqa: F401
    except ImportError:
        return "Pillow not installed (needed to synthesize the test image)"
    try:
        names = [m.get("model") or m.get("name") for m in Client(host=OLLAMA_URL).list().get("models", [])]
    except Exception as e:
        return f"ollama unreachable at {OLLAMA_URL}: {e}"
    if not any(VLM in (n or "") for n in names):
        return f"vision model {VLM} not installed"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")


def _red_square_data_uri() -> str:
    from PIL import Image

    img = Image.new("RGB", (128, 128), (220, 20, 20))  # unambiguous red
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_real_vlm_describes_image_through_native_path():
    from mem0.configs.llms.ollama import OllamaConfig
    from mem0.llms.ollama import OllamaLLM
    from mem0.memory.utils import get_image_description

    # `model` is the text model and is unused here (the image call routes to
    # vision_model); any name works. vision_model carries the actual VLM.
    llm = OllamaLLM(OllamaConfig(model="llama3.1", vision_model=VLM,
                                 ollama_base_url=OLLAMA_URL, max_tokens=256))
    # get_image_description is exactly what parse_vision_messages calls per image.
    description = get_image_description(_red_square_data_uri(), llm, "auto")

    assert description and description.strip(), "VLM returned an empty description"
    low = description.lower()
    # if the image bytes reached the model, it names the color; a broken
    # image_url->images translation would yield an error or a color-less/generic
    # answer. Accept EN or PT.
    assert any(w in low for w in ("red", "vermelh")), (
        f"description did not mention the red color -> image likely did not reach "
        f"the model. Got: {description!r}"
    )
