import json
import logging
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _client = OpenAI(api_key=api_key)
    return _client


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def chat_json(system_prompt: str, user_content: str, model: str | None = None,
              temperature: float = 0.2, max_tokens: int = 4000) -> dict[str, Any]:
    """Single chat call with JSON-object response. Raises on parse failure."""
    client = _get_client()
    chosen = model or os.getenv("OPENAI_MODEL_FAST", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=chosen,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)
