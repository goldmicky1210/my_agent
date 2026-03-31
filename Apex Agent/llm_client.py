"""LLM API client with retry logic and cost tracking."""
from __future__ import annotations
import os
import logging
import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)

# Cost per million tokens (gpt-4o-mini pricing)
_PROMPT_COST_PER_M = 0.15
_COMPLETION_COST_PER_M = 0.60


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503)
    return isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout))


class LLMClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", LLM_MODEL)
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", str(LLM_TEMPERATURE)))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", str(LLM_MAX_TOKENS)))
        self._client = httpx.Client(timeout=25.0)
        self._total_cost = 0.0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3.0),
        retry=retry_if_exception(_is_retryable),
    )
    def chat(self, task_id: str, messages: list[dict]) -> str:
        headers = {"Content-Type": "application/json", "IWA-Task-ID": task_id}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        resp = self._client.post(
            f"{self.base_url}/chat/completions", json=body, headers=headers
        )
        resp.raise_for_status()
        data = resp.json()

        # Cost tracking
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost = (prompt_tokens * _PROMPT_COST_PER_M + completion_tokens * _COMPLETION_COST_PER_M) / 1_000_000
        self._total_cost += cost
        logger.debug(
            f"LLM call cost=${cost:.6f} total=${self._total_cost:.6f} "
            f"prompt={prompt_tokens} completion={completion_tokens}"
        )

        return data["choices"][0]["message"]["content"]

    @property
    def total_cost(self) -> float:
        return self._total_cost
