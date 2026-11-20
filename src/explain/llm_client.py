"""LLM provider behind one interface, configured from .env:

  LLM_PROVIDER = template | ollama | anthropic | openai     (empty -> template)
  LLM_MODEL    = model name (defaults per provider below)
  LLM_API_KEY  = key for anthropic / openai
  LLM_BASE_URL = optional endpoint (ollama default http://localhost:11434;
                 openai: any OpenAI-compatible base URL, e.g. https://api.openai.com/v1)

`complete()` returns None on any failure or refusal; callers then fall back to the
deterministic template, so the pipeline never depends on an API being reachable.
"""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)

DEFAULT_MODELS = {"ollama": "llama3.2:3b", "anthropic": "claude-opus-5-5", "openai": "gpt-4o-mini"}


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 api_key: str | None = None, base_url: str | None = None, timeout: float = 60.0):
        self.provider = (provider if provider is not None else os.getenv("LLM_PROVIDER", "")).strip().lower() or "template"
        self.model = (model or os.getenv("LLM_MODEL", "")).strip() or DEFAULT_MODELS.get(self.provider, "template")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self.timeout = timeout
        self._sdk = None
        if self.provider not in ("template", "ollama", "anthropic", "openai"):
            raise ValueError(f"unknown LLM_PROVIDER {self.provider!r}")

    @property
    def name(self) -> str:
        return "template" if self.provider == "template" else f"{self.provider}:{self.model}"

    @property
    def is_template(self) -> bool:
        return self.provider == "template"

    def complete(self, system: str, user: str) -> str | None:
        if self.is_template:
            return None
        try:
            return getattr(self, f"_{self.provider}")(system, user)
        except Exception as e:  # network, auth, quota, refusal ... -> template fallback
            log.warning("LLM call failed (%s): %s", self.name, str(e)[:200])
            return None

    # -- providers -------------------------------------------------------------
    def _ollama(self, system: str, user: str) -> str | None:
        url = (self.base_url or "http://localhost:11434") + "/api/chat"
        r = requests.post(url, timeout=self.timeout, json={
            "model": self.model, "stream": False, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def _openai(self, system: str, user: str) -> str | None:
        url = (self.base_url or "https://api.openai.com/v1") + "/chat/completions"
        r = requests.post(url, timeout=self.timeout, headers={"Authorization": f"Bearer {self.api_key}"}, json={
            "model": self.model, "temperature": 0, "max_tokens": 200,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def _anthropic_client(self):
        if self._sdk is None:
            import anthropic  # optional dependency: pip install anthropic

            kw = {"api_key": self.api_key} if self.api_key else {}
            if self.base_url:
                kw["base_url"] = self.base_url
            self._sdk = anthropic.Anthropic(timeout=self.timeout, **kw)
        return self._sdk

    def _anthropic(self, system: str, user: str) -> str | None:
        client = self._anthropic_client()
        # Server-side refusal fallback is on by default for current Claude models.
        resp = client.beta.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low"},          # short phrasing task
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            return None
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text or None
