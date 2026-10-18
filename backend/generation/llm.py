"""Gemini client: one prompt in, one JSON text out, cached on disk.

Why a cache: re-running an experiment must not pay twice for the same call, and the raw
responses stay available for debugging and manual checks. The key is a hash of model,
output budget and prompt; the API key is never written anywhere.

The API key is read from the GEMINI_API_KEY environment variable or from a local `.env`
file (git-ignored). Only rate-limit / server errors are retried; a timeout or a bad answer
returns `error` and the caller falls back (see rerank.py, explain.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request

from pathlib import Path
from typing import Any


CHEAP_MODEL = "gemini-3.5-flash-lite"      # generation while developing and for the main runs
JUDGE_MODEL = "gemini-3.5-flash"           # a stronger model grades the explanations
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")

    if not key and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "GEMINI_API_KEY":
                key = value.strip()

    if not key:
        raise RuntimeError("Set GEMINI_API_KEY (environment or .env file)")

    return key


class Gemini:
    def __init__(
        self,
        model: str = CHEAP_MODEL,
        cache_path: Path | None = None,
        timeout: float = 20.0,
        retries: int = 3,
        thinking_level: str | None = None,
    ) -> None:
        """thinking_level ("minimal", "low", ...): Gemini 3 Flash "thinks" by default, and thinking
        tokens use the same output budget, which can cut the JSON answer in the middle."""
        self.model, self.cache_path = model, cache_path
        self.timeout, self.retries, self.thinking_level = timeout, retries, thinking_level
        self.cache: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()          # experiments call generate() from several threads

        if cache_path and cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                self.cache[record["key"]] = record

    def generate(self, prompt: str, max_output_tokens: int = 512, schema: dict[str, Any] | None = None) -> dict[str, Any]:
        """{"text", "input_tokens", "output_tokens", "latency_ms", "cache_hit", "error"}

        `schema` (Gemini responseSchema) makes the provider return exactly that JSON shape.
        """
        settings = f"{self.model}\n{self.thinking_level}\n{max_output_tokens}\n{json.dumps(schema, sort_keys=True)}"
        key = hashlib.sha256(f"{settings}\n{prompt}".encode("utf-8")).hexdigest()

        if key in self.cache:
            return {**self.cache[key], "cache_hit": True, "error": None}

        config: dict[str, Any] = {"responseMimeType": "application/json", "temperature": 0.0, "maxOutputTokens": max_output_tokens}
        if schema:
            config["responseSchema"] = schema
        if self.thinking_level:
            config["thinkingConfig"] = {"thinkingLevel": self.thinking_level}

        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}
        started = time.perf_counter()

        for attempt in range(self.retries + 1):
            try:
                response = self._post(body)
                break
            except urllib.error.HTTPError as error:
                # Rate limit or a busy server: wait and try again; anything else is a failed call.
                if error.code in (429, 500, 503) and attempt < self.retries:
                    time.sleep(5 * 2**attempt)
                    continue

                return failed(f"http_{error.code}")
            except (urllib.error.URLError, TimeoutError) as error:
                return failed(type(error).__name__)

        candidate = (response.get("candidates") or [{}])[0]
        parts = candidate.get("content", {}).get("parts") or []
        usage = response.get("usageMetadata", {})

        # An answer cut by the output budget is not valid JSON: report it, never cache it.
        if candidate.get("finishReason") == "MAX_TOKENS":
            return failed("max_tokens")

        record = {
            "key": key,
            "model": self.model,
            "text": "".join(part.get("text", "") for part in parts) or None,
            "input_tokens": usage.get("promptTokenCount"),
            "output_tokens": (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }

        if record["text"] is None:
            return {**record, "cache_hit": False, "error": "empty_response"}

        with self.lock:
            self.cache[key] = record

            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

        return {**record, "cache_hit": False, "error": None}

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            URL.format(model=self.model),
            data=json.dumps(body).encode("utf-8"),
            headers={"x-goog-api-key": api_key(), "Content-Type": "application/json"},
        )

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)


def failed(error: str) -> dict[str, Any]:
    return {"text": None, "input_tokens": None, "output_tokens": None, "latency_ms": None, "cache_hit": False, "error": error}
