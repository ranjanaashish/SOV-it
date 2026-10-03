"""Open-source LLM client.

Talks to any OpenAI-compatible endpoint: Ollama (default, local), vLLM,
llama.cpp server, LocalAI or TGI. No vendor SDK, no hard-coded keys.

    LLM_BASE_URL     default http://localhost:11434/v1   (Ollama)
    LLM_MODEL        default qwen2.5:7b-instruct          (any instruct model)
    LLM_API_KEY      optional; read from env only
    LLM_DISABLED     set to 1 to force deterministic-only mode
    LLM_TIMEOUT      seconds per request, default 45
    LLM_MAX_WORKERS  parallel requests for batched work, default 4
    LLM_KEEP_ALIVE   Ollama only: keep the model loaded in memory, default 30m

Speed features: one pooled HTTP session (keep-alive), an in-memory response
cache keyed on the exact prompt, parallel fan-out for batched calls
(`map_json`), Ollama keep_alive so the model is not reloaded between calls,
a one-off warm-up, and a circuit breaker that stops calling a dead endpoint.

The LLM only *proposes*. Every response is parsed as JSON and validated by the
calling agent; if the endpoint is down the pipeline still runs on rules.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests
from requests.adapters import HTTPAdapter

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.timeout = float(os.getenv("LLM_TIMEOUT", "45"))
        self.max_workers = max(1, int(os.getenv("LLM_MAX_WORKERS", "4")))
        self.keep_alive = os.getenv("LLM_KEEP_ALIVE", "30m")
        self.disabled = os.getenv("LLM_DISABLED", "0") == "1"
        self.is_ollama = ":11434" in self.base_url or "ollama" in self.base_url
        self._available: Optional[bool] = None
        self._checked_at = 0.0
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_size = 512
        self._consecutive_failures = 0
        self._warmed = False
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.total_seconds = 0.0

        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=self.max_workers, pool_maxsize=self.max_workers * 2)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._session.headers.update(self._headers())

    @property
    def label(self) -> str:
        return f"{self.model} @ {self.base_url}"

    def stats(self) -> dict:
        avg = self.total_seconds / self.calls if self.calls else 0.0
        return {"calls": self.calls, "cache_hits": self.cache_hits, "failures": self.failures,
                "avg_seconds": round(avg, 2)}

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def configure(self, base_url: str | None = None, model: str | None = None,
                  api_key: str | None = None, disabled: bool | None = None) -> None:
        if base_url is not None:
            self.base_url = base_url.rstrip("/")
        if model is not None:
            self.model = model.strip()
        if api_key is not None:
            self.api_key = api_key.strip()
        if disabled is not None:
            self.disabled = disabled
        self.is_ollama = ":11434" in self.base_url or "ollama" in self.base_url
        self._session.headers.update(self._headers())
        if not self.api_key and "Authorization" in self._session.headers:
            del self._session.headers["Authorization"]
        self._available = None
        self._checked_at = 0.0
        self._consecutive_failures = 0
        self._warmed = False

    def available(self) -> bool:
        if self.disabled:
            return False
        # circuit breaker: after 3 straight failures, back off for 60 s
        if self._consecutive_failures >= 3 and time.time() - self._checked_at < 60:
            return False
        if self._available is not None and time.time() - self._checked_at < 60:
            return self._available
        try:
            r = self._session.get(f"{self.base_url}/models", timeout=4)
            if r.status_code == 200:
                self._available = True
            elif r.status_code in (404, 405):  # Some gateways don't implement /models
                self._available = bool(self.base_url)
            else:
                self._available = False
        except requests.RequestException:
            self._available = False
        self._checked_at = time.time()
        if self._available:
            self._consecutive_failures = 0
            self._warm_up()
        return self._available

    def _warm_up(self) -> None:
        """Load the model into memory once so the first real call is not slow."""
        if self._warmed or not self.is_ollama:
            return
        self._warmed = True

        def _go():
            try:
                self._session.post(self.base_url.replace("/v1", "") + "/api/generate",
                                   json={"model": self.model, "prompt": "", "keep_alive": self.keep_alive},
                                   timeout=self.timeout)
            except requests.RequestException:
                pass

        threading.Thread(target=_go, daemon=True).start()

    def chat_json(self, system: str, user: str, max_tokens: int = 1200) -> Optional[dict]:
        """Return a parsed JSON object, or None on any failure (caller falls back)."""
        if not self.available():
            return None
        key = hashlib.sha1(f"{self.model}\x00{system}\x00{user}\x00{max_tokens}".encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.cache_hits += 1
                return self._cache[key]
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system + "\nRespond with a single JSON object only. Be concise."},
                {"role": "user", "content": user},
            ],
        }
        if self.is_ollama:
            payload["keep_alive"] = self.keep_alive
        t0 = time.time()
        with self._lock:
            self.calls += 1
        try:
            r = self._session.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            obj = _parse_json(text)
        except Exception as exc:  # network, HTTP, schema, JSON
            obj = None
            log.warning("LLM call failed: %s", exc)
        with self._lock:
            self.total_seconds += time.time() - t0
            if obj is None:
                self.failures += 1
                self._consecutive_failures += 1
                self._checked_at = time.time()
            else:
                self._consecutive_failures = 0
                self._cache[key] = obj
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return obj

    def map_json(self, requests_: list[tuple[str, str]], max_tokens: int = 1200) -> list[Optional[dict]]:
        """Run several (system, user) prompts in parallel; results keep input order."""
        if not requests_ or not self.available():
            return [None] * len(requests_)
        if len(requests_) == 1:
            return [self.chat_json(*requests_[0], max_tokens=max_tokens)]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(requests_))) as pool:
            return list(pool.map(lambda sp: self.chat_json(sp[0], sp[1], max_tokens=max_tokens), requests_))


def _parse_json(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None
