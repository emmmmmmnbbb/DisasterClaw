from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

import config as _cfg
from local_qwen_vl import get_local_qwen_vl_backend


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    return text.strip()


class LLMClient:
    def __init__(self, provider_cfg: dict, model: str):
        self._provider_cfg = dict(provider_cfg)
        self._api_type = provider_cfg["api_type"]
        self._base_url = provider_cfg["base_url"].rstrip("/")
        self._api_key = provider_cfg["api_key"]
        self._model = model
        self._timeout = provider_cfg.get("timeout", 60)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_url(self) -> str:
        return self._base_url

    def provider_option(self, key: str, default=None):
        return self._provider_cfg.get(key, default)

    def chat(self, messages: list[dict], temperature: float = 0.3, max_tokens: int | None = None) -> str:
        if self._api_type == "openai_compat":
            return self._chat_openai_compat(messages, temperature, max_tokens)
        if self._api_type == "local_qwen_vl":
            return self._chat_local_qwen_vl(messages, temperature, max_tokens)
        raise NotImplementedError(f"api_type '{self._api_type}' 暂不支持")

    def _chat_openai_compat(self, messages: list[dict], temperature: float, max_tokens: int | None) -> str:
        url = f"{self._base_url}/chat/completions"
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            text = body["choices"][0]["message"]["content"]
            return _strip_thinking(text)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"[LLMClient] HTTP {exc.code} from {url}\n{body[:400]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"[LLMClient] 无法连接到 {url}：{exc.reason}") from exc

    def _chat_local_qwen_vl(self, messages: list[dict], temperature: float, max_tokens: int | None) -> str:
        backend = get_local_qwen_vl_backend(self._provider_cfg)
        return backend.infer(
            messages=messages,
            max_new_tokens=max_tokens or 512,
            temperature=temperature,
        )


def get_client(module: str | None = None, provider: str | None = None, model: str | None = None) -> LLMClient:
    resolved_provider = provider or _resolve_module_field(module, "provider") or _cfg.ACTIVE_PROVIDER
    if resolved_provider not in _cfg.PROVIDERS:
        raise ValueError(f"[LLMClient] 未知厂商 '{resolved_provider}'，可选值：{list(_cfg.PROVIDERS.keys())}")
    provider_cfg = dict(_cfg.PROVIDERS[resolved_provider])
    if provider_cfg.get("api_type") == "local_qwen_vl":
        resolved_model = model or _resolve_module_field(module, "model") or provider_cfg.get("model_id") or provider_cfg["default_model"]
        provider_cfg["model_id"] = resolved_model
        provider_cfg["default_model"] = resolved_model
    else:
        resolved_model = model or _resolve_module_field(module, "model") or provider_cfg["default_model"]
    return LLMClient(provider_cfg, resolved_model)


def _resolve_module_field(module: str | None, field: str) -> str | None:
    if module is None:
        return None
    return _cfg.MODULE_CONFIG.get(module, {}).get(field)
