from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


PROVIDERS: dict[str, dict] = {
    "ollama_local": {
        "api_type": "openai_compat",
        "base_url": _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
        "api_key": "ollama-local",
        "default_model": _env("OLLAMA_MODEL", "qwen2.5:7b"),
        "timeout": 300,
    },
    "qwen_vl_local": {
        "api_type": "local_qwen_vl",
        "base_url": "local://qwen_vl_local",
        "api_key": "",
        "model_id": _env("BASE_MODEL", _env("VLM_LOCAL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")),
        "checkpoint": _env("CHECKPOINT_PATH", _env("VLM_LOCAL_CHECKPOINT", "")),
        "device": _env("VLM_LOCAL_DEVICE", "auto"),
        "torch_dtype": _env("VLM_LOCAL_TORCH_DTYPE", "auto"),
        "min_free_gpu_gb": float(_env("VLM_LOCAL_MIN_FREE_GPU_GB", _env("MIN_FREE_GPU_GB", "12")) or "12"),
        "top_p": float(_env("VLM_LOCAL_TOP_P", "0.9") or "0.9"),
        "repetition_penalty": float(_env("VLM_LOCAL_REPETITION_PENALTY", "1.1") or "1.1"),
        "default_model": _env("BASE_MODEL", _env("VLM_LOCAL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")),
        "timeout": 0,
    },
    "vlm": {
        "api_type": "openai_compat",
        "base_url": _env(
            "VLM_BASE_URL",
            _env("OLLAMA_BASE_URL", _env("LLM_BASE_URL", "http://127.0.0.1:11434/v1")),
        ),
        "api_key": _env("VLM_API_KEY", "ollama"),
        "default_model": _env("VLM_MODEL", "qwen2.5vl:7b"),
        "timeout": int(_env("VLM_TIMEOUT", "180") or "180"),
        "image_input_mode": _env("VLM_IMAGE_INPUT_MODE", "auto"),
    },
    "openai": {
        "api_type": "openai_compat",
        "base_url": _env("LLM_BASE_URL", "https://api.openai.com/v1"),
        "api_key": _env("LLM_API_KEY", ""),
        "default_model": _env("LLM_MODEL", "gpt-4o-mini"),
        "timeout": 150,
    },
    "deepseek": {
        "api_type": "openai_compat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": _env("DEEPSEEK_API_KEY", ""),
        "default_model": "deepseek-chat",
        "timeout": 60,
    },
    "moonshot": {
        "api_type": "openai_compat",
        "base_url": "https://api.moonshot.cn/v1",
        "api_key": _env("MOONSHOT_API_KEY", ""),
        "default_model": "moonshot-v1-8k",
        "timeout": 60,
    },
    "zhipu": {
        "api_type": "openai_compat",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": _env("ZHIPU_API_KEY", ""),
        "default_model": "glm-4",
        "timeout": 60,
    },
}

ACTIVE_PROVIDER: str = _env("ACTIVE_PROVIDER", "openai")

MODULE_CONFIG: dict[str, dict] = {
    "planner": {
        "provider": _env("PLANNER_LLM_PROVIDER", "") or None,
        "model": _env("PLANNER_LLM_MODEL", "") or None,
    },
    "vlm": {
        "provider": _env("VLM_PROVIDER", "vlm") or None,
        "model": _env("VLM_MODEL", "") or None,
    },
}
