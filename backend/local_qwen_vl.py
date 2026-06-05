from __future__ import annotations

import base64
import gc
import io
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any


_BACKEND_LOCK = threading.Lock()
_BACKEND_CACHE: dict[tuple, "LocalQwenVLBackend"] = {}


def _gpu_info() -> list[tuple[int, int, int, int, float]]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except Exception:
        return []

    gpus = []
    for line in output.strip().splitlines():
        if not line:
            continue
        try:
            idx, free, total, used = map(int, line.split(", "))
        except ValueError:
            continue
        gpus.append((idx, free, total, used, (free / total) if total else 0.0))
    return gpus


def _best_gpu(min_free_gb: float) -> int:
    candidates = [gpu for gpu in _gpu_info() if gpu[1] / 1024 >= min_free_gb]
    if not candidates:
        return 0
    return sorted(candidates, key=lambda item: item[4], reverse=True)[0][0]


def _clean_text(text: str) -> str:
    text = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|im_start\|>|<\|im_end\|>", "", text)
    text = re.sub(r"(!{3,}|\.{5,}|-{5,})", "", text)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    return text.replace("\ufffd", "").strip()


class LocalQwenVLBackend:
    def __init__(
        self,
        model_id: str,
        checkpoint: str = "",
        device: str = "auto",
        torch_dtype: str = "auto",
        min_free_gpu_gb: float = 12.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ):
        self.model_id = model_id
        self.checkpoint = checkpoint
        self.device_spec = device
        self.torch_dtype = torch_dtype
        self.min_free_gpu_gb = float(min_free_gpu_gb)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)

        self.model = None
        self.processor = None
        self.device = "cpu"
        self._torch = None
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self) -> None:
        if self.is_loaded:
            return

        with self._load_lock:
            if self.is_loaded:
                return

            try:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "本地 qwen_vl_local 后端需要安装 transformers、torch、accelerate、pillow"
                ) from exc

            self._torch = torch
            self.device = self._resolve_device(torch)
            dtype = self._resolve_dtype(torch)

            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

            load_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            if dtype is not None:
                load_kwargs["torch_dtype"] = dtype

            if self.device.startswith("cuda"):
                load_kwargs["device_map"] = {"": self.device}

            model = AutoModelForImageTextToText.from_pretrained(self.model_id, **load_kwargs)
            if not self.device.startswith("cuda"):
                model = model.to(self.device)

            if self.checkpoint and Path(self.checkpoint).exists():
                try:
                    from peft import PeftModel
                except ImportError as exc:
                    raise RuntimeError("加载 LoRA checkpoint 需要安装 peft") from exc
                model = PeftModel.from_pretrained(model, self.checkpoint).merge_and_unload()

            self.model = model.eval()

    def unload(self) -> None:
        if self.model is None:
            return
        del self.model, self.processor
        self.model = None
        self.processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        gc.collect()

    def infer(self, messages: list[dict], max_new_tokens: int = 512, temperature: float = 0.3) -> str:
        if not self.is_loaded:
            self.load()
        assert self.processor is not None and self.model is not None and self._torch is not None

        normalized_messages, images = _normalize_messages(messages)
        prompt_text = self.processor.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[prompt_text],
            images=images or None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        new_ids = output_ids[:, inputs.input_ids.shape[1]:]
        text = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        return _clean_text(text)

    def _resolve_device(self, torch_module) -> str:
        spec = (self.device_spec or "auto").strip().lower()
        if spec == "cpu":
            return "cpu"
        if spec.startswith("cuda:"):
            return spec
        if spec in {"cuda", "gpu"}:
            return "cuda:0" if torch_module.cuda.is_available() else "cpu"
        if spec == "auto":
            if torch_module.cuda.is_available():
                return f"cuda:{_best_gpu(self.min_free_gpu_gb)}"
            return "cpu"
        return self.device_spec

    def _resolve_dtype(self, torch_module):
        if self.device == "cpu":
            return torch_module.float32
        spec = (self.torch_dtype or "auto").strip().lower()
        if spec == "float16":
            return torch_module.float16
        if spec == "float32":
            return torch_module.float32
        if spec == "bfloat16":
            return torch_module.bfloat16
        return torch_module.bfloat16


def get_local_qwen_vl_backend(provider_cfg: dict) -> LocalQwenVLBackend:
    key = (
        provider_cfg.get("model_id", ""),
        provider_cfg.get("checkpoint", ""),
        provider_cfg.get("device", "auto"),
        provider_cfg.get("torch_dtype", "auto"),
        float(provider_cfg.get("min_free_gpu_gb", 12.0) or 12.0),
        float(provider_cfg.get("top_p", 0.9) or 0.9),
        float(provider_cfg.get("repetition_penalty", 1.1) or 1.1),
    )
    with _BACKEND_LOCK:
        backend = _BACKEND_CACHE.get(key)
        if backend is None:
            backend = LocalQwenVLBackend(
                model_id=provider_cfg.get("model_id", ""),
                checkpoint=provider_cfg.get("checkpoint", ""),
                device=provider_cfg.get("device", "auto"),
                torch_dtype=provider_cfg.get("torch_dtype", "auto"),
                min_free_gpu_gb=float(provider_cfg.get("min_free_gpu_gb", 12.0) or 12.0),
                top_p=float(provider_cfg.get("top_p", 0.9) or 0.9),
                repetition_penalty=float(provider_cfg.get("repetition_penalty", 1.1) or 1.1),
            )
            _BACKEND_CACHE[key] = backend
    return backend


def _normalize_messages(messages: list[dict]) -> tuple[list[dict], list]:
    normalized: list[dict] = []
    images = []

    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")

        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            normalized.append({"role": role, "content": str(content)})
            continue

        new_content = []
        for item in content:
            item_type = item.get("type")
            if item_type == "text":
                new_content.append({"type": "text", "text": str(item.get("text", ""))})
                continue
            if item_type in {"image", "image_url"}:
                pil_image = _load_image_from_message(item)
                if pil_image is None:
                    continue
                images.append(pil_image)
                new_content.append({"type": "image", "image": pil_image})

        normalized.append({"role": role, "content": new_content or ""})

    return normalized, images


def _load_image_from_message(item: dict):
    from PIL import Image

    if item.get("type") == "image":
        raw = item.get("image")
    else:
        raw = item.get("image_url")
        if isinstance(raw, dict):
            raw = raw.get("url", "")

    if raw is None:
        return None

    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, bytes):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(raw, str):
        if raw.startswith("data:image/"):
            return _load_data_url_image(raw)
        if raw.startswith("http://") or raw.startswith("https://"):
            raise RuntimeError("本地 qwen_vl_local 后端不支持直接拉取远程图片 URL")
        if os.path.exists(raw):
            return Image.open(raw).convert("RGB")
    raise RuntimeError(f"无法解析图片输入: {type(raw)}")


def _load_data_url_image(data_url: str):
    from PIL import Image

    header, _, payload = data_url.partition(",")
    if ";base64" not in header or not payload:
        raise RuntimeError("不支持的 data URL 图片格式")
    image_bytes = base64.b64decode(payload)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
