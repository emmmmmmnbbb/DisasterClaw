from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm_client  # noqa: E402
from local_qwen_vl import LocalQwenVLBackend  # noqa: E402


class _Inputs(dict):
    def __init__(self):
        super().__init__(input_ids=torch.tensor([[1, 2]], dtype=torch.long))
        self.input_ids = self["input_ids"]

    def to(self, device):
        return self


class _Processor:
    class _Tokenizer:
        eos_token_id = 0

    tokenizer = _Tokenizer()

    def apply_chat_template(self, *args, **kwargs):
        return "prompt"

    def __call__(self, *args, **kwargs):
        return _Inputs()

    def batch_decode(self, ids, skip_special_tokens=True):
        return [str(int(ids[0, 0]))]


class _Model:
    def generate(self, input_ids, **kwargs):
        sampled = torch.randint(3, 100000, (1, 1), dtype=torch.long)
        return torch.cat([input_ids, sampled], dim=1)


def test_local_qwen_seed_repeats_and_restores_global_rng() -> None:
    backend = LocalQwenVLBackend("fake", device="cpu")
    backend.model = _Model()
    backend.processor = _Processor()
    backend._torch = torch
    backend.device = "cpu"

    torch.manual_seed(9)
    state_before = torch.random.get_rng_state().clone()
    first = backend.infer([{"role": "user", "content": "x"}], seed=123)
    state_after = torch.random.get_rng_state().clone()
    second = backend.infer([{"role": "user", "content": "x"}], seed=123)
    third = backend.infer([{"role": "user", "content": "x"}], seed=124)

    assert first == second
    assert first != third
    assert torch.equal(state_before, state_after)


def test_llm_client_forwards_generation_seed(monkeypatch) -> None:
    captured = {}

    class _Backend:
        def infer(self, **kwargs):
            captured.update(kwargs)
            return "ok"

    monkeypatch.setattr(llm_client, "get_local_qwen_vl_backend", lambda cfg: _Backend())
    client = llm_client.LLMClient({
        "api_type": "local_qwen_vl", "base_url": "local://test",
        "api_key": "", "timeout": 0,
    }, "fake")
    assert client.chat([], temperature=0.1, max_tokens=12, seed=987) == "ok"
    assert captured["seed"] == 987
    assert captured["temperature"] == 0.1
    assert captured["max_new_tokens"] == 12
