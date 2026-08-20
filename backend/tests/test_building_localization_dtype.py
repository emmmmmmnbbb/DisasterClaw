"""Regression tests for deterministic building-localizer dtype at startup."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

import building_localization as localization  # noqa: E402


class _TinyLocalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def test_loader_forces_fp32_during_external_bfloat16_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        state = _TinyLocalizer().state_dict()
        checkpoint = tmp_path / "localizer.pt"
        torch.save({"model_state": state, "imgsz": 4}, checkpoint)
        monkeypatch.setattr(
            localization, "ResNet34UNet", lambda pretrained=False: _TinyLocalizer(),
        )

        # Reproduce the short global-dtype window created by a concurrent
        # Transformers model load.
        torch.set_default_dtype(torch.bfloat16)
        localizer = localization.load_building_localizer(checkpoint, device="cpu")
    finally:
        torch.set_default_dtype(original_dtype)

    assert next(localizer.model.parameters()).dtype == torch.float32
