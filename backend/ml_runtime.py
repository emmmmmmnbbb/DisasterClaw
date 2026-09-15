"""Process-wide guards for third-party model loaders with global torch state."""

from __future__ import annotations

import threading


# Transformers model construction temporarily changes torch's process-wide
# default dtype. Perception and Qwen warm up in different threads, so loaders
# and dtype-sensitive preprocessing must share one lock.
MODEL_LOAD_LOCK = threading.RLock()
