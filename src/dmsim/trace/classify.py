from __future__ import annotations

import re

from dmsim.trace.schema import TensorCategory

_WEIGHT = re.compile(r"weight|w_|\.w\b|kernel", re.I)
_KV = re.compile(r"kv|cache|k_proj|v_proj|key|value", re.I)
_HIDDEN = re.compile(r"hidden|h_|intermediate|mlp", re.I)
_ACT = re.compile(r"act|activation|relu|gelu|silu", re.I)


def classify_tensor(name: str) -> TensorCategory:
    if _KV.search(name):
        return TensorCategory.KV_CACHE
    if _WEIGHT.search(name):
        return TensorCategory.WEIGHT
    if _HIDDEN.search(name):
        return TensorCategory.HIDDEN
    if _ACT.search(name):
        return TensorCategory.ACTIVATION
    return TensorCategory.OTHER
