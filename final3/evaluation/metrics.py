from __future__ import annotations


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num) / float(den) if den else default
