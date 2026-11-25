"""Numeric faithfulness: every dollar figure in the narration must appear in a tool output.

Used at run time (the UI flags unverified figures) and by eval/run_eval.py.
"""
import re

DOLLAR = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k\b)?", re.I)


def dollar_figures(text):
    """Dollar amounts written in a text, as floats ('$1,300' -> 1300.0, '$2k' -> 2000.0)."""
    out = []
    for m in DOLLAR.finditer(text or ""):
        v = float(m.group(1).replace(",", ""))
        out.append(v * 1000 if m.group(2) else v)
    return out


def numbers_in(obj):
    """Every number anywhere inside a tool output (nested dicts and lists)."""
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in numbers_in(v)]
    if isinstance(obj, (list, tuple)):
        return [n for v in obj for n in numbers_in(v)]
    return []


def unverified_figures(text, tool_outputs, tolerance=1.0):
    """Dollar figures in `text` that match no number in the tool outputs (within $1 rounding).
    The tools return deltas and totals themselves, so only figures copied from an output pass."""
    nums = sorted(set(numbers_in(tool_outputs)))
    if not nums:
        return dollar_figures(text)
    bad = []
    for v in dollar_figures(text):
        if any(abs(v - n) <= tolerance or abs(v + n) <= tolerance for n in nums):
            continue
        bad.append(v)
    return bad
