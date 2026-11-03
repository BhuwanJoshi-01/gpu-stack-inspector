"""Escaping holds for any collected text."""

from __future__ import annotations

import re

from hypothesis import given
from hypothesis import strategies as st

from gpu_stack_inspector.reports.markdown import code, text

collected = st.text(st.characters(blacklist_categories=("Cs",)), max_size=60)


def unescape(value: str) -> str:
    return re.sub(r"\\(.)", r"\1", value)


@given(collected)
def test_text_round_trips_and_leaves_nothing_unescaped(value: str) -> None:
    escaped = text(value)
    assert unescape(escaped) == " ".join(value.split())
    without_escapes = re.sub(r"\\.", "", escaped)
    assert not set(without_escapes) & set("\\`*_[]<>|#")
    assert "\n" not in escaped


@given(collected)
def test_code_spans_keep_the_value_and_outgrow_its_backticks(value: str) -> None:
    span = code(value)
    flat = " ".join(value.split())
    fence = re.match(r"`+", span)
    assert fence is not None
    inner = span[len(fence.group(0)) : len(span) - len(fence.group(0))]
    assert inner.strip(" ") == flat.strip(" ") or inner == flat
    longest = max((len(run) for run in re.findall(r"`+", flat)), default=0)
    assert len(fence.group(0)) > longest
