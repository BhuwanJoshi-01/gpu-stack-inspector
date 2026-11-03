import pytest
from hypothesis import given
from hypothesis import strategies as st

from gpu_stack_inspector.compatibility.versions import major_minor, version_tuple


@pytest.mark.parametrize(
    ("text", "parts"),
    [
        ("12.8", (12, 8)),
        ("595.99.02", (595, 99, 2)),
        ("12.4.131", (12, 4, 131)),
        ("580", (580,)),
        (" 13.0 ", (13, 0)),
        ("2.11.0+cu128", (2, 11, 0)),
        ("unknown", None),
        ("", None),
        (None, None),
    ],
)
def test_version_tuple(text: str | None, parts: tuple[int, ...] | None) -> None:
    assert version_tuple(text) == parts


def test_major_minor() -> None:
    assert major_minor("12.4.131") == (12, 4)
    assert major_minor("13") == (13, 0)
    assert major_minor("n/a") is None


def test_driver_versions_compare_numerically_not_as_text() -> None:
    assert version_tuple("595.99.02") > version_tuple("580")  # type: ignore[operator]
    assert version_tuple("1000.1") > version_tuple("999.99")  # type: ignore[operator]


@given(st.text(max_size=30))
def test_any_text_gives_a_tuple_or_none(text: str) -> None:
    parts = version_tuple(text)
    assert parts is None or all(isinstance(part, int) for part in parts)
