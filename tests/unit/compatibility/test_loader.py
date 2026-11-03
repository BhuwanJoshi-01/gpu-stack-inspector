import copy
import json
from importlib.resources import files
from typing import Any

import pytest

from gpu_stack_inspector.compatibility.loader import CompatibilityDataError, load, parse

RAW: dict[str, Any] = json.loads(
    files("gpu_stack_inspector.compatibility").joinpath("data.json").read_text("utf-8")
)


def test_the_shipped_data_loads_and_cites_nvidia() -> None:
    data = load()
    assert data.retrieved == "2026-11-03"
    assert {major: driver for major, (driver, _) in data.minimum_driver.items()} == {
        11: "450",
        12: "525",
        13: "580",
    }
    assert data.minor_version_from_major == 11
    for _, source in data.minimum_driver.values():
        assert source.url.startswith("https://docs.nvidia.com/")


def test_rows_with_an_unknown_source_are_refused() -> None:
    broken = copy.deepcopy(RAW)
    broken["minimum_driver"][0]["source"] = "a-blog-post"
    with pytest.raises(CompatibilityDataError, match="unknown source"):
        parse(broken)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("retrieved",), "last week", "YYYY-MM-DD"),
        (("schema",), 2, "unsupported schema"),
        (("sources", "nvidia-cuda-compatibility", "url"), "http://example.invalid", "https"),
    ],
)
def test_data_without_a_date_schema_or_https_source_is_refused(
    path: tuple[str, ...], value: object, message: str
) -> None:
    broken = copy.deepcopy(RAW)
    target = broken
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(CompatibilityDataError, match=message):
        parse(broken)


def test_a_non_numeric_driver_version_is_refused() -> None:
    broken = copy.deepcopy(RAW)
    broken["minimum_driver"][1]["driver"] = "recent"
    with pytest.raises(CompatibilityDataError, match="not a version"):
        parse(broken)
