"""Loads data.json and refuses data that is not tied to a source."""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Any


@dataclass(frozen=True, slots=True)
class Source:
    id: str
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class CompatibilityData:
    """CUDA compatibility facts, each tied to the document it was taken from."""

    retrieved: str
    sources: dict[str, Source]
    minor_version_from_major: int
    minor_version_source: Source
    minor_version_note: str
    minimum_driver: dict[int, tuple[str, Source]]
    """CUDA major version → (minimum driver version, source)."""


class CompatibilityDataError(ValueError):
    pass


@functools.cache
def load() -> CompatibilityData:
    raw = files("gpu_stack_inspector.compatibility").joinpath("data.json").read_text("utf-8")
    return parse(json.loads(raw))


def parse(data: dict[str, Any]) -> CompatibilityData:
    if data.get("schema") != 1:
        raise CompatibilityDataError(f"unsupported schema {data.get('schema')!r}")
    retrieved = data.get("retrieved", "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(retrieved)):
        raise CompatibilityDataError("retrieved must be a YYYY-MM-DD date")
    sources = {
        key: Source(key, value["title"], value["url"]) for key, value in data["sources"].items()
    }
    for source in sources.values():
        if not source.url.startswith("https://"):
            raise CompatibilityDataError(f"source {source.id} has no https URL")

    def source_of(entry: dict[str, Any]) -> Source:
        key = entry.get("source")
        if key not in sources:
            raise CompatibilityDataError(f"unknown source {key!r} in {entry}")
        return sources[key]

    minor = data["minor_version_compatibility"]
    minimum: dict[int, tuple[str, Source]] = {}
    for row in data["minimum_driver"]:
        if not re.fullmatch(r"\d+(\.\d+)*", str(row["driver"])):
            raise CompatibilityDataError(f"driver version {row['driver']!r} is not a version")
        minimum[int(row["cuda_major"])] = (str(row["driver"]), source_of(row))
    return CompatibilityData(
        retrieved=retrieved,
        sources=sources,
        minor_version_from_major=int(minor["from_cuda_major"]),
        minor_version_source=source_of(minor),
        minor_version_note=str(minor["note"]),
        minimum_driver=minimum,
    )
