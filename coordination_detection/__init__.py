"""Compatibility shim that exposes the src-layout package from the repo root."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
_src_package = Path(__file__).resolve().parent.parent / "src" / "coordination_detection"
if _src_package.exists():
    __path__.append(str(_src_package))
