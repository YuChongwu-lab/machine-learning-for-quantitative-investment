"""Data path resolution: locates the repo's top-level data/ directory.
"""

from __future__ import annotations

from pathlib import Path


def _find_repo_root() -> Path:
    """Walk up from this file's location to find the repo root (the first
    ancestor that contains both a data/ and a tools/ directory)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "data").is_dir() and (parent / "tools").is_dir():
            return parent
    raise RuntimeError(
        "Could not locate the repo root: make sure this file lives inside "
        "the repo and that a top-level data/ directory exists."
    )


REPO_ROOT = _find_repo_root()

#: The single location where all of the book's data lives
DATA_DIR = REPO_ROOT / "data"

#: Daily per-stock price data
STOCK_DIR = DATA_DIR / "stocks"


def data_file(name: str) -> Path:
    """Return the path to a data file under data/, with an actionable
    error message if it's missing."""
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Please set up the data/ directory as described in the "
            f"top-level README.md's 'Data' section."
        )
    return path


def stock_file(name: str) -> Path:
    """Return the path to a per-stock data file under data/stocks/."""
    path = STOCK_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Per-stock data file not found: {path}\n"
            f"Please set up the data/stocks/ directory as described in the "
            f"top-level README.md's 'Data' section."
        )
    return path
