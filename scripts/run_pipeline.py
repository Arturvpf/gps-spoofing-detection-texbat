#!/usr/bin/env python3
"""Entry-point script to extract TEXBAT SQM features and run the full ML evaluation suite."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from texbat_spoofing.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
