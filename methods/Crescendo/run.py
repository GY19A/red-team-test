#!/usr/bin/env python3
"""Entry: PyRIT Crescendo multi-turn attack (see runner.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from methods.Crescendo.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
