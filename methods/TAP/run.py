#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from methods.TAP.tap_runner import main  # noqa: E402

if __name__ == "__main__":
    main()
