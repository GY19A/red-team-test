#!/usr/bin/env python3
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Load runner.py directly since folder name has dash
spec = importlib.util.spec_from_file_location("cka_runner", str(Path(__file__).parent / "runner.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
main = runner.main

if __name__ == "__main__":
    main()
