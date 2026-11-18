"""Backwards-compatible entry point. Prefer `python -m frametime_analyser` or the
`frametime-analyser` command installed by `pip install .`."""
import sys

from frametime_analyser.cli import main

if __name__ == "__main__":
    sys.exit(main())
