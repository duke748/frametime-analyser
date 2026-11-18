"""Backwards-compatible entry point. The analyser lives in frametime_analyser.py."""
import sys

from frametime_analyser import main

if __name__ == "__main__":
    sys.exit(main())
