"""Entry point for the PyInstaller executable."""
import sys

from frametime_analyser.cli import main

if __name__ == "__main__":
    sys.exit(main())
