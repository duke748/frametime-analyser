"""Frametime Analyser: detect repeated frames in game footage and measure
frame rate, frametimes, stutter and tearing."""

__version__ = "2.0.0"


def main(argv=None):
    from .cli import main as _main
    return _main(argv)
