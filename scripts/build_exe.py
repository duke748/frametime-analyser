"""Build a standalone executable with PyInstaller (no Python needed to run it).

    pip install .[build]
    python scripts/build_exe.py

Output: dist/frametime-analyser.exe (or dist/frametime-analyser on Linux/macOS).
The bundled FFmpeg from imageio-ffmpeg is included.
"""
import os
import sys

import PyInstaller.__main__

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    PyInstaller.__main__.run([
        os.path.join(ROOT, "scripts", "entry.py"),
        "--name", "frametime-analyser",
        "--onefile",
        "--console",
        "--noconfirm",
        "--clean",
        "--paths", ROOT,
        "--collect-data", "imageio_ffmpeg",
        "--distpath", os.path.join(ROOT, "dist"),
        "--workpath", os.path.join(ROOT, "build"),
        "--specpath", os.path.join(ROOT, "build"),
        *sys.argv[1:],
    ])


if __name__ == "__main__":
    main()
