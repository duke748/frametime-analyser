# Testing Guide

The test suite covers every module, plus end-to-end runs on generated videos with known frame timing: dropped frames, a pause, a facecam overlay, two captures to compare, and a V-Sync-off capture with tearing.

## Running Tests

Run commands from the repository root.

```bash
# Install the runtime and test dependencies (pytest, pytest-cov, ruff)
pip install -r requirements-dev.txt

# Lint
ruff check .

# Run all tests
pytest -v

# With coverage
pytest --cov --cov-report=term-missing   # or --cov-report=html (written to htmlcov/)

# One file
pytest tests/test_analyser.py -v
```

The built-in runner works too: `python -m unittest discover tests -v`.

Tests that need FFmpeg are skipped when neither `ffmpeg` on PATH nor `imageio-ffmpeg` is available.

## Test Files

| File | Covers |
|---|---|
| `tests/helpers.py` | Generators for test videos with known timing: repeated frames, overlays, and V-Sync-off tearing from a flip schedule |
| `tests/test_stats.py` | Histogram buckets for any target FPS, FPS colours, the FPS counter, and the 30-minute frametime window (mean, percentiles, lows, CDF) |
| `tests/test_video.py` | Canvas fitting, OpenCV and FFmpeg decoding (including letterboxing and stopping early), OpenCV and FFmpeg writers, and error handling in the background writer |
| `tests/test_detection.py` | The 128×72 analysis image, mean and peak frame changes, ROI/ignore masks, and the tear detector (tears, catch-up rows, learned static rows that can recover, fallback for scattered changes) |
| `tests/test_analyser.py` | Duplicate detection (mean and peak), the noise floor, static scenes, stutter thresholds (including floating-point edge cases), torn-frame timing, and the V-Sync-off warning (including a static sky that must not trigger it) |
| `tests/test_overlay.py` | Layer compositing and caching, graphs, and every overlay option at several target FPS values, single and compare |
| `tests/test_cli.py` | Argument parsing, the legacy entry points, and end-to-end runs: exact frametimes for known drops (OpenCV and FFmpeg paths), report, output folder, FPS detection, facecam `--ignore`, static scenes, tearing, the tearing warning, compare, and a missing capture device |

## Adding New Tests

1. Put tests next to the module they cover (`test_<module>.py`)
2. For behaviour that depends on real video, generate a small clip with the helpers in `tests/helpers.py` and run `cli.main([...])` with `--no-display`
3. Include docstrings explaining what each test validates

## Continuous Integration

GitHub Actions (`.github/workflows/tests.yml`) lints with ruff and runs the tests on Windows with Python 3.11 and 3.13 for every push and pull request. For version tags (`v*`) and manual runs, it also builds the standalone `.exe` and uploads it as an artifact.

## Building the Executable Locally

```bash
pip install .[build]
python scripts/build_exe.py
dist/frametime-analyser.exe --version
```
