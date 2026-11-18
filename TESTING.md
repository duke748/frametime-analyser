# Testing Guide

The Frametime Analyser project includes a test suite covering unit and integration tests, including an end-to-end run on a generated video.

## Running Tests

Run all test commands from the repository root (the folder containing `frametime_analyser.py`).

### Install dependencies

```bash
pip install -r requirements-dev.txt
```

This installs the runtime dependencies plus `pytest` and `pytest-cov`.

### Using pytest (recommended)

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report (written to htmlcov/)
pytest tests/ --cov=libraries --cov=frametime_analyser --cov-report=html

# Run specific test file
pytest tests/test_libraries.py -v
```

### Using unittest (built-in)

```bash
# Run all tests
python -m unittest discover tests -v

# Run specific test file
python -m unittest tests.test_libraries
```

## Test Coverage

### Unit Tests (`test_libraries.py`)

- **Resizing**: `ResizeWithAspectRatio`, and `fit_to_canvas` letterboxing/pillarboxing any shape onto the 1280×720 canvas
- **FileVideoStream**: queue setup, initial state, and reading every frame of a real video before returning `None`
- **Frame difference**: identical and different frames
- **Histogram buckets** (`get_histogram_buckets`, used by the overlay and the CSV export)
  - 60fps and 120fps bucket ranges
  - Buckets for other target FPS values scale from the target frametime
  - Buckets are contiguous, the ideal frametime lands in "Perfect", and long freezes land in "Bad"
  - Bucket range formatting
- **FPS counter** (`FpsCounter`): one-second window at any target FPS, repeated frames lower the FPS
- **FPS colours** (`fps_color`): thresholds scale with the target FPS
- **Frametime window** (`FrametimeWindow`)
  - Mean, average FPS, percentiles, 1%/0.1% lows and CDF
  - Only the most recent 30 minutes of video are kept
  - Bucket counts agree with `count_frames_per_bucket`

### Integration Tests (`test_integration.py`)

- **Command-line parsing** (`parse_args`)
  - Defaults, any target FPS from 1 to 1000, invalid FPS values
  - Both `--flag` and legacy `-flag` spellings, and the FPS value after options
  - Missing video or watermark files, missing option values, unknown options
- **Legacy entry point**: `Frametime Analyser.py --help`, error exit codes and a headless run in a subprocess
- **Analyser** (`FrametimeAnalyser`), fed frame differences directly
  - Smooth, half-rate and 120fps video
  - Repeated frames lengthen the frametime
  - Moderate and severe stutter detection
  - Frame-to-frame delta and summary statistics
  - Every overlay option draws at several target FPS values
- **End to end**: a generated video with known dropped frames is analysed with `--no-display`; the per-frame CSV must match the drops exactly, and the summary CSV and output video are checked
- **FPS mismatch warning** when the video's frame rate differs from the target

## Test Files

- `tests/__init__.py` - Test runner and discovery
- `tests/test_libraries.py` - Unit tests for `libraries.py`
- `tests/test_integration.py` - Integration and end-to-end tests for `frametime_analyser.py`

## Adding New Tests

When adding new features, please include corresponding tests:

1. **Unit tests** for helpers in `libraries.py` go in `test_libraries.py`
2. **Integration tests** for `frametime_analyser.py` go in `test_integration.py`. Import it with `import frametime_analyser`; importing has no side effects
3. Follow existing naming conventions: `test_<feature_name>`
4. Include docstrings explaining what each test validates

## Continuous Integration

GitHub Actions runs the test suite with coverage on every push and pull request (`.github/workflows/tests.yml`). Run the tests locally before pushing:

```bash
pytest tests/ --cov=libraries --cov=frametime_analyser --cov-report=term-missing
```
