# Testing Guide

The Frametime Analyser project includes a comprehensive test suite covering unit and integration tests.

## Running Tests

Run all test commands from the repository root (the folder containing `libraries.py`).

### Install dependencies

```bash
pip install -r requirements.txt
```

### Using unittest (built-in)

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_libraries

# Run with verbose output
python -m unittest discover tests -v
```

### Using pytest (recommended)

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report (written to htmlcov/)
pytest tests/ --cov=libraries --cov-report=html

# Run specific test file
pytest tests/test_libraries.py -v
```

## Test Coverage

The test suite includes:

### Unit Tests (`test_libraries.py`)

- **ResizeWithAspectRatio function testing**
  - Resize by width while maintaining aspect ratio
  - Resize by height while maintaining aspect ratio
  - No resize parameters (returns unchanged image)
  - Small image resizing
  - Color channel preservation

- **FileVideoStream class testing**
  - Initialization with invalid paths
  - Queue initialization and size configuration
  - Default queue size verification
  - Thread daemon flag validation
  - Initial state verification

- **Frametime calculation testing**
  - 60fps frametime calculation (16.67ms)
  - 120fps frametime calculation (8.33ms)
  - Stutter threshold calculation for 60fps
  - Severe stutter threshold calculation for 120fps

- **Histogram bucket testing** (`HISTOGRAM_BUCKETS`, used by both the on-screen histogram and the CSV export)
  - 60fps histogram bucket ranges
  - 120fps histogram bucket ranges
  - Buckets are contiguous and the ideal frametime lands in "Perfect"
  - Performance category thresholds at the bucket edges
  - Frame counting per bucket

- **FPS counter testing** (`FpsCounter`)
  - Window spans one second at both 60fps and 120fps
  - Smooth video reaches the full target FPS
  - Repeated frames lower the FPS; old frames leave the window

- **Image processing testing**
  - BGR to grayscale conversion
  - Frame difference calculation
  - Identical frame difference detection

### Integration Tests (`test_integration.py`)

- **Video generation testing**
  - Test video creation with various parameters
  - Video with duplicate frames (simulating frame drops)
  - Frame count verification
  - OpenCV compatibility validation

- **Command-line argument parsing** (runs `Frametime Analyser.py` in a subprocess)
  - Invalid FPS value rejection
  - Unknown argument rejection
  - Missing video file rejection
  - `-watermark` without a path rejection
  - Valid arguments are not exercised end to end, because they start video playback

- **Output file generation**
  - Analyzed video filename generation
  - Histogram CSV filename generation
  - Full path handling

- **Performance metrics**
  - FPS calculation from frame lists
  - Average frametime calculation
  - Stutter detection logic
  - Moderate vs severe stutter classification

- **Data structure testing**
  - Frametime graph initialization
  - Histogram buffer size (30-minute capacity)

## Test Files

- `tests/__init__.py` - Test runner and discovery
- `tests/test_libraries.py` - Unit tests for core functions
- `tests/test_integration.py` - Integration and system tests

## Adding New Tests

When adding new features, please include corresponding tests:

1. **Unit tests** for individual functions in `test_libraries.py`
2. **Integration tests** for feature workflows in `test_integration.py`
3. Follow existing naming conventions: `test_<feature_name>`
4. Include docstrings explaining what each test validates

## Before Committing

There is no CI set up for this repository, so run the tests yourself before committing.

```bash
# Quick test run before commit
python -m unittest discover tests -v

# Coverage report in the terminal
pytest tests/ --cov=libraries --cov-report=term-missing
```

Coverage only measures `libraries.py`. `Frametime Analyser.py` can't be imported (its name contains a space, and it starts analysing on import), so logic that needs testing belongs in `libraries.py`.
