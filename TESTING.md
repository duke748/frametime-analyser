# Testing Guide

The Frametime Analyser project includes a comprehensive test suite covering unit and integration tests.

## Running Tests

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
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=html

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

- **Histogram bucket testing**
  - 60fps histogram bucket ranges
  - 120fps histogram bucket ranges
  - Performance category thresholds

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

- **Command-line argument parsing**
  - Valid FPS argument parsing (60, 120)
  - Invalid FPS value rejection
  - Frametime calculation from FPS

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
  - FPS list structure (1-second window)

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

## Continuous Integration

Tests should be run before committing changes to ensure code quality and prevent regressions.

```bash
# Quick test run before commit
python -m unittest discover tests -v

# Full coverage report
pytest tests/ --cov=. --cov-report=term-missing
```
