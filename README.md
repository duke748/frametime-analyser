# Frametime Analyser (Enhanced Fork)

**Frametime Analyser** is an advanced tool that detects duplicate frames in game footage to calculate FPS, frame times, and performance metrics. This enhanced version provides comprehensive visual analysis with multiple graphs, statistics, and export capabilities.

## About This Fork

This is an enhanced fork of the original [Frametime Analyser by AndrewKeYanzhe](https://github.com/AndrewKeYanzhe/frametime-analyser). The original tool was inspired by Digital Foundry's FPSGui tool:

### Major Enhancements Added:

- ✅ **Python 3 compatibility** (upgraded from Python 2.7)
- ✅ **60fps and 120fps mode support** with dynamic scaling
- ✅ **Video output** with embedded analysis overlays
- ✅ **Command-line argument parsing** for flexible usage
- ✅ **Real-time graphs**: Frametime, FPS, and optional Frame Delta plot
- ✅ **Frametime Distribution Histogram** with color-coded performance buckets
- ✅ **Cumulative Distribution Function (CDF)** smoothness graph
- ✅ **Average statistics** for frametime and FPS over 30-minute rolling window
- ✅ **CSV export** of histogram data and performance metrics
- ✅ **Anti-aliased text** with black borders for improved readability
- ✅ **Micro-stutter detection** via Frame-to-Frame Delta analysis
- ✅ **Extended histogram buffer** (30 minutes of data)
- ✅ **Stutter detection** for shader compilation and traversal stutters

---

## Features

### 📊 Real-Time Visual Analysis

1. **Frametime Graph (Left)**
   - Shows absolute frametime in milliseconds for each unique frame
   - Scale: 0 to 2.5× target frametime (e.g., 0-41.7ms for 60fps)
   - Helps identify individual frame drops and stutter events

2. **Frame-to-Frame Delta Plot (Optional)**
   - Detects micro-stutter by plotting the absolute difference between consecutive frame times
   - Color-coded: Green (smooth) → Yellow (noticeable) → Red (micro-stutter)
   - Reveals frame pacing issues invisible in FPS averages
   - Enable with `-delta-plot` flag

3. **Cumulative Distribution Function / Smoothness Graph**
   - **What it shows**: The percentage of frames rendered faster than a given frametime
   - **X-axis**: Frametime in milliseconds (0 to 3× target frametime)
   - **Y-axis**: Cumulative percentage (0-100%)
   - **Interpretation**:
     - **Steep curve** = Consistent performance (most frames cluster around target frametime)
     - **Gradual curve** = Inconsistent performance (frames spread across many timings)
   - **Reference lines**:
     - Vertical green line at target frametime (16.7ms for 60fps, 8.3ms for 120fps)
     - Horizontal gray lines at 50% (median) and 95% (95th percentile)
   - **Example**: If the curve reaches 95% at the target frametime line, then 95% of your frames hit the target - excellent consistency!
   - This is the same metric used by CapFrameX's "Smoothness Graph"

4. **FPS Graph (Bottom)**
   - Real-time FPS tracking over time
   - Shows 120/60/30/0 scale markers
   - Reference lines at key FPS values for easy reading

5. **Frametime Distribution Histogram**
   - Shows percentage distribution across 5 performance buckets:
     - **Perfect**: 0-18ms (60fps) or 0-9ms (120fps) - Green
     - **Good**: 18-25ms (60fps) or 9-12ms (120fps) - Green
     - **Fair**: 25-35ms (60fps) or 12-17ms (120fps) - Yellow
     - **Stutter**: 35-50ms (60fps) or 17-25ms (120fps) - Orange
     - **Bad**: 50ms+ (60fps) or 25ms+ (120fps) - Red
   - Displays both bar graph and percentage table
   - Shows frametime range thresholds for each bucket
   - Includes average frametime and average FPS indicators

6. **Stutter Detection (Optional)**
   - Automatically detects and highlights severe frametime spikes
   - Two severity levels:
     - **Moderate Stutters**: 2.5× target frametime (≥41.7ms @ 60fps, ≥20.8ms @ 120fps) - Orange markers
     - **Severe Stutters**: 4× target frametime (≥66.7ms @ 60fps, ≥33.3ms @ 120fps) - Red markers
   - Visual indicators:
     - Vertical lines on frametime graph at stutter locations
     - Real-time counter showing total/severe/moderate stutter count
   - Useful for detecting:
     - Shader compilation stutters (first-time asset loading)
     - Traversal stutters (moving to new game areas)
     - Texture streaming issues
   - Statistics exported to CSV including average and worst-case stutter severity
   - Enable with `-detect-stutters` flag

### 📁 Data Export

- **CSV Export**: Automatically generates `[filename]_histogram.csv` containing:
  - Video metadata (file path, target FPS)
  - Average frametime and FPS over analysis window
  - Total unique frames analyzed
  - Analysis duration
  - Percentage distribution across all performance buckets

- **Video Export**: Generates `[filename]_analyzed.mp4` with all overlays embedded

---

## Setup

### Prerequisites

* **Python 3.x** (tested with Python 3.13+)
* **OpenCV** (`opencv-python`)
* **NumPy**
* **CSV module** (built-in)

### Installation

```bash
# Clone this repository
git clone https://github.com/duke748/frametime-analyser.git
cd frametime-analyser

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install numpy opencv-python matplotlib
```

---

## Usage

### Basic Command

```bash
python "Frametime Analyser.py" <video_file> [fps] [options]
```

### Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `video_file` | string | ✅ Yes | - | Path to the video file to analyze |
| `fps` | integer | ❌ No | 60 | Target FPS mode: `60` or `120` |
| `-delta-plot` | flag | ❌ No | disabled | Enable Frame-to-Frame Delta plot for micro-stutter detection |
| `-watermark <file>` | flag + path | ❌ No | disabled | Add a 32x32 watermark image to bottom left corner (50% opacity) |
| `-title <text>` | flag + string | ❌ No | disabled | Add a centered title at the top of the screen with opaque background |
| `-detect-stutters` | flag | ❌ No | disabled | Detect and highlight severe frametime spikes (shader/traversal stutters) |

### Examples

#### 1. Basic analysis at 60fps (default)
```bash
python "Frametime Analyser.py" "gameplay.mp4"
```

#### 2. Analyze 120fps video
```bash
python "Frametime Analyser.py" "gameplay.mp4" 120
```

#### 3. Enable micro-stutter detection (delta plot)
```bash
python "Frametime Analyser.py" "gameplay.mp4" 60 -delta-plot
```

#### 4. 120fps with delta plot (arguments can be in any order)
```bash
python "Frametime Analyser.py" "gameplay.mp4" -delta-plot 120
```

#### 5. Add watermark
```bash
python "Frametime Analyser.py" "gameplay.mp4" -watermark "./bh.jpg"
```

#### 6. All options combined
```bash
python "Frametime Analyser.py" "gameplay.mp4" 120 -delta-plot -watermark "./logo.png"
```

#### 7. Add title
```bash
python "Frametime Analyser.py" "gameplay.mp4" -title "Cyberpunk 2077 - Ultra Settings"
```

#### 8. Complete example with all options
```bash
python "Frametime Analyser.py" "gameplay.mp4" 120 -delta-plot -watermark "./logo.png" -title "Game Performance Test"
```

#### 9. Detect severe stutters (shader compilation, traversal)
```bash
python "Frametime Analyser.py" "gameplay.mp4" -detect-stutters
```

#### 10. Full analysis with stutter detection
```bash
python "Frametime Analyser.py" "gameplay.mp4" 120 -delta-plot -detect-stutters -title "Full Analysis"
```

### Output Files

After analysis completes, two files are generated:

1. **`[filename]_analyzed.mp4`** - Video with overlay graphs and statistics
2. **`[filename]_histogram.csv`** - Performance data export

Example CSV output:
```csv
Frametime Analysis Results
Video File,C:\path\to\gameplay.mp4
Target FPS,60
Average Frametime (ms),20.16
Average FPS,48.50
Total Unique Frames Analyzed,2160
Analysis Duration,36.0 seconds (0.60 minutes)
Sample Window (Max),216000 frames (last 30 minutes of video)

Stutter Detection Results
Total Stutters Detected,15
Severe Stutters (>= 4x target frametime),3
Moderate Stutters (>= 2.5x target frametime),12
Severe Threshold (ms),66.7
Moderate Threshold (ms),41.7
Average Stutter Frametime (ms),52.4
Worst Stutter Frametime (ms),125.8

Bucket,Range (ms),Frame Count,Percentage
Perfect,0-18,1766,81.67%
Good,18-25,216,10.00%
Fair,25-35,108,5.00%
Stutter,35-50,43,2.00%
Bad,50+,27,1.33%
```

*Note: Stutter detection section only appears when `-detect-stutters` flag is used.*

---

## Technical Overview

### Key Processing Steps

1. **Frame Difference Calculation**
   - Downscale frame to 10% for faster processing
   - Convert to grayscale
   - Calculate absolute difference from previous frame
   - Average difference across all pixels = `frame_diff`

2. **Duplicate Frame Detection**
   - Compare `frame_diff` with exponential moving average
   - If `frame_diff` < 25% of moving average → duplicate frame
   - Unique frames increment the frametime counter
   - Duplicate frames continue incrementing the counter

3. **Frametime Recording**
   - When a unique frame is detected, record accumulated frametime
   - Convert frame count to milliseconds: `frametime_count × target_frametime`
   - Store in rolling 30-minute buffer (216,000 frames at 120fps)

4. **Statistical Analysis**
   - Calculate real-time FPS over 60-frame window
   - Track frametime distribution across performance buckets
   - Compute frame-to-frame deltas for micro-stutter detection
   - Generate CDF curve from sorted frametime data

### Performance Notes

- **CPU-intensive**: OpenCV decodes video in software
- **Recommended**: i7-3770 or better for H.264 @ 1080p
- **Higher resolutions/H.265**: Requires faster CPU
- **Memory usage**: ~1.7MB for 30-minute histogram buffer (negligible)

---

## Testing

The project includes a comprehensive test suite. See [TESTING.md](TESTING.md) for detailed testing instructions and coverage information.

**Quick Start:**
```bash
# Run all tests
python -m unittest discover tests -v

# Or using pytest
pytest tests/ -v
```

---

## Limitations

* Input video must be recorded at a **constant frame rate** matching the target FPS (60 or 120)
* Dropped frames must be encoded as **duplicate frames** in the video
* Does **not account for screen tearing** - V-Sync should be enabled during recording
* Analysis assumes game is rendering frames at target rate; repeated frames indicate performance issues
* Very short videos (<10 frames) may not generate meaningful histogram data

---

## Comparison with Original

| Feature | Original | This Fork |
|---------|----------|-----------|
| Python Version | 2.7 | 3.x |
| FPS Modes | 60fps only | 60fps & 120fps |
| Command-line Args | No | Yes |
| Video Output | No | Yes |
| Histogram | No | Yes (5 buckets) |
| CDF Graph | No | Yes |
| Delta Plot | No | Yes (optional) |
| CSV Export | No | Yes |
| Statistics | Basic | Average FPS, Average Frametime |
| Rolling Window | N/A | 30 minutes |
| Text Visibility | Basic | Anti-aliased with borders |

---

## License

Same as original repository.

---

## Credits

- **Original tool**: [AndrewKeYanzhe/frametime-analyser](https://github.com/AndrewKeYanzhe/frametime-analyser)
- **Inspired by**: Digital Foundry's FPSGui tool
- **Enhanced fork**: [duke748/frametime-analyser](https://github.com/duke748/frametime-analyser)

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests with improvements.

