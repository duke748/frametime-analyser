# Frametime Analyser (Enhanced Fork)

**Frametime Analyser** detects repeated frames in game footage to work out FPS, frametimes and stutter. It plays the video back with performance graphs overlaid, saves an annotated copy, and exports the results to CSV.

![Frametime Analyser overlay on Red Dead Redemption 2 footage](Media/RDR2.gif)

*Output from the original version of the tool. This fork adds the delta, smoothness (CDF) and histogram panels described below.*

## About This Fork

This is an enhanced fork of the original [Frametime Analyser by AndrewKeYanzhe](https://github.com/AndrewKeYanzhe/frametime-analyser). The original tool was inspired by Digital Foundry's FPSGui tool.

### Major Enhancements Added

- **Python 3 compatibility** (upgraded from Python 2.7)
- **Any target frame rate** (60, 120, 144, 240, ...) with dynamic scaling
- **Video output** with embedded analysis overlays
- **Command-line options** with `--help`
- **Headless mode** (`--no-display`) that analyses as fast as the CPU allows
- **Real-time graphs**: Frametime, FPS, and optional Frame Delta plot
- **Frametime Distribution Histogram** with color-coded performance buckets
- **Cumulative Distribution Function (CDF)** smoothness graph
- **Average FPS, 1% low and 0.1% low FPS** over the last 30 minutes of video
- **CSV export** of summary statistics and of every frame's frametime
- **Micro-stutter detection** via Frame-to-Frame Delta analysis
- **Stutter detection** for shader compilation and traversal stutters

---

## Features

### Real-Time Visual Analysis

1. **Frametime Graph**
   - Shows the frametime in milliseconds of each unique frame
   - Scale: 0 to 2.5× target frametime (e.g. 0–41.7ms at 60fps), with a grey reference line at the target frametime
   - Helps identify individual frame drops and stutter events

2. **Frame-to-Frame Delta Plot (Optional)**
   - Detects micro-stutter by plotting the absolute difference between consecutive frametimes
   - Color-coded: Green (smooth) → Yellow (noticeable) → Red (micro-stutter)
   - Reveals frame pacing issues invisible in FPS averages
   - Enable with `--delta-plot`

3. **Cumulative Distribution Function / Smoothness Graph**
   - **What it shows**: The percentage of frames rendered at or faster than a given frametime
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
   - FPS over the last ~10 seconds of video
   - Scaled from 0 to the target FPS, with reference lines at 30, 60, 120 and 240 where they fit

5. **Frametime Distribution Histogram**
   - Shows percentage distribution across 5 performance buckets:

     | Bucket | 60fps | 120fps | Other targets (× target frametime) | Color |
     |---|---|---|---|---|
     | **Perfect** | 0–18ms | 0–9ms | 0–1.08× | Green |
     | **Good** | 18–25ms | 9–12ms | 1.08–1.5× | Green |
     | **Fair** | 25–35ms | 12–17ms | 1.5–2.1× | Yellow |
     | **Stutter** | 35–50ms | 17–25ms | 2.1–3× | Orange |
     | **Bad** | 50ms+ | 25ms+ | 3×+ | Red |

   - Displays both bar graph and percentage table
   - Shows average frametime, average FPS and 1% low FPS above the graph

6. **Stutter Detection (Optional)**
   - Automatically detects and highlights severe frametime spikes
   - Two severity levels:
     - **Moderate Stutters**: 2.5× target frametime (≥41.7ms @ 60fps, ≥20.8ms @ 120fps) - Orange markers
     - **Severe Stutters**: 4× target frametime (≥66.7ms @ 60fps, ≥33.3ms @ 120fps) - Red markers
   - Visual indicators:
     - Vertical lines on the frametime graph at the stuttering frames
     - Counter showing total/severe/moderate stutter count
   - Useful for detecting:
     - Shader compilation stutters (first-time asset loading)
     - Traversal stutters (moving to new game areas)
     - Texture streaming issues
   - Statistics exported to CSV, including average and worst-case stutter severity
   - Enable with `--detect-stutters`

---

## Setup

### Prerequisites

* **Python 3.11+**
* **OpenCV** (`opencv-python`)
* **NumPy**

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
pip install -r requirements.txt
```

---

## Usage

### Basic Command

```bash
python frametime_analyser.py <video_file> [fps] [options]
```

`python "Frametime Analyser.py" ...` still works too.

By default the video plays in a window **in real time**, so a 10-minute capture takes 10 minutes. Press **`q`** in the window to stop early; results up to that point are still saved. Use `--no-display` to skip the window and analyse as fast as possible.

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `video_file` | Yes | - | Path to the video file to analyse |
| `fps` | No | 60 | Frame rate the video was recorded at (1–1000), e.g. `60`, `120`, `144` |
| `--delta-plot` | No | off | Show the Frame-to-Frame Delta plot for micro-stutter detection |
| `--detect-stutters` | No | off | Detect and highlight severe frametime spikes (shader/traversal stutters) |
| `--watermark <file>` | No | off | Add a 32×32 watermark image to the bottom left corner (50% opacity) |
| `--title <text>` | No | off | Add a centered title at the top of the screen with an opaque background |
| `--no-display` | No | off | No preview window and no real-time playback; prints progress instead |
| `--help` | No | - | Show all options |

The single-dash spellings from earlier versions (`-delta-plot`, `-detect-stutters`, `-watermark`, `-title`) still work. Options and the `fps` value can be given in any order.

### Examples

```bash
# Basic analysis at 60fps (default)
python frametime_analyser.py gameplay.mp4

# Analyse a 120fps or 144fps capture
python frametime_analyser.py gameplay.mp4 120
python frametime_analyser.py gameplay.mp4 144

# Micro-stutter (delta plot) and stutter detection
python frametime_analyser.py gameplay.mp4 --delta-plot --detect-stutters

# Add a watermark and title
python frametime_analyser.py gameplay.mp4 --watermark logo.png --title "Cyberpunk 2077 - Ultra Settings"

# Analyse a long capture as fast as possible, without the preview window
python frametime_analyser.py gameplay.mp4 120 --detect-stutters --no-display
```

### Output Files

Three files are written next to the input video:

1. **`[filename]_analyzed.mp4`** - The video with overlay graphs and statistics. It is always 1280×720 at the target frame rate; other aspect ratios are letterboxed or pillarboxed to fit.
2. **`[filename]_histogram.csv`** - Summary statistics (see below)
3. **`[filename]_frametimes.csv`** - One row per unique frame: `unique_frame, video_frame, time_s, frametime_ms`. Use this to analyse or chart the data in other tools.

Example `_histogram.csv`:
```csv
Frametime Analysis Results
Video File,C:\path\to\gameplay.mp4
Target FPS,60
Analysis Duration,36.0 seconds (0.60 minutes)
Total Unique Frames Analyzed,1746
Statistics Window,last 30 minutes of video
Unique Frames in Window,1746
Average Frametime (ms),20.62
Average FPS,48.50
99th Percentile Frametime (ms),50.00
1% Low FPS,20.00
0.1% Low FPS,12.00

Stutter Detection Results
Total Stutters Detected,28
Severe Stutters (>= 4x target frametime),6
Moderate Stutters (>= 2.5x target frametime),22
Severe Threshold (ms),66.7
Moderate Threshold (ms),41.7
Average Stutter Frametime (ms),60.7
Worst Stutter Frametime (ms),133.3

Bucket,Range (ms),Frame Count,Percentage
Perfect,0-18,1426,81.67%
Good,18-25,175,10.02%
Fair,25-35,87,4.98%
Stutter,35-50,35,2.00%
Bad,50+,23,1.32%
```

*The stutter section only appears when `--detect-stutters` is used.*

**How the statistics are calculated** (over the last 30 minutes of video):
- **Average FPS** = unique frames ÷ seconds of video, which equals 1000 ÷ average frametime
- **99th percentile frametime**: 99% of frames were this fast or faster
- **1% low / 0.1% low FPS** = 1000 ÷ the 99th / 99.9th percentile frametime

---

## Technical Overview

### Key Processing Steps

1. **Frame Difference Calculation**
   - Fit the frame onto a 1280×720 canvas, then downscale to 10% for faster processing
   - Convert to grayscale
   - Average absolute difference from the previous frame = `frame_diff`

2. **Duplicate Frame Detection**
   - Compare `frame_diff` with an exponential moving average of recent differences
   - If `frame_diff` < 25% of the moving average → repeated (dropped) frame
   - Otherwise it is a new, unique frame

3. **Frametime Recording**
   - A unique frame's frametime is the number of video frames since the previous unique frame × the target frametime
   - Kept in a rolling window covering the last 30 minutes of video

4. **Statistical Analysis**
   - Real-time FPS = unique frames in the last second of video
   - Distribution across performance buckets, percentiles and the CDF are computed from the rolling window
   - Frame-to-frame deltas for micro-stutter detection

### Performance Notes

- **CPU-intensive**: OpenCV decodes video in software, on a background thread
- **Recommended**: i7-3770 or better for real-time H.264 @ 1080p
- **Higher resolutions/H.265**: Requires a faster CPU for real-time playback; with `--no-display` it just takes longer
- Statistics are updated in constant time per frame, however long the video

---

## Testing

The project includes a test suite. See [TESTING.md](TESTING.md) for how to run it and what it covers.

---

## Limitations

* The `fps` argument must match the frame rate the video was **recorded at** (the analyser warns if the file says otherwise)
* Dropped frames must be encoded as **duplicate frames** in the video
* Does **not account for screen tearing** - V-Sync should be enabled during recording
* Analysis assumes the game is meant to render a new frame every refresh; repeated frames indicate performance issues
* Scenes with no movement at all (e.g. a paused menu) can be misread, because every frame looks like a repeat
* The histogram and CDF panels appear once more than 10 unique frames have been analysed

---

## Comparison with Original

| Feature | Original | This Fork |
|---------|----------|-----------|
| Python Version | 2.7 | 3.11+ |
| FPS Modes | 60fps only | Any (60, 120, 144, ...) |
| Command-line Args | No | Yes |
| Headless mode | No | Yes |
| Video Output | No | Yes |
| Histogram | No | Yes (5 buckets) |
| CDF Graph | No | Yes |
| Delta Plot | No | Yes (optional) |
| CSV Export | No | Summary + per-frame |
| Statistics | Basic | Average, 1% low, 0.1% low, 99th percentile |
| Rolling Window | N/A | 30 minutes |
| Text Visibility | Basic | Anti-aliased with borders |

---

## License

[GNU Affero General Public License v3.0](LICENSE), the same as the original repository.

---

## Credits

- **Original tool**: [AndrewKeYanzhe/frametime-analyser](https://github.com/AndrewKeYanzhe/frametime-analyser)
- **Inspired by**: Digital Foundry's FPSGui tool
- **Enhanced fork**: [duke748/frametime-analyser](https://github.com/duke748/frametime-analyser)

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests with improvements. Please run the tests first; see [TESTING.md](TESTING.md).
