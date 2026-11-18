# Frametime Analyser (Enhanced Fork)

**Frametime Analyser** detects repeated frames in game footage to work out FPS, frametimes, stutter and tearing. It plays the video back with performance graphs overlaid, saves an annotated copy, and exports the results to CSV and an HTML report. It can also compare two captures side by side, or analyse live from a capture card.

![Frametime Analyser overlay on Red Dead Redemption 2 footage](Media/RDR2.gif)

*Output from the original version of the tool. This fork adds the delta, smoothness (CDF) and histogram panels described below.*

## About This Fork

This is an enhanced fork of the original [Frametime Analyser by AndrewKeYanzhe](https://github.com/AndrewKeYanzhe/frametime-analyser). The original tool was inspired by Digital Foundry's FPSGui tool.

### Major Enhancements Added

- **Python 3**, installable with `pip`, or as a standalone Windows `.exe`
- **Any target frame rate** (60, 120, 144, 240, ...), read from the video automatically
- **Fast**: FFmpeg decoding and encoding (including NVIDIA/Intel/AMD GPU encoders); analysis-only runs at over 2,000 fps on 1080p
- **Side-by-side comparison** of two captures (`--compare`)
- **Live analysis** from a capture card (`--capture`)
- **Tear detection** for V-Sync-off footage (`--detect-tears`, experimental)
- **Stutter detection** for shader compilation and traversal stutters
- **Static scene handling**: pause menus don't wreck the statistics
- **Region of interest / ignore regions** for webcams and chat overlays
- **Real-time graphs**: Frametime, FPS, Frame Delta (micro-stutter), Smoothness (CDF) and a frametime histogram
- **Average FPS, 1% low and 0.1% low FPS** over the last 30 minutes of video
- **Outputs**: annotated video, summary CSV, per-frame CSV and an HTML report with charts

---

## Features

### Real-Time Visual Analysis

1. **Frametime Graph**
   - Shows the frametime in milliseconds of each new frame
   - Scale: 0 to 2.5× target frametime (e.g. 0–41.7ms at 60fps), with a grey reference line at the target frametime
   - Helps identify individual frame drops and stutter events

2. **Frame-to-Frame Delta Plot (Optional)**
   - Detects micro-stutter by plotting the absolute difference between consecutive frametimes
   - Color-coded: Green (smooth) → Yellow (noticeable) → Red (micro-stutter)
   - Reveals frame pacing issues invisible in FPS averages
   - Enable with `--delta-plot`

3. **Cumulative Distribution Function / Smoothness Graph**
   - **What it shows**: The percentage of frames rendered at or faster than a given frametime
   - **X-axis**: Frametime in milliseconds (0 to 3× target frametime); **Y-axis**: Cumulative percentage
   - **Steep curve** = consistent performance; **gradual curve** = frames spread across many timings
   - Reference lines: green at the target frametime, grey at 50% (median) and 95%
   - **Example**: If the curve reaches 95% at the target frametime line, then 95% of your frames hit the target
   - The same metric as CapFrameX's "Smoothness Graph"

4. **FPS Graph (Bottom)**
   - FPS over the last ~10 seconds of video
   - Scaled from 0 to the target FPS, with reference lines at 30, 60, 120 and 240 where they fit

5. **Frametime Distribution Histogram**

     | Bucket | 60fps | 120fps | Other targets (× target frametime) | Color |
     |---|---|---|---|---|
     | **Perfect** | 0–18ms | 0–9ms | 0–1.08× | Green |
     | **Good** | 18–25ms | 9–12ms | 1.08–1.5× | Green |
     | **Fair** | 25–35ms | 12–17ms | 1.5–2.1× | Yellow |
     | **Stutter** | 35–50ms | 17–25ms | 2.1–3× | Orange |
     | **Bad** | 50ms+ | 25ms+ | 3×+ | Red |

   - Bar graph and percentage table, with average frametime, average FPS and 1% low FPS above it

6. **Stutter Detection (Optional)**
   - **Moderate**: ≥2.5× target frametime (≥41.7ms @ 60fps) - orange markers
   - **Severe**: ≥4× target frametime (≥66.7ms @ 60fps) - red markers
   - Marked on the frametime graph, counted on screen, listed in the CSV and report
   - Useful for shader compilation stutters, traversal stutters and texture streaming issues
   - Enable with `--detect-stutters`

7. **Tear Detection (Optional, experimental)**
   - For footage recorded with **V-Sync off**, where the game can swap frames mid-refresh. The analyser compares frames row by row: rows above a tear are unchanged and rows below it are new.
   - Torn frames are timed by where the tear is (a tear halfway down is half a refresh later), so frametimes stay accurate
   - Shows the percentage of torn frames and marks each tear line at the right edge of the video
   - Rows that never change (a static HUD) are learned and ignored; use `--ignore` for anything else static
   - Enable with `--detect-tears`

8. **Static Scenes**
   - If the picture doesn't change for more than 2 seconds (`--static-after`), it's treated as a static scene, such as a pause menu, and left out of the statistics instead of counting as one enormous stutter. Shorter freezes are still counted.
   - Tiny differences below `--min-diff` (compression noise on a still picture) never count as new frames

### Comparing Two Captures

`--compare other.mp4` plays two videos side by side, with each one's FPS and frametime graphs in its own colour, for example PC vs console or two graphics settings. Each video also gets its own CSV files and report. Use `--labels` to name them.

### Live Capture

`--capture 0` analyses straight from a capture card or other video device, until you press `q` or Ctrl+C. Outputs are named `capture_<date>-<time>_...` in the current folder (or `--output-dir`). If processing can't keep up, captured frames are dropped and you're warned; use `--no-video` or `--no-display` to lighten the load.

---

## Setup

### Option 1: Standalone Windows executable

Download `frametime-analyser.exe` from the [Actions artifacts or releases](https://github.com/duke748/frametime-analyser/actions) and run it from a terminal. No Python needed; FFmpeg is included.

### Option 2: Install with pip

Requires **Python 3.11+**.

```bash
git clone https://github.com/duke748/frametime-analyser.git
cd frametime-analyser

python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install .
```

This installs the `frametime-analyser` command, plus NumPy, OpenCV and `imageio-ffmpeg` (which bundles FFmpeg). If you'd rather use your own FFmpeg, any `ffmpeg` on your PATH is used first.

---

## Usage

```bash
frametime-analyser <video_file> [fps] [options]
```

`python -m frametime_analyser ...` and `python "Frametime Analyser.py" ...` also work.

By default the video plays in a window **in real time**. Press **`q`** in the window to stop early; results up to that point are still saved. Use `--no-display` to skip the window and run as fast as possible, and `--no-video` as well if you only need the numbers.

### Options

| Option | Description |
|--------|-------------|
| `video_file` | Video file to analyse |
| `fps` | Frame rate the video was recorded at, 1–1000. Default: read from the video, else 60 |
| **Analysis** | |
| `--detect-stutters` | Detect and mark frametime spikes |
| `--detect-tears` | Detect screen tearing in V-Sync-off footage (experimental) |
| `--roi X,Y,W,H` | Only analyse this region (source video pixels) |
| `--ignore X,Y,W,H` | Ignore this region, e.g. a webcam or chat overlay. Repeatable |
| `--min-diff N` | Smallest difference (0–255) that counts as a new frame. Default 0.1 |
| `--static-after SECONDS` | Repeats longer than this are a static scene, left out of the statistics. Default 2; 0 to always count them |
| **Overlay** | |
| `--delta-plot` | Show the frame-to-frame delta plot |
| `--title <text>` | Title at the top of the video |
| `--watermark <file>` | 32×32 image at 50% opacity |
| **Output** | |
| `--no-display` | No preview window, no real-time playback; prints progress |
| `--no-video` | Don't write the annotated video (much faster) |
| `--no-report` | Don't write the HTML report |
| `--encoder` | `auto` (default: FFmpeg libx264, else OpenCV), `opencv`, `libx264`, `h264_nvenc`/`hevc_nvenc` (NVIDIA), `h264_qsv` (Intel), `h264_amf` (AMD) |
| `--output-dir DIR` | Folder for output files. Default: next to the video |
| **Input** | |
| `--decoder` | `auto` (default: FFmpeg if available), `ffmpeg` or `opencv` |
| `--hw-decode` | Decode on the GPU. Often *slower* than a fast CPU; try it for 4K HEVC on a slower CPU |
| `--compare VIDEO_B` | Compare with a second video side by side |
| `--labels A B` | Names for the two videos in `--compare` |
| `--capture DEVICE` | Analyse live from a capture device (0 = first) |
| `--capture-size WxH` | Resolution to request from the capture device |
| `--fps N` | Same as the `fps` argument (handy with `--capture`) |

The single-dash spellings from earlier versions (`-delta-plot`, `-detect-stutters`, `-watermark`, `-title`) still work, and options can come in any order.

### Examples

```bash
# Basic analysis (frame rate read from the video)
frametime-analyser gameplay.mp4

# Stutter detection and micro-stutter plot, encoded on an NVIDIA GPU
frametime-analyser gameplay.mp4 --detect-stutters --delta-plot --encoder h264_nvenc

# Just the numbers, as fast as possible
frametime-analyser gameplay.mp4 --no-display --no-video --detect-stutters

# A stream recording with a facecam in the top-right corner
frametime-analyser stream.mp4 --ignore 1500,0,420,300

# PC vs console, side by side
frametime-analyser pc.mp4 --compare ps5.mp4 --labels "PC" "PS5" --title "Performance mode"

# V-Sync off footage
frametime-analyser vsync_off.mp4 144 --detect-tears

# Live from the first capture device at 60fps
frametime-analyser --capture 0 --fps 60 --capture-size 1920x1080
```

### Output Files

Written next to the input video (or in `--output-dir`):

1. **`[filename]_analyzed.mp4`** - The video with overlay graphs and statistics, 1280×720 at the target frame rate. Other aspect ratios are letterboxed or pillarboxed. With `--compare` this is `[a]_vs_[b]_compare.mp4`.
2. **`[filename]_report.html`** - A self-contained report: summary figures, frametime and FPS charts for the whole run, the distribution, and the worst stutters
3. **`[filename]_histogram.csv`** - Summary statistics (see below)
4. **`[filename]_frametimes.csv`** - One row per new frame: `unique_frame, video_frame, time_s, frametime_ms, tear_position, after_static`

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
Static Scenes Excluded,0
Static Time Excluded (s),0.0

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

*The stutter section only appears with `--detect-stutters`, and a "Tear Detection Results" section with `--detect-tears`.*

**How the statistics are calculated** (over the last 30 minutes of video, excluding static scenes):
- **Average FPS** = new frames ÷ seconds of video, which equals 1000 ÷ average frametime
- **99th percentile frametime**: 99% of frames were this fast or faster
- **1% low / 0.1% low FPS** = 1000 ÷ the 99th / 99.9th percentile frametime

---

## Technical Overview

### Key Processing Steps

1. **Frame Difference**: each frame is shrunk to 128×72 greyscale (fitted the same way as the 1280×720 canvas, with letterbox bars and ignored regions masked out) and compared with the previous frame
2. **Duplicate Detection**: a frame is a repeat if its difference is below 25% of the recent average difference, or below `--min-diff`
3. **Frametime**: a new frame's frametime is the time since the previous new frame. With tear detection, the tear's position adds a fraction of a refresh.
4. **Statistics**: kept for the last 30 minutes of video; percentiles, the CDF and the histogram are updated in constant time per frame

### Performance

Measured on a fast desktop (RTX 5090), 60-second clips:

| Mode | 1080p H.264 | 4K HEVC |
|---|---|---|
| Analysis only (`--no-display --no-video`) | ~2,750 fps | ~570 fps |
| Annotated video, `--encoder h264_nvenc` | ~470 fps | ~360 fps |
| Annotated video, `--encoder libx264` | ~450 fps | ~310 fps |
| Annotated video, `--decoder opencv --encoder opencv` (no FFmpeg) | ~300 fps | ~80 fps |

How it gets there:
- FFmpeg decodes with every CPU core and scales straight to the size needed. In analysis-only mode it outputs just the 128×72 greyscale image.
- Encoding runs on its own thread, and through FFmpeg it can use the GPU
- The overlay's static parts are drawn once; the histogram, CDF and text are redrawn about 10 times a second; only the scrolling graphs are drawn every frame

---

## Testing

See [TESTING.md](TESTING.md).

---

## Limitations

* The `fps` must match the frame rate the video was **recorded at**. The analyser reads it from the file when you don't give it, and warns if they disagree.
* Dropped frames must be encoded as **duplicate frames** in the video (true of capture cards and most recorders)
* Without `--detect-tears`, V-Sync should be on during recording
* Tear detection is experimental. It needs detail across the screen, and when the game renders faster than the refresh rate (more than one new frame per refresh), it can undercount new frames.
* A frozen picture can't be told apart from a paused one, so freezes longer than `--static-after` are treated as static scenes. Raise the limit (or set it to 0) if you're measuring long hitches.
* `--min-diff` is a fixed threshold. Very subtle motion (a character idling in a dark scene) can fall below it; lower it if new frames are being missed.
* The histogram and CDF panels fill in once more than 10 new frames have been analysed

---

## Comparison with Original

| Feature | Original | This Fork |
|---------|----------|-----------|
| Python Version | 2.7 | 3.11+ (or standalone .exe) |
| FPS Modes | 60fps only | Any, read from the video |
| Command-line Args | No | Yes |
| Speed | Real time only | Up to ~45× real time |
| Video Output | No | Yes (H.264/H.265, GPU encoders) |
| Histogram / CDF / Delta Plot | No | Yes |
| Stutter / Tear Detection | No | Yes |
| Compare / Live Capture | No | Yes |
| CSV Export | No | Summary + per-frame |
| HTML Report | No | Yes |
| Statistics | Basic | Average, 1% low, 0.1% low, 99th percentile |

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

Contributions are welcome! Feel free to open issues or submit pull requests with improvements. Please run the linter and tests first; see [TESTING.md](TESTING.md).
