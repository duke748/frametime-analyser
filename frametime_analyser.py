"""Frametime Analyser: detect repeated frames in game footage and measure
frame rate, frametimes and stutter.

Usage: python frametime_analyser.py <video_file> [fps] [options]
Run with --help for the full list of options.
"""
import argparse
import csv
import os
import sys
import time
from collections import deque, namedtuple

import cv2
import numpy as np

from libraries import (
    CANVAS_SIZE,
    GREEN,
    ORANGE,
    RED,
    YELLOW,
    FileVideoStream,
    FpsCounter,
    FrametimeWindow,
    categorize_frametime,
    format_bucket_range,
    fps_color,
    frame_difference,
    get_histogram_buckets,
)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
FONT = cv2.FONT_HERSHEY_DUPLEX

MIN_FPS, MAX_FPS = 1, 1000
DEFAULT_FPS = 60

# A frame whose difference from the previous one is below this fraction of the
# recent average difference is treated as a repeated (dropped) frame.
DUPLICATE_THRESHOLD = 0.25

# Stutters are frametimes of at least these multiples of the target frametime.
STUTTER_MULTIPLIER = 2.5
SEVERE_STUTTER_MULTIPLIER = 4.0

# Averages, percentiles, the CDF and the histogram cover this much video.
WINDOW_MINUTES = 30

FRAMETIME_SAMPLES = 90   # Unique frames kept for the frametime and delta graphs
FPS_GRAPH_SAMPLES = 580  # Video frames kept for the FPS graph

Stutter = namedtuple("Stutter", "unique_index video_frame frametime_ms severity")
OutputPaths = namedtuple("OutputPaths", "video summary_csv frametimes_csv")


# --- Command line ------------------------------------------------------------

def target_fps_type(text):
    try:
        value = int(text)
    except ValueError:
        value = None
    if value is None or not MIN_FPS <= value <= MAX_FPS:
        raise argparse.ArgumentTypeError(
            f"FPS must be a whole number between {MIN_FPS} and {MAX_FPS}, got '{text}'")
    return value


def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="frametime_analyser.py",
        description="Detect repeated frames in game footage and measure frame rate, "
                    "frametimes and stutter.",
        epilog="Single-dash spellings (-delta-plot, -detect-stutters, -watermark, -title) "
               "are also accepted.",
    )
    parser.add_argument("video", help="video file to analyse")
    parser.add_argument("fps", nargs="?", type=target_fps_type, default=DEFAULT_FPS,
                        help=f"frame rate the video was recorded at (default {DEFAULT_FPS})")
    parser.add_argument("--delta-plot", "-delta-plot", action="store_true",
                        help="show the frame-to-frame delta plot (micro-stutter)")
    parser.add_argument("--detect-stutters", "-detect-stutters", action="store_true",
                        help="detect and mark frametime spikes (shader/traversal stutter)")
    parser.add_argument("--watermark", "-watermark", metavar="IMAGE",
                        help="image to show at 32x32, 50%% opacity, bottom left")
    parser.add_argument("--title", "-title", metavar="TEXT",
                        help="title shown at the top of the video")
    parser.add_argument("--no-display", action="store_true",
                        help="don't open a preview window or play in real time; "
                             "analyse as fast as possible")
    return parser


def parse_args(argv=None):
    parser = build_arg_parser()
    # Intermixed parsing lets the fps value come after options, e.g. "video -delta-plot 120"
    args = parser.parse_intermixed_args(argv)
    if not os.path.isfile(args.video):
        parser.error(f"video file '{args.video}' not found")
    if args.watermark and not os.path.isfile(args.watermark):
        parser.error(f"watermark file '{args.watermark}' not found")
    return args


def output_paths(video_path):
    base = os.path.splitext(video_path)[0]
    return OutputPaths(
        video=base + "_analyzed.mp4",
        summary_csv=base + "_histogram.csv",
        frametimes_csv=base + "_frametimes.csv",
    )


# --- Analysis ----------------------------------------------------------------

class FrametimeAnalyser:
    """Turns a stream of frame differences into frametimes and statistics.

    Call process() once per video frame (after the first) with the difference
    between that frame and the one before it.
    """

    def __init__(self, target_fps, detect_stutters=False, window_minutes=WINDOW_MINUTES):
        self.target_fps = target_fps
        self.target_frametime = 1000.0 / target_fps
        self.detect_stutters = detect_stutters
        self.moderate_threshold = self.target_frametime * STUTTER_MULTIPLIER
        self.severe_threshold = self.target_frametime * SEVERE_STUTTER_MULTIPLIER

        self.frames_processed = 0   # Video frames processed (frame 0 is only a reference)
        self.unique_frames = 0
        self.frames_since_unique = 0
        self.moving_avg_diff = None

        self.fps_counter = FpsCounter(target_fps)
        self.fps = 0
        self.frametime_graph = deque([0.0] * FRAMETIME_SAMPLES, maxlen=FRAMETIME_SAMPLES)
        self.delta_graph = deque([0.0] * FRAMETIME_SAMPLES, maxlen=FRAMETIME_SAMPLES)
        self.fps_graph = deque([0] * FPS_GRAPH_SAMPLES, maxlen=FPS_GRAPH_SAMPLES)
        self.window = FrametimeWindow(window_minutes * 60 * 1000)

        self.stutters = []
        self.severe_stutters = 0
        self.moderate_stutters = 0

    @property
    def video_frame(self):
        """Index of the most recently processed video frame."""
        return self.frames_processed

    def process(self, diff):
        """Record one video frame. Returns its frametime in ms if it is a new
        (unique) frame, or None if it repeats the previous frame."""
        self.frames_processed += 1
        self.frames_since_unique += 1
        if self.moving_avg_diff is None:
            self.moving_avg_diff = diff

        unique = diff >= DUPLICATE_THRESHOLD * self.moving_avg_diff
        self.moving_avg_diff = 0.75 * self.moving_avg_diff + 0.25 * diff

        frametime = None
        if unique:
            frametime = self.frames_since_unique * self.target_frametime
            previous = self.frametime_graph[-1] or frametime
            self.delta_graph.append(abs(frametime - previous))
            self.frametime_graph.append(frametime)
            self.window.add(frametime)
            if self.detect_stutters:
                self._check_stutter(frametime)
            self.unique_frames += 1
            self.frames_since_unique = 0

        self.fps = self.fps_counter.record(unique)
        self.fps_graph.append(self.fps)
        return frametime

    def _check_stutter(self, frametime):
        if frametime >= self.severe_threshold:
            severity = "severe"
            self.severe_stutters += 1
        elif frametime >= self.moderate_threshold:
            severity = "moderate"
            self.moderate_stutters += 1
        else:
            return
        self.stutters.append(Stutter(self.unique_frames, self.video_frame, frametime, severity))

    def summary(self):
        w = self.window
        return {
            "duration_s": self.frames_processed / self.target_fps,
            "unique_frames": self.unique_frames,
            "window_frames": len(w),
            "avg_frametime": w.mean(),
            "avg_fps": w.average_fps(),
            "p99_frametime": w.percentile(99),
            "low_1_fps": w.low_fps(1),
            "low_01_fps": w.low_fps(0.1),
        }


# --- Drawing -----------------------------------------------------------------

def draw_text(img, text, org, scale, color=WHITE, thickness=1, border=1):
    """Anti-aliased text with a black outline."""
    # The outline is the text drawn in black at 8 offsets. (A single thicker
    # black pass doesn't work: in OpenCV thickness also widens the glyphs.)
    x, y = org
    for dx in (-border, 0, border):
        for dy in (-border, 0, border):
            if dx or dy:
                cv2.putText(img, text, (x + dx, y + dy), FONT, scale, BLACK, thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


PT_WIDTH = 3    # Width of one sample on the frametime/delta graphs
PT_SPACING = 3  # Distance between the starts of two samples
GRAPH_HEIGHT = 100
GRAPH_BOTTOM = 420


def draw_line_graph(img, values, valid, x0, width, max_value, color_for):
    """Step graph of the newest values, scaled so max_value reaches the top.
    `valid` is how many of the newest values are real data (the rest are padding)."""
    y0 = GRAPH_BOTTOM
    top = y0 - GRAPH_HEIGHT
    n = width // PT_SPACING
    values = list(values)[-n:]
    first = len(values) - min(valid, len(values))
    scale = GRAPH_HEIGHT / max_value
    for key in range(max(1, first + 1), len(values)):
        y = int(max(top, y0 - min(values[key], max_value) * scale))
        y_prev = int(max(top, y0 - min(values[key - 1], max_value) * scale))
        x = x0 + key * PT_SPACING
        color = color_for(values[key])
        cv2.line(img, (x, y), (x + PT_WIDTH, y), color, 2)
        cv2.line(img, (x, y_prev), (x, y), color, 2)
    cv2.rectangle(img, (x0, top), (x0 + width, y0), WHITE, 2)
    return len(values)


class Overlay:
    """Draws the graphs and statistics onto each frame."""

    def __init__(self, target_fps, show_delta_plot=False, detect_stutters=False,
                 title=None, watermark=None):
        self.target_fps = target_fps
        self.target_frametime = 1000.0 / target_fps
        self.show_delta_plot = show_delta_plot
        self.detect_stutters = detect_stutters
        self.title = title
        self.watermark = watermark
        self.buckets = get_histogram_buckets(target_fps)

        self.fps_text = ""
        self.fps_text_timer = 0
        self.fps_text_interval = max(1, target_fps // 2)  # Update every half second

        # Graph columns. With the delta plot on, three graphs share the space
        # left of the histogram, so they are narrower.
        if show_delta_plot:
            self.graph_width = 240
            self.frametime_x, self.delta_x, self.cdf_x = 40, 345, 650
        else:
            self.graph_width = 270
            self.frametime_x, self.delta_x, self.cdf_x = 60, None, 410
        self.hist_x = 1000

    def draw(self, img, a):
        """Draw everything for analyser `a` onto img (a CANVAS_SIZE frame)."""
        self._draw_fps_text(img, a)
        self._draw_frametime_graph(img, a)
        if self.show_delta_plot:
            self._draw_delta_graph(img, a)
        if len(a.window) > 10:
            self._draw_cdf(img, a)
            self._draw_histogram(img, a)
        self._draw_fps_graph(img, a)
        if self.title:
            self._draw_title(img)
        if self.watermark is not None:
            self._draw_watermark(img)

    def _draw_fps_text(self, img, a):
        # Wait until the one-second FPS window is full, then update every half second
        if a.frames_processed < self.target_fps:
            self.fps_text = ""
        else:
            self.fps_text_timer += 1
            if not self.fps_text or self.fps_text_timer >= self.fps_text_interval:
                self.fps_text = str(a.fps)
                self.fps_text_timer = 0
        if self.fps_text:
            (w, _), _ = cv2.getTextSize(self.fps_text, FONT, 2, 5)
            draw_text(img, self.fps_text, (1240 - w, 150), 2, thickness=5, border=2)

    def _draw_frametime_graph(self, img, a):
        x0, width = self.frametime_x, self.graph_width
        top = GRAPH_BOTTOM - GRAPH_HEIGHT
        max_ft = self.target_frametime * 2.5
        valid = min(a.unique_frames, FRAMETIME_SAMPLES)
        n = draw_line_graph(img, a.frametime_graph, valid, x0, width, max_ft, lambda v: GREEN)

        # Target frametime reference line
        target_y = int(GRAPH_BOTTOM - self.target_frametime * GRAPH_HEIGHT / max_ft)
        cv2.line(img, (x0, target_y), (x0 + width, target_y), GREY, 1)

        if self.detect_stutters:
            newest = a.unique_frames - 1
            for s in reversed(a.stutters):
                pos = n - 1 - (newest - s.unique_index)
                if pos < 1:
                    break  # This and all older stutters have scrolled off the graph
                x = x0 + pos * PT_SPACING + 1
                color, thickness = (RED, 3) if s.severity == "severe" else (ORANGE, 2)
                cv2.line(img, (x, top), (x, GRAPH_BOTTOM), color, thickness)

        label_x = x0 + width + 8
        draw_text(img, "0", (label_x, GRAPH_BOTTOM + 4), 0.4)
        draw_text(img, f"{self.target_frametime:.1f}", (label_x, target_y + 4), 0.4)
        draw_text(img, f"{max_ft:.1f}", (label_x, top + 8), 0.4)
        draw_text(img, "FRAMETIME (MS)", (x0, top - 15), 0.45)

    def _draw_delta_graph(self, img, a):
        x0, width = self.delta_x, self.graph_width
        top = GRAPH_BOTTOM - GRAPH_HEIGHT
        max_delta = self.target_frametime * 1.5
        t = self.target_frametime

        def color_for(v):
            if v < t * 0.5:
                return GREEN   # smooth
            if v < t:
                return YELLOW  # noticeable
            return RED         # micro-stutter

        valid = min(a.unique_frames, FRAMETIME_SAMPLES)
        draw_line_graph(img, a.delta_graph, valid, x0, width, max_delta, color_for)
        label_x = x0 + width + 8
        draw_text(img, "0", (label_x, GRAPH_BOTTOM + 4), 0.4)
        draw_text(img, f"{max_delta:.1f}", (label_x, top + 8), 0.4)
        draw_text(img, "FRAME DELTA (MS)", (x0, top - 15), 0.45)

    def _draw_cdf(self, img, a):
        x0, width = self.cdf_x, self.graph_width
        y0, height = GRAPH_BOTTOM, GRAPH_HEIGHT
        top = y0 - height
        max_ft = self.target_frametime * 3
        samples = 100
        thresholds = [max_ft * i / samples for i in range(samples + 1)]
        percentages = a.window.cdf(thresholds)

        points = [(x0 + int(ft / max_ft * width), y0 - int(pct / 100 * height))
                  for ft, pct in zip(thresholds, percentages)]
        for i in range(1, len(points)):
            pct = percentages[i]
            color = GREEN if pct >= 90 else YELLOW if pct >= 70 else ORANGE
            cv2.line(img, points[i - 1], points[i], color, 2)

        median_y = y0 - int(0.5 * height)
        p95_y = y0 - int(0.95 * height)
        cv2.line(img, (x0, median_y), (x0 + width, median_y), GREY, 1)
        cv2.line(img, (x0, p95_y), (x0 + width, p95_y), GREY, 1)
        cv2.rectangle(img, (x0, top), (x0 + width, y0), WHITE, 2)
        target_x = x0 + int(self.target_frametime / max_ft * width)
        cv2.line(img, (target_x, top), (target_x, y0), GREEN, 1)

        label_x = x0 + width + 8
        draw_text(img, "0%", (label_x, y0 - 5), 0.3)
        draw_text(img, "50%", (label_x, median_y + 5), 0.3)
        draw_text(img, "100%", (label_x, top + 5), 0.3)
        draw_text(img, "0", (x0 - 5, y0 + 15), 0.3)
        draw_text(img, f"{self.target_frametime:.0f}", (target_x - 8, y0 + 15), 0.3, GREEN)
        draw_text(img, f"{max_ft:.0f}", (x0 + width - 15, y0 + 15), 0.3)
        draw_text(img, "SMOOTHNESS (CDF)", (x0, top - 15), 0.45)

    def _draw_histogram(self, img, a):
        x0, y0, width, height = self.hist_x, GRAPH_BOTTOM, 270, GRAPH_HEIGHT
        top = y0 - height
        buckets = self.buckets
        counts = a.window.bucket_counts(buckets)
        total = sum(counts)
        percentages = [(c / total * 100) if total else 0 for c in counts]

        cv2.rectangle(img, (x0, top), (x0 + width, y0), WHITE, 2)

        avg_ft = a.window.mean()
        avg_bucket = categorize_frametime(avg_ft, buckets)
        avg_ft_color = buckets[avg_bucket][3] if avg_bucket is not None else WHITE
        avg_fps = a.window.average_fps()
        low_fps = a.window.low_fps(1)

        draw_text(img, "FRAMETIME DISTRIBUTION", (x0, top - 55), 0.45)
        draw_text(img, f"Frametime Avg: {avg_ft:.1f}ms", (x0, top - 40), 0.35, avg_ft_color)
        draw_text(img, f"FPS Avg: {avg_fps:.1f}  1% Low: {low_fps:.1f}", (x0, top - 25), 0.35,
                  fps_color(avg_fps, self.target_fps))

        if self.detect_stutters and a.stutters:
            color = RED if a.severe_stutters else ORANGE
            text = (f"Stutters: {len(a.stutters)} ({a.severe_stutters} severe, "
                    f"{a.moderate_stutters} moderate)")
            draw_text(img, text, (x0, top - 10), 0.35, color)

        bar_width = width // len(buckets)
        for i, (pct, (lo, hi, label, color)) in enumerate(zip(percentages, buckets)):
            bar_x = x0 + i * bar_width
            if pct > 0:
                bar_y = y0 - int(pct / 100 * height)
                cv2.rectangle(img, (bar_x + 2, bar_y), (bar_x + bar_width - 2, y0 - 2), color, -1)
                if pct > 5:
                    text = f"{pct:.0f}%"
                    (tw, _), _ = cv2.getTextSize(text, FONT, 0.3, 1)
                    draw_text(img, text, (bar_x + (bar_width - tw) // 2, bar_y + 15), 0.3)

            label_x = bar_x + 3
            cv2.rectangle(img, (label_x, y0 + 15), (label_x + 6, y0 + 21), color, -1)
            draw_text(img, label, (label_x + 10, y0 + 21), 0.28)
            draw_text(img, format_bucket_range(lo, hi), (label_x, y0 + 30), 0.25)
            draw_text(img, f"{pct:.1f}%", (label_x, y0 + 45), 0.28)

    def _draw_fps_graph(self, img, a):
        x0, y0 = 60, 500
        height = 160
        step = 2  # Pixels per video frame
        width = FPS_GRAPH_SAMPLES * step
        max_fps = self.target_fps
        scale = (height - 1) / max_fps  # Leave 1px at the top so the line is visible at peak

        def y_for(fps):
            return int(y0 + height - max(0, min(fps, max_fps)) * scale)

        values = list(a.fps_graph)
        # Only draw samples recorded after the one-second FPS window filled
        first = FPS_GRAPH_SAMPLES - a.frames_processed + self.target_fps
        for key in range(max(1, first), len(values)):
            cv2.line(img, (x0 + key * step, y_for(values[key - 1])),
                     (x0 + (key + 1) * step, y_for(values[key])), GREEN, 2)
        cv2.rectangle(img, (x0, y0), (x0 + width, y0 + height), WHITE, 2)

        label_x = x0 + width + 10
        for ref in (30, 60, 120, 240):
            if ref < max_fps:
                y = y_for(ref)
                cv2.line(img, (x0, y), (x0 + width, y), WHITE, 1)
                draw_text(img, str(ref), (label_x, y + 5), 0.4)
        draw_text(img, str(max_fps), (label_x, y0 + 5), 0.4)
        draw_text(img, "0", (label_x, y0 + height + 4), 0.4)
        draw_text(img, "FRAME-RATE (FPS)", (x0 + width - 150, y0 - 15), 0.5)

    def _draw_title(self, img):
        w = img.shape[1]
        (tw, th), _ = cv2.getTextSize(self.title, FONT, 1.0, 2)
        x, y, pad = (w - tw) // 2, 40, 10
        cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + pad), BLACK, -1)
        draw_text(img, self.title, (x, y), 1.0, thickness=2)

    def _draw_watermark(self, img):
        wm = self.watermark
        h = img.shape[0]
        wm_h, wm_w = wm.shape[:2]
        x, y = 10, h - wm_h - 10
        region = img[y:y + wm_h, x:x + wm_w].astype(np.float32)
        if wm.shape[2] == 4:
            alpha = wm[:, :, 3:4].astype(np.float32) / 255.0 * 0.5
            blended = region * (1 - alpha) + wm[:, :, :3].astype(np.float32) * alpha
        else:
            blended = region * 0.5 + wm.astype(np.float32) * 0.5
        img[y:y + wm_h, x:x + wm_w] = blended.astype(np.uint8)


def load_watermark(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Warning: could not load watermark from {path}")
        return None
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.resize(image, (32, 32))


# --- Output ------------------------------------------------------------------

def export_summary_csv(path, video_path, a):
    """Write the summary statistics and histogram buckets to a CSV file."""
    s = a.summary()
    buckets = get_histogram_buckets(a.target_fps)
    counts = a.window.bucket_counts(buckets)
    total = sum(counts)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frametime Analysis Results"])
        writer.writerow(["Video File", video_path])
        writer.writerow(["Target FPS", a.target_fps])
        writer.writerow(["Analysis Duration",
                         f"{s['duration_s']:.1f} seconds ({s['duration_s'] / 60:.2f} minutes)"])
        writer.writerow(["Total Unique Frames Analyzed", s["unique_frames"]])
        writer.writerow(["Statistics Window", f"last {WINDOW_MINUTES} minutes of video"])
        writer.writerow(["Unique Frames in Window", s["window_frames"]])
        writer.writerow(["Average Frametime (ms)", f"{s['avg_frametime']:.2f}"])
        writer.writerow(["Average FPS", f"{s['avg_fps']:.2f}"])
        writer.writerow(["99th Percentile Frametime (ms)", f"{s['p99_frametime']:.2f}"])
        writer.writerow(["1% Low FPS", f"{s['low_1_fps']:.2f}"])
        writer.writerow(["0.1% Low FPS", f"{s['low_01_fps']:.2f}"])
        writer.writerow([])

        if a.detect_stutters:
            writer.writerow(["Stutter Detection Results"])
            writer.writerow(["Total Stutters Detected", len(a.stutters)])
            writer.writerow([f"Severe Stutters (>= {SEVERE_STUTTER_MULTIPLIER:g}x target frametime)",
                             a.severe_stutters])
            writer.writerow([f"Moderate Stutters (>= {STUTTER_MULTIPLIER:g}x target frametime)",
                             a.moderate_stutters])
            writer.writerow(["Severe Threshold (ms)", f"{a.severe_threshold:.1f}"])
            writer.writerow(["Moderate Threshold (ms)", f"{a.moderate_threshold:.1f}"])
            if a.stutters:
                stutter_fts = [st.frametime_ms for st in a.stutters]
                writer.writerow(["Average Stutter Frametime (ms)",
                                 f"{sum(stutter_fts) / len(stutter_fts):.1f}"])
                writer.writerow(["Worst Stutter Frametime (ms)", f"{max(stutter_fts):.1f}"])
            writer.writerow([])

        writer.writerow(["Bucket", "Range (ms)", "Frame Count", "Percentage"])
        for (lo, hi, label, _), count in zip(buckets, counts):
            pct = count / total * 100 if total else 0
            writer.writerow([label, format_bucket_range(lo, hi), count, f"{pct:.2f}%"])


class Pacer:
    """Sleeps so frames are shown at the target frame rate."""

    def __init__(self, fps):
        self.interval = 1.0 / fps
        self.deadline = None

    def wait(self):
        now = time.perf_counter()
        if self.deadline is None or now - self.deadline > self.interval:
            self.deadline = now  # First frame, or running behind: don't try to catch up
        elif self.deadline > now:
            time.sleep(self.deadline - now)
        self.deadline += self.interval


# --- Main --------------------------------------------------------------------

def run(args):
    target_fps = args.fps
    paths = output_paths(args.video)

    print(f"Loading video: {args.video}")
    print(f"Target FPS: {target_fps}")
    print(f"Delta Plot: {'Enabled' if args.delta_plot else 'Disabled'}")
    print(f"Watermark: {args.watermark or 'Disabled'}")
    print(f"Title: {args.title or 'Disabled'}")
    print(f"Stutter Detection: {'Enabled' if args.detect_stutters else 'Disabled'}")

    fvs = FileVideoStream(args.video)
    if not fvs.is_opened():
        print(f"Error: could not open video '{args.video}'", file=sys.stderr)
        return 1

    video_fps = fvs.fps()
    if video_fps and abs(video_fps - target_fps) > 0.5:
        print(f"Warning: the video is {video_fps:.2f} fps but the target is {target_fps} fps. "
              f"Results are only meaningful when the video was recorded at the target frame rate.")
    total_frames = fvs.frame_count()

    watermark = load_watermark(args.watermark) if args.watermark else None
    analyser = FrametimeAnalyser(target_fps, detect_stutters=args.detect_stutters)
    overlay = Overlay(target_fps, args.delta_plot, args.detect_stutters, args.title, watermark)

    writer = cv2.VideoWriter(paths.video, cv2.VideoWriter_fourcc(*"mp4v"),
                             float(target_fps), CANVAS_SIZE)
    if not writer.isOpened():
        print(f"Warning: could not create output video '{paths.video}'")
        writer = None
    else:
        print(f"Output will be saved to: {paths.video}")

    display = not args.no_display
    pacer = Pacer(target_fps) if display else None
    window_name = "Frametime Analyser"

    fvs.start()
    prev = fvs.read()
    if prev is None:
        print("Error: the video contains no frames", file=sys.stderr)
        fvs.stop()
        if writer:
            writer.release()
        return 1

    frametimes_file = open(paths.frametimes_csv, "w", newline="")
    frametimes_csv = csv.writer(frametimes_file)
    frametimes_csv.writerow(["unique_frame", "video_frame", "time_s", "frametime_ms"])

    start = time.perf_counter()
    last_progress = start
    try:
        while True:
            frame = fvs.read()
            if frame is None:
                break

            frametime = analyser.process(frame_difference(frame, prev))
            if frametime is not None:
                frametimes_csv.writerow([
                    analyser.unique_frames - 1, analyser.video_frame,
                    f"{analyser.video_frame / target_fps:.4f}", f"{frametime:.3f}",
                ])

            annotated = frame.copy()
            overlay.draw(annotated, analyser)
            prev = frame

            if writer:
                writer.write(annotated)

            if display:
                pacer.wait()
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped early (q pressed)")
                    break
            elif time.perf_counter() - last_progress >= 5:
                last_progress = time.perf_counter()
                done = analyser.frames_processed
                pct = f" ({done / total_frames:.0%})" if total_frames else ""
                print(f"Processed {done} frames{pct}")
    except KeyboardInterrupt:
        print("Interrupted; saving results so far")
    finally:
        fvs.stop()
        frametimes_file.close()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    print(f"Processed {analyser.frames_processed} frames in {elapsed:.1f}s")
    if writer:
        print(f"Video saved to: {paths.video}")
    print(f"Per-frame frametimes exported to: {paths.frametimes_csv}")

    if analyser.unique_frames:
        try:
            export_summary_csv(paths.summary_csv, args.video, analyser)
            print(f"Histogram data exported to: {paths.summary_csv}")
        except OSError as e:
            print(f"Warning: could not export CSV: {e}")
    return 0


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
