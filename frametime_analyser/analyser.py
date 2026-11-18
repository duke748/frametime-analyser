"""Turns frame differences into frametimes and statistics."""
from collections import deque, namedtuple

import numpy as np

from .detection import TearDetector
from .stats import FpsCounter, FrametimeWindow

# A frame is a repeat when its change from the previous frame is below this
# fraction of the recent average change...
DUPLICATE_THRESHOLD = 0.25
# ...or below this floor, so a still scene's compression noise isn't mistaken
# for new frames. It applies to the peak change (see detection.frame_change),
# 0-255; compression noise is typically 0.3-0.7.
DEFAULT_MIN_DIFF = 0.75
# Floor for per-row mean differences in tear detection
ROW_MIN_DIFF = 0.1
# Repeats lasting longer than this are a static scene (pause menu, paused
# video) rather than a freeze, and are left out of the statistics.
DEFAULT_STATIC_AFTER = 2.0

# Stutters are frametimes of at least these multiples of the target frametime.
STUTTER_MULTIPLIER = 2.5
SEVERE_STUTTER_MULTIPLIER = 4.0

WINDOW_MINUTES = 30      # Statistics cover this much video
FRAMETIME_SAMPLES = 90   # New frames kept for the frametime and delta graphs
FPS_GRAPH_SAMPLES = 580  # Video frames kept for the FPS graph

# graph_index: position in the frametime graph (counts only frames that are graphed)
Stutter = namedtuple("Stutter", "unique_index video_frame frametime_ms severity graph_index")

# One new frame. `tear` is where on screen it started (0-1) if it tore, else None.
# `after_static` frames end a static scene; their frametime is not counted.
NewFrame = namedtuple("NewFrame", "index video_frame time_ms frametime_ms tear after_static")


class FrametimeAnalyser:
    """Call process() once per video frame (after the first) with the
    difference between that frame and the one before it."""

    def __init__(self, target_fps, detect_stutters=False, detect_tears=False,
                 picture_rows=None, min_diff=DEFAULT_MIN_DIFF,
                 static_after=DEFAULT_STATIC_AFTER, window_minutes=WINDOW_MINUTES,
                 frametime_samples=FRAMETIME_SAMPLES):
        self.target_fps = target_fps
        self.target_frametime = 1000.0 / target_fps
        self.detect_stutters = detect_stutters
        self.min_diff = min_diff
        self.static_after_ms = static_after * 1000 if static_after else None
        self.moderate_threshold = self.target_frametime * STUTTER_MULTIPLIER
        self.severe_threshold = self.target_frametime * SEVERE_STUTTER_MULTIPLIER
        from .detection import ANALYSIS_SIZE
        rows = picture_rows or (0, ANALYSIS_SIZE[1])
        self.tear_detector = TearDetector(rows) if detect_tears else None
        # Without tear detection, still watch for tearing so the user can be
        # warned: it makes every torn frame count as new, overstating FPS
        self.tear_probe = None if detect_tears else TearDetector(rows)

        self.frames_processed = 0   # Video frames processed (frame 0 is only a reference)
        self.unique_frames = 0      # New frames seen, including any after a static scene
        self.moving_avg_diff = None  # Of the mean change (tear detection rows)
        self.moving_avg_peak = None  # Of the peak change (new frame or repeat)
        self.last_new_refresh = 0   # When the latest new frame started, in refreshes
        self.in_static = False
        self.static_scenes = 0
        self.static_ms = 0.0
        self.torn_frames = 0
        self.last_new_frames = []   # New frames found by the latest process() call

        self.fps_counter = FpsCounter(target_fps)
        self.fps = 0
        self.frametime_graph = deque([0.0] * frametime_samples, maxlen=frametime_samples)
        self.delta_graph = deque([0.0] * frametime_samples, maxlen=frametime_samples)
        self.graph_samples = 0      # Real samples in the frametime/delta graphs
        self.fps_graph = deque([0] * FPS_GRAPH_SAMPLES, maxlen=FPS_GRAPH_SAMPLES)
        self.window = FrametimeWindow(window_minutes * 60 * 1000)

        self.stutters = []
        self.severe_stutters = 0
        self.moderate_stutters = 0

    @property
    def video_frame(self):
        """Index of the most recently processed video frame."""
        return self.frames_processed

    def tearing_share(self):
        """With tear detection off: the share of video frames that completed a
        tear from the frame before (only tearing does that; a static sky above
        a moving foreground doesn't)."""
        if self.tear_probe is None or not self.frames_processed:
            return 0.0
        return self.tear_probe.catch_ups / self.frames_processed

    def looks_torn(self):
        """True if tear detection is off but the video shows a lot of tearing
        (V-Sync off), so the FPS is overstated."""
        return self.frames_processed >= 120 and self.tearing_share() >= 0.1

    @property
    def counted_frames(self):
        """New frames whose frametime counts towards the statistics."""
        return self.unique_frames - self.static_scenes

    def process(self, diff, rows=None, peak=None):
        """Record one video frame.

        diff: mean change from the previous frame. peak: peak change (see
        detection.frame_change); when given, it decides new frame vs repeat,
        else diff does. rows: per-row changes, needed for tear detection.
        Returns the list of new frames this video frame contains (usually zero
        or one)."""
        self.frames_processed += 1
        k = self.frames_processed
        if self.moving_avg_diff is None:
            self.moving_avg_diff = diff
        level = diff if peak is None else peak
        if self.moving_avg_peak is None:
            self.moving_avg_peak = level

        positions = None
        if rows is not None:
            row_threshold = max(ROW_MIN_DIFF, DUPLICATE_THRESHOLD * self.moving_avg_diff)
            if self.tear_detector is not None:
                positions = self.tear_detector.detect(rows, row_threshold)
            elif self.tear_probe is not None:
                self.tear_probe.detect(rows, row_threshold)
        if positions is None:  # Whole-frame rule
            positions = [0.0] if level >= max(self.min_diff, DUPLICATE_THRESHOLD * self.moving_avg_peak) else []
        self.moving_avg_diff = 0.75 * self.moving_avg_diff + 0.25 * diff
        self.moving_avg_peak = 0.75 * self.moving_avg_peak + 0.25 * level

        new_frames = []
        for pos in positions:
            # Work in refreshes so a frame N refreshes long is exactly N x the
            # target frametime (no rounding error at the stutter/bucket edges)
            time_ms = (k + pos) * self.target_frametime
            frametime = (k + pos - self.last_new_refresh) * self.target_frametime
            if pos or self.last_new_refresh % 1:
                frametime = round(frametime, 3)  # Keeps the set of distinct values small
            if pos:
                self.torn_frames += 1
            after_static = self.static_after_ms is not None and frametime > self.static_after_ms
            if after_static:
                self.static_scenes += 1
                self.static_ms += frametime
            else:
                self._record(frametime)
            new_frames.append(NewFrame(self.unique_frames, k, time_ms, frametime,
                                       pos if pos else None, after_static))
            self.unique_frames += 1
            self.last_new_refresh = k + pos

        self.in_static = (not new_frames and self.static_after_ms is not None
                          and (k - self.last_new_refresh) * self.target_frametime > self.static_after_ms)
        self.fps = self.fps_counter.record(len(new_frames))
        self.fps_graph.append(self.fps)
        self.last_new_frames = new_frames
        return new_frames

    def _record(self, frametime):
        previous = self.frametime_graph[-1] if self.graph_samples else frametime
        self.delta_graph.append(abs(frametime - previous))
        self.frametime_graph.append(frametime)
        self.graph_samples += 1
        self.window.add(frametime)
        if self.detect_stutters:
            self._check_stutter(frametime)

    def _check_stutter(self, frametime):
        if frametime >= self.severe_threshold:
            severity = "severe"
            self.severe_stutters += 1
        elif frametime >= self.moderate_threshold:
            severity = "moderate"
            self.moderate_stutters += 1
        else:
            return
        self.stutters.append(Stutter(self.unique_frames, self.video_frame, frametime, severity,
                                    self.graph_samples - 1))

    def fps_graph_array(self):
        return np.fromiter(self.fps_graph, dtype=np.float64, count=len(self.fps_graph))

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
            "torn_frames": self.torn_frames,
            "torn_pct": 100.0 * self.torn_frames / self.unique_frames if self.unique_frames else 0.0,
            "static_scenes": self.static_scenes,
            "static_s": self.static_ms / 1000.0,
        }
