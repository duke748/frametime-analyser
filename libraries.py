"""Helpers for the frametime analyser: video decoding, frame comparison and statistics.

Everything here is independent of the on-screen overlay, so it can be unit tested.
"""
import math
import time
from bisect import bisect_right
from collections import Counter, deque
from queue import Empty, Queue
from threading import Thread

import cv2
import numpy as np

# Every frame is fitted onto a canvas of this size before analysis, so the
# overlay (drawn at fixed positions) always lines up whatever the source size.
CANVAS_SIZE = (1280, 720)


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_LINEAR):
    """Resize to the given width or height, keeping the aspect ratio."""
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def fit_to_canvas(image, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1]):
    """Scale an image to fit inside width x height, keeping its aspect ratio,
    and centre it on a black canvas of exactly that size (letterbox/pillarbox)."""
    (h, w) = image.shape[:2]
    if (w, h) == (width, height):
        return image
    scale = min(width / w, height / h)
    new_w = max(1, min(width, round(w * scale)))
    new_h = max(1, min(height, round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((height, width) + image.shape[2:], dtype=image.dtype)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def frame_difference(frame, prev_frame, scale=0.1):
    """Mean absolute greyscale difference between two frames.

    Frames are downscaled first, which is much faster and ignores
    compression noise. A repeated frame gives a value close to zero.
    """
    small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    small_prev = cv2.resize(prev_frame, (0, 0), fx=scale, fy=scale)
    small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    small_prev = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(small, small_prev).mean())


class FileVideoStream:
    """Decode a video on a background thread and hand frames to the caller.

    Each frame is passed through `transform` (if given) and then fitted onto
    a `size` canvas (see fit_to_canvas) on the decoding thread.
    """

    def __init__(self, path, transform=None, queue_size=196, size=CANVAS_SIZE):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.transform = transform
        self.size = size
        self.Q = Queue(maxsize=queue_size)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def is_opened(self):
        return self.stream.isOpened()

    def fps(self):
        """Frame rate reported by the container (0 if unknown)."""
        return self.stream.get(cv2.CAP_PROP_FPS) or 0

    def frame_count(self):
        """Number of frames reported by the container (0 if unknown)."""
        return int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            if self.Q.full():
                time.sleep(0.01)  # Queue is full: rest for 10ms
                continue

            grabbed, frame = self.stream.read()
            if not grabbed:
                break  # End of the video

            if self.transform:
                frame = self.transform(frame)
            if self.size:
                frame = fit_to_canvas(frame, *self.size)
            # Only this thread puts, and only when the queue has room,
            # so this never blocks.
            self.Q.put(frame)

        self.stream.release()
        # Set after the last put, so a reader that sees `stopped` and an
        # empty queue knows no more frames are coming.
        self.stopped = True

    def read(self):
        """Return the next frame, or None once the video has ended."""
        while True:
            try:
                return self.Q.get(timeout=0.1)
            except Empty:
                if self.stopped and self.Q.empty():
                    return None

    def running(self):
        """True while frames are queued or still being decoded."""
        return self.more() or not self.stopped

    def more(self):
        """True if a frame is queued. Waits briefly if decoding is still in progress."""
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()


# --- Histogram buckets -------------------------------------------------------

GREEN = (93, 232, 130)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)

# Frametime histogram buckets per target FPS: (min_ms, max_ms, label, BGR colour).
# A frametime falls in a bucket when min_ms <= frametime < max_ms. The last
# bucket is open-ended so long freezes are still counted.
# Ranges are chosen so the ideal frametime (16.67ms / 8.33ms) lands in "Perfect".
HISTOGRAM_BUCKETS = {
    60: [
        (0, 18, "Perfect", GREEN),       # includes ideal 16.67ms
        (18, 25, "Good", GREEN),         # still smooth
        (25, 35, "Fair", YELLOW),        # noticeable
        (35, 50, "Stutter", ORANGE),     # severe
        (50, math.inf, "Bad", RED),      # critical
    ],
    120: [
        (0, 9, "Perfect", GREEN),        # includes ideal 8.33ms
        (9, 12, "Good", GREEN),
        (12, 17, "Fair", YELLOW),
        (17, 25, "Stutter", ORANGE),
        (25, math.inf, "Bad", RED),
    ],
}

# For other target FPS values, bucket edges are these multiples of the target
# frametime. They reproduce the 60fps table exactly.
BUCKET_MULTIPLIERS = (1.08, 1.5, 2.1, 3.0)


def get_histogram_buckets(target_fps):
    """Return the histogram bucket definitions for a target FPS."""
    if target_fps in HISTOGRAM_BUCKETS:
        return HISTOGRAM_BUCKETS[target_fps]
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    target_frametime = 1000.0 / target_fps
    edges = [0] + [round(target_frametime * m, 1) for m in BUCKET_MULTIPLIERS] + [math.inf]
    labels = [b[2] for b in HISTOGRAM_BUCKETS[60]]
    colors = [b[3] for b in HISTOGRAM_BUCKETS[60]]
    return [(edges[i], edges[i + 1], labels[i], colors[i]) for i in range(len(labels))]


def format_bucket_range(min_ms, max_ms):
    """Human-readable bucket range, e.g. '18-25' or '50+'."""
    if math.isinf(max_ms):
        return f"{min_ms:g}+"
    return f"{min_ms:g}-{max_ms:g}"


def categorize_frametime(frametime_ms, buckets):
    """Return the index of the bucket a frametime falls in, or None if outside all buckets."""
    for i, (min_ft, max_ft, _label, _color) in enumerate(buckets):
        if min_ft <= frametime_ms < max_ft:
            return i
    return None


def count_frames_per_bucket(frametimes, buckets):
    """Count how many frametimes fall into each bucket."""
    counts = [0] * len(buckets)
    for ft in frametimes:
        i = categorize_frametime(ft, buckets)
        if i is not None:
            counts[i] += 1
    return counts


def fps_color(avg_fps, target_fps):
    """Colour for an average FPS: green near target, then yellow, orange, red."""
    ratio = avg_fps / target_fps if target_fps else 0
    if ratio >= 0.96:
        return GREEN
    if ratio >= 5 / 6:
        return YELLOW
    if ratio >= 2 / 3:
        return ORANGE
    return RED


# --- Statistics --------------------------------------------------------------

class FpsCounter:
    """Rolling one-second window of frame flags (1 = unique frame, 0 = repeated frame).

    The window holds target_fps samples, so it always spans one second of video
    and the sum of the window is the current FPS.
    """

    def __init__(self, target_fps):
        self.window = deque([0] * target_fps, maxlen=target_fps)
        self.fps = 0

    def record(self, unique):
        """Record one video frame and return the updated FPS."""
        value = 1 if unique else 0
        self.fps += value - self.window[0]
        self.window.append(value)
        return self.fps


class FrametimeWindow:
    """Frametimes covering the most recent `max_duration_ms` of video.

    Frametimes are whole multiples of the target frametime, so there are only
    a handful of distinct values. Keeping a count per value makes the mean,
    percentiles, CDF and bucket counts cheap however long the window is.
    """

    def __init__(self, max_duration_ms):
        self.max_duration_ms = max_duration_ms
        self.values = deque()
        self.counts = Counter()
        self.total_ms = 0.0

    def __len__(self):
        return len(self.values)

    def add(self, frametime_ms):
        self.values.append(frametime_ms)
        self.counts[frametime_ms] += 1
        self.total_ms += frametime_ms
        # Drop the oldest frames once the window holds more than its duration
        while self.total_ms > self.max_duration_ms and len(self.values) > 1:
            old = self.values.popleft()
            self.total_ms -= old
            self.counts[old] -= 1
            if self.counts[old] == 0:
                del self.counts[old]

    def mean(self):
        return self.total_ms / len(self.values) if self.values else 0.0

    def average_fps(self):
        """Unique frames per second of video covered by the window."""
        return 1000.0 * len(self.values) / self.total_ms if self.total_ms > 0 else 0.0

    def _cumulative(self):
        keys = sorted(self.counts)
        running = 0
        cumulative = []
        for k in keys:
            running += self.counts[k]
            cumulative.append(running)
        return keys, cumulative

    def percentile(self, p):
        """Nearest-rank percentile (0-100) of the frametimes, or 0 if empty."""
        n = len(self.values)
        if n == 0:
            return 0.0
        rank = max(1, math.ceil(p / 100.0 * n))
        keys, cumulative = self._cumulative()
        for k, c in zip(keys, cumulative):
            if c >= rank:
                return k
        return keys[-1]

    def low_fps(self, percent):
        """'1% low'-style FPS: 1000 / the (100 - percent)th percentile frametime."""
        ft = self.percentile(100 - percent)
        return 1000.0 / ft if ft > 0 else 0.0

    def cdf(self, thresholds):
        """Percentage of frametimes <= each threshold."""
        n = len(self.values)
        if n == 0:
            return [0.0 for _ in thresholds]
        keys, cumulative = self._cumulative()
        result = []
        for t in thresholds:
            i = bisect_right(keys, t)
            result.append(100.0 * (cumulative[i - 1] if i else 0) / n)
        return result

    def bucket_counts(self, buckets):
        counts = [0] * len(buckets)
        for value, c in self.counts.items():
            i = categorize_frametime(value, buckets)
            if i is not None:
                counts[i] += c
        return counts
