"""Frame-rate statistics: histogram buckets, the rolling FPS counter and the
rolling frametime window."""
import math
from bisect import bisect_right
from collections import Counter, deque

# BGR colours shared by the overlay and the histogram
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


class FpsCounter:
    """Rolling one-second window of new-frame counts, one entry per video frame.

    The window holds target_fps entries, so it always spans one second of video
    and its sum is the current FPS. An entry is normally 0 (repeated frame) or
    1 (new frame), but can be higher when tearing shows several new frames in
    one refresh.
    """

    def __init__(self, target_fps):
        self.window = deque([0] * target_fps, maxlen=target_fps)
        self.fps = 0

    def record(self, new_frames):
        """Record one video frame (a bool or a count of new frames) and return the FPS."""
        value = int(new_frames)
        self.fps += value - self.window[0]
        self.window.append(value)
        return self.fps


class FrametimeWindow:
    """Frametimes covering the most recent `max_duration_ms` of video.

    Frametimes take only a limited set of values (multiples of the target
    frametime, or of a fraction of it when tearing). Keeping a count per value
    makes the mean, percentiles, CDF and bucket counts cheap however long the
    window is.
    """

    def __init__(self, max_duration_ms):
        self.max_duration_ms = max_duration_ms
        self.values = deque()
        self.counts = Counter()
        self.total_ms = 0.0
        self._sorted = None  # Cache of (keys, cumulative counts)

    def __len__(self):
        return len(self.values)

    def add(self, frametime_ms):
        self.values.append(frametime_ms)
        self.counts[frametime_ms] += 1
        self.total_ms += frametime_ms
        self._sorted = None
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
        """New frames per second of video covered by the window."""
        return 1000.0 * len(self.values) / self.total_ms if self.total_ms > 0 else 0.0

    def _cumulative(self):
        if self._sorted is None:
            keys = sorted(self.counts)
            running = 0
            cumulative = []
            for k in keys:
                running += self.counts[k]
                cumulative.append(running)
            self._sorted = (keys, cumulative)
        return self._sorted

    def percentile(self, p):
        """Nearest-rank percentile (0-100) of the frametimes, or 0 if empty."""
        n = len(self.values)
        if n == 0:
            return 0.0
        rank = max(1, math.ceil(p / 100.0 * n))
        keys, cumulative = self._cumulative()
        i = bisect_right(cumulative, rank - 1)
        return keys[min(i, len(keys) - 1)]

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
