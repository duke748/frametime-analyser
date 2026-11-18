"""Synthetic test videos with known frame timing."""
import math
import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from frametime_analyser.video import find_ffmpeg

HAS_FFMPEG = find_ffmpeg() is not None
needs_ffmpeg = unittest.skipUnless(HAS_FFMPEG, "ffmpeg not available")


def gray_level(pid):
    return 30 + (pid * 37) % 200


def write_video(path, frame_ids, fps=60, size=(320, 180), overlay=None):
    """Write a video where frame i shows picture frame_ids[i] (a flat grey level).

    Consecutive equal ids are exact repeats, like the duplicate frames a
    capture card records when the game misses a refresh. `overlay(i, frame)`
    can draw something extra on each frame.
    """
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
    for i, pid in enumerate(frame_ids):
        frame = np.full((size[1], size[0], 3), gray_level(pid), np.uint8)
        if overlay:
            overlay(i, frame)
        out.write(frame)
    out.release()


def expected_frametime_counts(frame_ids):
    """Frametimes (in video frames) the analyser should report for frame_ids."""
    counts, last_unique = [], 0
    for i in range(1, len(frame_ids)):
        if frame_ids[i] != frame_ids[i - 1]:
            counts.append(i - last_unique)
            last_unique = i
    return counts


def dropped_frame_ids(n=300):
    """Mostly smooth 60fps, with single, double and a long (6 frame) repeat."""
    ids, pid = [], 0
    for i in range(n):
        if not (i % 10 == 5 or i % 50 == 20 or 200 <= i < 206):
            pid += 1
        ids.append(pid)
    return ids


def block_pattern(pid, size, block=10):
    """A random blocky picture, different for each pid, with detail on every row."""
    rng = np.random.default_rng(pid + 1)
    small = rng.integers(0, 256, (size[1] // block, size[0] // block, 3), dtype=np.uint8)
    return cv2.resize(small, size, interpolation=cv2.INTER_NEAREST)


def write_torn_video(path, flip_times, frames, fps=60, size=(640, 360)):
    """A V-Sync-off capture: the game swaps to picture n at flip_times[n]
    (in refreshes). Each captured row shows the newest picture that had
    started when that row was scanned out."""
    pictures = {}
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
    rows = np.arange(size[1])
    times = np.asarray(flip_times)
    for k in range(frames):
        scan = k + rows / size[1]
        ids = np.searchsorted(times, scan, side="right") - 1  # -1 = before the first flip
        frame = np.empty((size[1], size[0], 3), np.uint8)
        for pid in np.unique(ids):
            if pid not in pictures:
                pictures[pid] = block_pattern(int(pid), size)
            sel = ids == pid
            frame[sel] = pictures[pid][sel]
        out.write(frame)
    out.release()


def tearing_schedule(game_fps=50, refresh=60, offset=0.25, frames=120):
    step = refresh / game_fps
    return [offset + n * step for n in range(math.ceil(frames / step) + 1)]


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.temp_dir, name)
