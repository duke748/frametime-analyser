"""Frame comparison: the low-resolution analysis image, the region mask and
tear detection."""
import math
from collections import namedtuple

import cv2
import numpy as np

from .video import CANVAS_SIZE, canvas_fit, fit_to_canvas

# Frames are compared at 1/10 of the canvas size: 128 x 72. Each analysis
# pixel covers a 10 x 10 block of the canvas.
ANALYSIS_SCALE = 10
ANALYSIS_SIZE = (CANVAS_SIZE[0] // ANALYSIS_SCALE, CANVAS_SIZE[1] // ANALYSIS_SCALE)

Rect = namedtuple("Rect", "x y w h")


def parse_rect(text):
    """Parse 'X,Y,W,H' (source video pixels)."""
    parts = text.replace(" ", "").split(",")
    if len(parts) != 4:
        raise ValueError("expected X,Y,W,H")
    x, y, w, h = (int(p) for p in parts)
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        raise ValueError("X and Y must be 0 or more, W and H more than 0")
    return Rect(x, y, w, h)


def analysis_image(frame):
    """Greyscale 128x72 version of a source frame, fitted the same way as the canvas.

    INTER_AREA averages each block, so compression noise mostly cancels out.
    """
    small = fit_to_canvas(frame, *ANALYSIS_SIZE, inter=cv2.INTER_AREA)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small


class AnalysisMask:
    """Which analysis pixels are compared: the picture area (not letterbox bars),
    limited to a region of interest and minus ignored regions if given."""

    def __init__(self, src_width, src_height, roi=None, ignore=()):
        w, h = ANALYSIS_SIZE
        fit = canvas_fit(src_width, src_height, w, h)
        self.picture = (fit.x, fit.y, fit.width, fit.height)
        self.picture_rows = (fit.y, fit.y + fit.height)
        self.mask = np.zeros((h, w), np.uint8)
        self.mask[fit.y:fit.y + fit.height, fit.x:fit.x + fit.width] = 255

        # Rects in analysis-grid coordinates, for drawing on the canvas too
        self.roi = self._to_grid(roi, fit) if roi else None
        self.ignore = [self._to_grid(r, fit) for r in ignore]
        if self.roi:
            keep = np.zeros_like(self.mask)
            x0, y0, x1, y1 = self.roi
            keep[y0:y1, x0:x1] = 255
            self.mask &= keep
        for x0, y0, x1, y1 in self.ignore:
            self.mask[y0:y1, x0:x1] = 0

        self.count = int(np.count_nonzero(self.mask))
        if self.count < 16:
            raise ValueError("the region left to analyse is too small; check --roi / --ignore")
        self.row_counts = np.count_nonzero(self.mask, axis=1)
        self.full = self.count == fit.width * fit.height  # Nothing excluded but letterbox bars
        self.full_frame = self.count == w * h               # Nothing excluded at all

    @staticmethod
    def _to_grid(rect, fit):
        """Source-pixel rect -> analysis-grid (x0, y0, x1, y1), rounded outwards."""
        x0 = fit.x + math.floor(rect.x * fit.scale)
        y0 = fit.y + math.floor(rect.y * fit.scale)
        x1 = fit.x + math.ceil((rect.x + rect.w) * fit.scale)
        y1 = fit.y + math.ceil((rect.y + rect.h) * fit.scale)
        x1 = min(max(x1, x0 + 1), fit.x + fit.width)
        y1 = min(max(y1, y0 + 1), fit.y + fit.height)
        return (x0, y0, x1, y1)

    def canvas_rects(self):
        """ROI and ignored regions in canvas pixels, for drawing outlines."""
        s = ANALYSIS_SCALE
        to_canvas = lambda r: tuple(v * s for v in r)  # noqa: E731
        return (to_canvas(self.roi) if self.roi else None), [to_canvas(r) for r in self.ignore]


def frame_difference(gray, prev_gray, mask=None, rows=False):
    """Mean absolute difference between two analysis images (0-255).

    With rows=True also returns the mean difference of each row, as a float
    array with NaN for rows the mask excludes entirely.
    """
    diff = cv2.absdiff(gray, prev_gray)
    if mask is None:
        mean = float(diff.mean())
        row_means = diff.mean(axis=1) if rows else None
    else:
        m = mask.mask
        mean = float(cv2.mean(diff, m)[0])
        if rows:
            sums = np.where(m > 0, diff, 0).sum(axis=1, dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                row_means = np.where(mask.row_counts > 0, sums / mask.row_counts, np.nan)
        else:
            row_means = None
    return (mean, row_means) if rows else mean


Change = namedtuple("Change", "mean peak rows")


def frame_change(gray, prev_gray, mask=None, rows=False):
    """How much changed between two analysis images.

    mean: average difference (0-255), used for tear detection.
    peak: 99th percentile of the lightly blurred difference. Compression noise
          is low everywhere, while even a small real change (an NPC shifting,
          a flickering light) makes a few strongly changed spots, so this tells
          new frames from repeats far better than the mean in still scenes.
    rows: per-row mean differences (NaN where masked out), if requested.
    """
    mean, row_means = frame_difference(gray, prev_gray, mask, rows=True) if rows else \
        (frame_difference(gray, prev_gray, mask), None)
    blurred = cv2.blur(cv2.absdiff(gray, prev_gray).astype(np.float32), (3, 3))
    values = blurred[mask.mask > 0] if mask is not None and not mask.full_frame else blurred
    peak = float(np.percentile(values, 99))
    return Change(mean, peak, row_means)


class TearDetector:
    """Find where new frames start within a captured frame (screen tearing).

    With V-Sync off, the game can swap to a new frame while the screen is
    being scanned out, so the top of a captured frame shows the old frame and
    the bottom shows the new one. Compared with the previous captured frame,
    rows above the tear line are unchanged and rows below it have changed.

    detect() returns the position (0 = top, 1 = bottom of the picture) of each
    new frame that starts in this captured frame. A change that starts at the
    top is a new frame at position 0, unless it just completes a tear from the
    previous captured frame (the rows above that tear catching up).

    Rows that almost never change (a static HUD, a black sky) would look like
    the top of a tear, so they are learned and ignored.

    Returns None when the changes don't have the shape tearing makes (for
    example scattered small changes in a still scene); the caller then decides
    with its whole-frame rule.
    """

    def __init__(self, picture_rows, min_band=2, tolerance=2, static_limit=0.9, max_gap=2, edge_margin=2):
        self.top, self.bottom = picture_rows
        self.height = self.bottom - self.top
        self.min_band = min_band
        self.tolerance = tolerance
        self.static_limit = static_limit
        self.max_gap = max_gap          # Unchanged runs this short inside a band are filled in
        self.edge_margin = edge_margin  # A band this close to the top/bottom touches it
        self.fragmented_frames = 0      # Frames that didn't look like tearing (diagnostic)
        self.catch_ups = 0              # Frames completing the previous frame's tear: proof of tearing
        self.static_score = np.zeros(ANALYSIS_SIZE[1])
        self.prev_tears = []
        self.last_tear_rows = []

    # A frame where at least this share of rows changed is a (near) full update
    FULL_UPDATE = 0.85

    def detect(self, rows, threshold):
        self.last_tear_rows = []
        cov_idx = np.flatnonzero(~np.isnan(rows))
        if len(cov_idx) < 4:
            self.prev_tears = []
            return None

        changed_all = rows[cov_idx] >= threshold
        if len(changed_all) >= 3:  # Median filter: drop single-row blips
            c = changed_all.astype(np.int8)
            mid = (c[:-2] + c[1:-1] + c[2:]) >= 2
            changed_all = np.concatenate(([changed_all[0]], mid, [changed_all[-1]]))
        self._learn_static_rows(cov_idx, changed_all)

        valid = self.static_score[cov_idx] < self.static_limit
        idx = cov_idx[valid]
        if len(idx) < 4:
            self.prev_tears = []
            return None
        changed = _fill_gaps(changed_all[valid], self.max_gap)
        n = len(changed)

        bands = [(s, e) for s, e in _runs(changed) if e - s >= self.min_band or (s == 0 and e == n)]
        catch_up, new_top, tear_bands = [], [], []
        for s, e in bands:
            touches_top = s <= self.edge_margin
            touches_bottom = e >= n - self.edge_margin
            end_row = idx[e - 1] + 1
            # Rows above last frame's tear catching up: they end at (or, where the
            # picture is flat near the tear, above) that tear line
            if any(end_row <= t + self.tolerance and (touches_top or abs(end_row - t) <= self.tolerance)
                   for t in self.prev_tears):
                catch_up.append((s, e))
                self.catch_ups += 1
            elif touches_top:
                new_top.append((s, e))
            elif touches_bottom:
                tear_bands.append((s, e))
            else:
                break  # A band touching neither edge: not the shape tearing makes
        else:
            if len(new_top) <= 1 and len(tear_bands) <= 1:
                positions = [0.0] * len(new_top)
                tears = [idx[s] for s, _e in tear_bands]
                positions += [(row - self.top) / self.height for row in tears]
                self.prev_tears = tears
                self.last_tear_rows = tears
                return positions

        # Scattered changes in a mostly still scene. Fall back to the
        # whole-frame rule (at most one new frame), and keep the last known
        # tear line in case the next frame completes it.
        self.fragmented_frames += 1
        return None

    def _learn_static_rows(self, cov_idx, changed):
        """Update which rows count as static content.

        A row is only evidence of static content when it stays the same while
        the picture around it changes: inside the span of changed rows, or in a
        near-full update. Rows above or below a tear are unchanged for a
        different reason and teach nothing. Every covered row keeps being
        updated, so a row wrongly marked static can recover.
        """
        hits = np.flatnonzero(changed)
        if len(hits) == 0:
            return
        if changed.mean() >= self.FULL_UPDATE:
            use = np.ones(len(changed), bool)
        else:
            use = np.zeros(len(changed), bool)
            use[hits[0]:hits[-1] + 1] = True
        rows = cov_idx[use]
        same = (~changed[use]).astype(float)
        self.static_score[rows] += 0.1 * (same - self.static_score[rows])


def _fill_gaps(flags, max_gap):
    """Set short runs of False that sit between True values to True."""
    flags = flags.copy()
    hits = np.flatnonzero(flags)
    for a, b in zip(hits, hits[1:]):
        if 1 < b - a <= max_gap + 1:
            flags[a + 1:b] = True
    return flags


def _runs(flags):
    """(start, end) of each run of True values."""
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs
