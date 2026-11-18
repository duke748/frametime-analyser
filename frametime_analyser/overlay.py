"""Drawing the graphs and statistics onto video frames.

Each frame is drawn in three layers, cheapest first:
- static: boxes, labels and reference lines, drawn once
- slow: FPS readout, CDF, histogram and statistics text, redrawn ~10x a second
- per-frame: the scrolling frametime, delta and FPS graphs
The cached layers are composited onto the frame with a mask.
"""
import cv2
import numpy as np

from .stats import (
    GREEN,
    ORANGE,
    RED,
    YELLOW,
    categorize_frametime,
    format_bucket_range,
    fps_color,
    get_histogram_buckets,
)
from .video import CANVAS_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
CYAN = (255, 200, 0)
FONT = cv2.FONT_HERSHEY_DUPLEX

# Layers are filled with this colour; pixels still this colour are transparent.
KEY = (1, 2, 3)

PT_WIDTH = 3    # Width of one sample on the frametime/delta graphs
PT_SPACING = 3  # Distance between the starts of two samples


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


def draw_text_right(img, text, right, y, scale, color=WHITE, thickness=1, border=1):
    (w, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    draw_text(img, text, (right - w, y), scale, color, thickness, border)


class Layer:
    """A cached drawing composited onto frames with a mask."""

    def __init__(self, size=CANVAS_SIZE):
        self.img = np.empty((size[1], size[0], 3), np.uint8)
        self.blank = np.empty_like(self.img)
        self.blank[:] = KEY
        self.mask = None
        self.key = None
        self.clear()

    def clear(self):
        np.copyto(self.img, self.blank)  # Much faster than filling with a tuple
        self.mask = None

    def finish(self):
        mask = cv2.inRange(self.img, KEY, KEY)
        self.mask = cv2.bitwise_not(mask)

    def apply(self, dst):
        if self.mask is not None:
            cv2.copyTo(self.img, self.mask, dst)


class GraphBox:
    def __init__(self, x, bottom, width, height):
        self.x, self.bottom, self.width, self.height = x, bottom, width, height
        self.top = bottom - height
        self.right = x + width


def scaled_y(values, box, max_value):
    ys = box.bottom - np.minimum(values, max_value) * (box.height / max_value)
    return np.maximum(ys, box.top).astype(np.int32)


def draw_step_graph(img, values, valid, box, max_value, color):
    """Step graph of the newest values with one polyline. `valid` is how many of
    the newest values are real data (the rest are padding). Returns samples shown."""
    n = box.width // PT_SPACING
    values = np.asarray(values, dtype=np.float64)[-n:]
    n = len(values)
    start = max(1, n - min(valid, n) + 1)
    if start < n:
        ys = scaled_y(values, box, max_value)
        keys = np.arange(start, n)
        xs = box.x + keys * PT_SPACING
        pts = np.empty((2 * len(keys) + 1, 2), np.int32)
        pts[0] = (xs[0], ys[start - 1])
        pts[1::2, 0], pts[1::2, 1] = xs, ys[keys]
        pts[2::2, 0], pts[2::2, 1] = xs + PT_WIDTH, ys[keys]
        cv2.polylines(img, [pts], False, color, 2)
    return n


def draw_colored_step_graph(img, values, valid, box, max_value, thresholds, colors):
    """Step graph where each sample's colour depends on its value:
    colors[i] for values below thresholds[i], the last colour above them all."""
    n = box.width // PT_SPACING
    values = np.asarray(values, dtype=np.float64)[-n:]
    n = len(values)
    start = max(1, n - min(valid, n) + 1)
    if start >= n:
        return
    ys = scaled_y(values, box, max_value)
    keys = np.arange(start, n)
    xs = box.x + keys * PT_SPACING
    segs = np.empty((len(keys), 3, 2), np.int32)
    segs[:, 0, 0], segs[:, 0, 1] = xs, ys[keys - 1]
    segs[:, 1, 0], segs[:, 1, 1] = xs, ys[keys]
    segs[:, 2, 0], segs[:, 2, 1] = xs + PT_WIDTH, ys[keys]
    band = np.searchsorted(np.asarray(thresholds), values[keys], side="right")
    for i, color in enumerate(colors):
        sel = segs[band == i]
        if len(sel):
            cv2.polylines(img, list(sel), False, color, 2)


def fps_graph_points(a, x0, y0, height, step, max_fps):
    """Polyline points for the FPS graph, or None if there is nothing to draw yet."""
    values = a.fps_graph_array()
    n = len(values)
    # Only draw samples recorded after the one-second FPS window filled
    first = max(1, n - a.frames_processed + a.target_fps)
    if first >= n:
        return None
    scale = (height - 1) / max_fps  # Leave 1px at the top so the line is visible at peak
    j = np.arange(first - 1, n)
    ys = (y0 + height - np.clip(values[j], 0, max_fps) * scale).astype(np.int32)
    return np.stack([x0 + (j + 1) * step, ys], axis=1).astype(np.int32)


def fps_reference_lines(img, x0, y0, width, height, max_fps, label_x):
    scale = (height - 1) / max_fps
    for ref in (30, 60, 120, 240):
        if ref < max_fps:
            y = int(y0 + height - ref * scale)
            cv2.line(img, (x0, y), (x0 + width, y), WHITE, 1)
            draw_text(img, str(ref), (label_x, y + 5), 0.4)
    draw_text(img, str(max_fps), (label_x, y0 + 5), 0.4)
    draw_text(img, "0", (label_x, y0 + height + 4), 0.4)


def draw_title_bar(img, title, scale=1.0):
    w = img.shape[1]
    (tw, th), _ = cv2.getTextSize(title, FONT, scale, 2)
    x, y, pad = (w - tw) // 2, 40, 10
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + pad), BLACK, -1)
    draw_text(img, title, (x, y), scale, thickness=2)


def blend_watermark(img, wm, x, y):
    wm_h, wm_w = wm.shape[:2]
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


class FpsReadout:
    """The large FPS number: hidden for the first second, then updated every half second."""

    def __init__(self, target_fps):
        self.target_fps = target_fps
        self.text = ""
        self.timer = 0
        self.interval = max(1, target_fps // 2)

    def update(self, a):
        if a.frames_processed < self.target_fps:
            self.text = ""
        else:
            self.timer += 1
            if not self.text or self.timer >= self.interval:
                self.text = str(a.fps)
                self.timer = 0
        return self.text


class Overlay:
    """Graphs and statistics for one video, on a CANVAS_SIZE frame."""

    GRAPH_BOTTOM = 420
    GRAPH_HEIGHT = 100

    def __init__(self, target_fps, show_delta_plot=False, detect_stutters=False,
                 detect_tears=False, title=None, watermark=None, mask=None):
        self.target_fps = target_fps
        self.target_frametime = 1000.0 / target_fps
        self.show_delta_plot = show_delta_plot
        self.detect_stutters = detect_stutters
        self.detect_tears = detect_tears
        self.title = title
        self.watermark = watermark
        self.mask = mask
        self.buckets = get_histogram_buckets(target_fps)
        self.readout = FpsReadout(target_fps)
        self.slow_interval = max(1, target_fps // 10)

        # Graph columns. With the delta plot on, three graphs share the space
        # left of the histogram, so they are narrower.
        width = 240 if show_delta_plot else 270
        b, h = self.GRAPH_BOTTOM, self.GRAPH_HEIGHT
        if show_delta_plot:
            self.ft_box, self.delta_box, self.cdf_box = (
                GraphBox(40, b, width, h), GraphBox(345, b, width, h), GraphBox(650, b, width, h))
        else:
            self.ft_box, self.delta_box, self.cdf_box = (
                GraphBox(60, b, width, h), None, GraphBox(410, b, width, h))
        self.hist_box = GraphBox(1000, b, 270, h)
        self.fps_x, self.fps_y, self.fps_h, self.fps_step = 60, 500, 160, 2
        self.fps_w = 580 * self.fps_step
        self.max_ft = self.target_frametime * 2.5
        self.max_delta = self.target_frametime * 1.5
        self.max_cdf = self.target_frametime * 3

        self.static = Layer()
        self._draw_static(self.static.img)
        self.static.finish()
        self.slow = Layer()

    def draw(self, img, a):
        """Draw everything for analyser `a` onto img (a CANVAS_SIZE frame)."""
        fps_text = self.readout.update(a)
        key = (a.frames_processed // self.slow_interval, fps_text, a.in_static)
        if key != self.slow.key:
            self.slow.clear()
            self._draw_slow(self.slow.img, a, fps_text)
            self.slow.finish()
            self.slow.key = key

        self._draw_dynamic(img, a)
        self.slow.apply(img)
        self.static.apply(img)
        if self.watermark is not None:
            blend_watermark(img, self.watermark, 10, img.shape[0] - self.watermark.shape[0] - 10)

    # Static -----------------------------------------------------------------

    def _draw_static(self, img):
        t = self.target_frametime
        box = self.ft_box
        cv2.rectangle(img, (box.x, box.top), (box.right, box.bottom), WHITE, 2)
        target_y = int(box.bottom - t * box.height / self.max_ft)
        cv2.line(img, (box.x, target_y), (box.right, target_y), GREY, 1)
        label_x = box.right + 8
        draw_text(img, "0", (label_x, box.bottom + 4), 0.4)
        draw_text(img, f"{t:.1f}", (label_x, target_y + 4), 0.4)
        draw_text(img, f"{self.max_ft:.1f}", (label_x, box.top + 8), 0.4)
        draw_text(img, "FRAMETIME (MS)", (box.x, box.top - 15), 0.45)

        if self.delta_box:
            box = self.delta_box
            cv2.rectangle(img, (box.x, box.top), (box.right, box.bottom), WHITE, 2)
            draw_text(img, "0", (box.right + 8, box.bottom + 4), 0.4)
            draw_text(img, f"{self.max_delta:.1f}", (box.right + 8, box.top + 8), 0.4)
            draw_text(img, "FRAME DELTA (MS)", (box.x, box.top - 15), 0.45)

        box = self.cdf_box
        median_y = box.bottom - int(0.5 * box.height)
        p95_y = box.bottom - int(0.95 * box.height)
        cv2.line(img, (box.x, median_y), (box.right, median_y), GREY, 1)
        cv2.line(img, (box.x, p95_y), (box.right, p95_y), GREY, 1)
        cv2.rectangle(img, (box.x, box.top), (box.right, box.bottom), WHITE, 2)
        target_x = box.x + int(t / self.max_cdf * box.width)
        cv2.line(img, (target_x, box.top), (target_x, box.bottom), GREEN, 1)
        label_x = box.right + 8
        draw_text(img, "0%", (label_x, box.bottom - 5), 0.3)
        draw_text(img, "50%", (label_x, median_y + 5), 0.3)
        draw_text(img, "100%", (label_x, box.top + 5), 0.3)
        draw_text(img, "0", (box.x - 5, box.bottom + 15), 0.3)
        draw_text(img, f"{t:.0f}", (target_x - 8, box.bottom + 15), 0.3, GREEN)
        draw_text(img, f"{self.max_cdf:.0f}", (box.right - 15, box.bottom + 15), 0.3)
        draw_text(img, "SMOOTHNESS (CDF)", (box.x, box.top - 15), 0.45)

        box = self.hist_box
        cv2.rectangle(img, (box.x, box.top), (box.right, box.bottom), WHITE, 2)
        draw_text(img, "FRAMETIME DISTRIBUTION", (box.x, box.top - 55), 0.45)
        bar_width = box.width // len(self.buckets)
        for i, (lo, hi, label, color) in enumerate(self.buckets):
            label_x = box.x + i * bar_width + 3
            cv2.rectangle(img, (label_x, box.bottom + 15), (label_x + 6, box.bottom + 21), color, -1)
            draw_text(img, label, (label_x + 10, box.bottom + 21), 0.28)
            draw_text(img, format_bucket_range(lo, hi), (label_x, box.bottom + 30), 0.25)

        x0, y0, w, h = self.fps_x, self.fps_y, self.fps_w, self.fps_h
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), WHITE, 2)
        fps_reference_lines(img, x0, y0, w, h, self.target_fps, x0 + w + 10)
        draw_text(img, "FRAME-RATE (FPS)", (x0 + w - 150, y0 - 15), 0.5)

        if self.title:
            draw_title_bar(img, self.title)

        if self.mask is not None:
            roi, ignore = self.mask.canvas_rects()
            if roi:
                cv2.rectangle(img, roi[:2], (roi[2] - 1, roi[3] - 1), CYAN, 1)
            for r in ignore:
                cv2.rectangle(img, r[:2], (r[2] - 1, r[3] - 1), GREY, 1)

    # Slow -------------------------------------------------------------------

    def _draw_slow(self, img, a, fps_text):
        if fps_text:
            draw_text_right(img, fps_text, 1240, 150, 2, thickness=5, border=2)
        y = 180
        if self.detect_tears:
            s = a.summary()
            draw_text_right(img, f"TORN {s['torn_pct']:.1f}%", 1240, y, 0.55,
                            RED if s["torn_pct"] >= 5 else WHITE)
            y += 25
        if a.in_static:
            draw_text_right(img, "STATIC SCENE (NOT COUNTED)", 1240, y, 0.5, YELLOW)

        if len(a.window) > 10:
            self._draw_cdf_curve(img, a)
            self._draw_histogram(img, a)

    def _draw_cdf_curve(self, img, a):
        box = self.cdf_box
        samples = 100
        thresholds = [self.max_cdf * i / samples for i in range(samples + 1)]
        percentages = a.window.cdf(thresholds)
        pts = [(box.x + int(ft / self.max_cdf * box.width), box.bottom - int(p / 100 * box.height))
               for ft, p in zip(thresholds, percentages)]
        for i in range(1, len(pts)):
            p = percentages[i]
            color = GREEN if p >= 90 else YELLOW if p >= 70 else ORANGE
            cv2.line(img, pts[i - 1], pts[i], color, 2)

    def _draw_histogram(self, img, a):
        box = self.hist_box
        buckets = self.buckets
        counts = a.window.bucket_counts(buckets)
        total = sum(counts)
        percentages = [(c / total * 100) if total else 0 for c in counts]

        avg_ft = a.window.mean()
        avg_bucket = categorize_frametime(avg_ft, buckets)
        avg_ft_color = buckets[avg_bucket][3] if avg_bucket is not None else WHITE
        avg_fps = a.window.average_fps()
        draw_text(img, f"Frametime Avg: {avg_ft:.1f}ms", (box.x, box.top - 40), 0.35, avg_ft_color)
        draw_text(img, f"FPS Avg: {avg_fps:.1f}  1% Low: {a.window.low_fps(1):.1f}",
                  (box.x, box.top - 25), 0.35, fps_color(avg_fps, self.target_fps))
        if self.detect_stutters and a.stutters:
            color = RED if a.severe_stutters else ORANGE
            text = (f"Stutters: {len(a.stutters)} ({a.severe_stutters} severe, "
                    f"{a.moderate_stutters} moderate)")
            draw_text(img, text, (box.x, box.top - 10), 0.35, color)

        bar_width = box.width // len(buckets)
        for i, (pct, (_lo, _hi, _label, color)) in enumerate(zip(percentages, buckets)):
            bar_x = box.x + i * bar_width
            if pct > 0:
                bar_y = box.bottom - int(pct / 100 * box.height)
                cv2.rectangle(img, (bar_x + 2, bar_y), (bar_x + bar_width - 2, box.bottom - 2), color, -1)
                if pct > 5:
                    text = f"{pct:.0f}%"
                    (tw, _), _ = cv2.getTextSize(text, FONT, 0.3, 1)
                    draw_text(img, text, (bar_x + (bar_width - tw) // 2, bar_y + 15), 0.3)
            draw_text(img, f"{pct:.1f}%", (bar_x + 3, box.bottom + 45), 0.28)

    # Per frame --------------------------------------------------------------

    def _draw_dynamic(self, img, a):
        box = self.ft_box
        valid = min(a.graph_samples, len(a.frametime_graph))
        n = draw_step_graph(img, a.frametime_graph, valid, box, self.max_ft, GREEN)

        if self.detect_stutters:
            newest = a.graph_samples - 1
            for s in reversed(a.stutters):
                pos = n - 1 - (newest - s.graph_index)
                if pos < 1:
                    break  # This and all older stutters have scrolled off the graph
                x = box.x + pos * PT_SPACING + 1
                color, thickness = (RED, 3) if s.severity == "severe" else (ORANGE, 2)
                cv2.line(img, (x, box.top), (x, box.bottom), color, thickness)

        if self.delta_box:
            t = self.target_frametime
            draw_colored_step_graph(img, a.delta_graph, valid, self.delta_box, self.max_delta,
                                    [t * 0.5, t], [GREEN, YELLOW, RED])

        pts = fps_graph_points(a, self.fps_x, self.fps_y, self.fps_h, self.fps_step, self.target_fps)
        if pts is not None:
            cv2.polylines(img, [pts], False, GREEN, 2)

        if self.detect_tears and a.tear_detector is not None:
            from .detection import ANALYSIS_SCALE
            w = img.shape[1]
            for row in a.tear_detector.last_tear_rows:
                y = row * ANALYSIS_SCALE
                cv2.line(img, (w - 24, y), (w, y), RED, 3)


class CompareOverlay:
    """Two videos side by side with their graphs, on a CANVAS_SIZE frame."""

    VIDEO_SIZE = (640, 360)
    VIDEO_Y = 55
    COLORS = (GREEN, CYAN)
    FRAMETIME_SAMPLES = 180

    def __init__(self, target_fps, labels, detect_stutters=False, detect_tears=False,
                 title=None, watermark=None):
        self.target_fps = target_fps
        self.target_frametime = 1000.0 / target_fps
        self.labels = labels
        self.detect_stutters = detect_stutters
        self.detect_tears = detect_tears
        self.title = title
        self.watermark = watermark
        self.readouts = (FpsReadout(target_fps), FpsReadout(target_fps))
        self.slow_interval = max(1, target_fps // 10)
        self.max_ft = self.target_frametime * 2.5
        self.ft_boxes = (GraphBox(40, 565, 540, 90), GraphBox(680, 565, 540, 90))
        self.fps_x, self.fps_y, self.fps_h, self.fps_step = 60, 600, 105, 2
        self.fps_w = 580 * self.fps_step

        self.static = Layer()
        self._draw_static(self.static.img)
        self.static.finish()
        self.slow = Layer()

    def compose(self, frame_a, frame_b):
        """New canvas with both (VIDEO_SIZE) frames side by side."""
        canvas = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), np.uint8)
        w, h = self.VIDEO_SIZE
        y = self.VIDEO_Y
        canvas[y:y + h, 0:w] = frame_a
        canvas[y:y + h, w:2 * w] = frame_b
        return canvas

    def draw(self, img, a, b):
        texts = (self.readouts[0].update(a), self.readouts[1].update(b))
        key = (a.frames_processed // self.slow_interval, texts)
        if key != self.slow.key:
            self.slow.clear()
            self._draw_slow(self.slow.img, (a, b), texts)
            self.slow.finish()
            self.slow.key = key

        for analyser, box, color in zip((a, b), self.ft_boxes, self.COLORS):
            valid = min(analyser.graph_samples, len(analyser.frametime_graph))
            n = draw_step_graph(img, analyser.frametime_graph, valid, box, self.max_ft, color)
            if self.detect_stutters:
                newest = analyser.graph_samples - 1
                for s in reversed(analyser.stutters):
                    pos = n - 1 - (newest - s.graph_index)
                    if pos < 1:
                        break
                    x = box.x + pos * PT_SPACING + 1
                    cv2.line(img, (x, box.top), (x, box.bottom),
                             RED if s.severity == "severe" else ORANGE, 2)
        for analyser, color in zip((a, b), self.COLORS):
            pts = fps_graph_points(analyser, self.fps_x, self.fps_y, self.fps_h, self.fps_step,
                                   self.target_fps)
            if pts is not None:
                cv2.polylines(img, [pts], False, color, 2)

        self.slow.apply(img)
        self.static.apply(img)
        if self.watermark is not None:
            blend_watermark(img, self.watermark, 10, 10)

    def _draw_static(self, img):
        w, h = self.VIDEO_SIZE
        for i, (label, color) in enumerate(zip(self.labels, self.COLORS)):
            draw_text(img, label, (i * w + 12, self.VIDEO_Y + 26), 0.7, color)
        cv2.line(img, (w, self.VIDEO_Y), (w, self.VIDEO_Y + h), BLACK, 2)

        t = self.target_frametime
        for box in self.ft_boxes:
            cv2.rectangle(img, (box.x, box.top), (box.right, box.bottom), WHITE, 2)
            target_y = int(box.bottom - t * box.height / self.max_ft)
            cv2.line(img, (box.x, target_y), (box.right, target_y), GREY, 1)
            draw_text(img, "0", (box.right + 8, box.bottom + 4), 0.4)
            draw_text(img, f"{t:.1f}", (box.right + 8, target_y + 4), 0.4)
            draw_text(img, f"{self.max_ft:.1f}", (box.right + 8, box.top + 8), 0.4)
            draw_text(img, "FRAMETIME (MS)", (box.x, box.top - 8), 0.4)

        x0, y0, fw, fh = self.fps_x, self.fps_y, self.fps_w, self.fps_h
        cv2.rectangle(img, (x0, y0), (x0 + fw, y0 + fh), WHITE, 2)
        fps_reference_lines(img, x0, y0, fw, fh, self.target_fps, x0 + fw + 10)

        if self.title:
            draw_title_bar(img, self.title, 0.9)

    def _draw_slow(self, img, analysers, texts):
        w, _ = self.VIDEO_SIZE
        for i, (a, text, color) in enumerate(zip(analysers, texts, self.COLORS)):
            if text:
                draw_text_right(img, text, i * w + w - 15, self.VIDEO_Y + 60, 1.5, color, 4, 2)
            if len(a.window) == 0:
                continue
            avg_fps = a.window.average_fps()
            parts = [f"Avg {avg_fps:.1f} fps", f"1% low {a.window.low_fps(1):.1f}",
                     f"{a.window.mean():.1f} ms"]
            if self.detect_stutters:
                parts.append(f"Stutters {len(a.stutters)}")
            if self.detect_tears:
                parts.append(f"Torn {a.summary()['torn_pct']:.0f}%")
            draw_text(img, "   ".join(parts), (i * w + 40, 445), 0.45, fps_color(avg_fps, self.target_fps))
            if a.in_static:
                draw_text(img, "STATIC SCENE", (i * w + 40, 425), 0.45, YELLOW)
