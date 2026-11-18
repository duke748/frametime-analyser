"""Results: the per-frame CSV, the summary CSV and the HTML report."""
import csv
import datetime
import html
import math
import os
from array import array

from . import __version__
from .analyser import SEVERE_STUTTER_MULTIPLIER, STUTTER_MULTIPLIER, WINDOW_MINUTES
from .stats import format_bucket_range, get_histogram_buckets


class Recorder:
    """Writes every new frame to the per-frame CSV and keeps the whole run in
    memory (a few MB per hour of video) for the HTML report."""

    HEADER = ["unique_frame", "video_frame", "time_s", "frametime_ms", "tear_position", "after_static"]

    def __init__(self, path):
        self.path = path
        self.file = open(path, "w", newline="")
        self.csv = csv.writer(self.file)
        self.csv.writerow(self.HEADER)
        self.times = array("d")       # Seconds, counted frames only
        self.frametimes = array("d")  # Milliseconds, counted frames only
        self.all_times = array("d")   # Seconds, every new frame (for FPS over time)

    def add(self, f):
        t = f.time_ms / 1000.0
        self.csv.writerow([f.index, f.video_frame, f"{t:.4f}", f"{f.frametime_ms:.3f}",
                           f"{f.tear:.3f}" if f.tear is not None else "", int(f.after_static)])
        self.all_times.append(t)
        if not f.after_static:
            self.times.append(t)
            self.frametimes.append(f.frametime_ms)

    def close(self):
        self.file.close()


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
        writer.writerow(["Static Scenes Excluded", s["static_scenes"]])
        writer.writerow(["Static Time Excluded (s)", f"{s['static_s']:.1f}"])
        writer.writerow([])

        if a.tear_detector is not None:
            writer.writerow(["Tear Detection Results"])
            writer.writerow(["Torn Frames", s["torn_frames"]])
            writer.writerow(["Torn Frame Percentage", f"{s['torn_pct']:.2f}%"])
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


# --- HTML report ---------------------------------------------------------------

CHART_W, CHART_H = 1000, 220
PAD_L, PAD_B, PAD_T = 44, 24, 8


def _max_per_column(xs, ys, x_max, columns):
    """Downsample to one point per pixel column, keeping the highest value so spikes survive."""
    best = [None] * columns
    for x, y in zip(xs, ys):
        c = min(columns - 1, int(x / x_max * columns)) if x_max > 0 else 0
        if best[c] is None or y > best[c][1]:
            best[c] = (x, y)
    return [p for p in best if p is not None]


def _svg_chart(points, x_max, y_max, refs, css_class, unit, step=False):
    """Inline SVG line chart. points are (x seconds, y value); refs are (y, label)."""
    plot_w, plot_h = CHART_W - PAD_L - 8, CHART_H - PAD_B - PAD_T
    sx = lambda x: PAD_L + x / x_max * plot_w if x_max else PAD_L  # noqa: E731
    sy = lambda y: PAD_T + plot_h - min(y, y_max) / y_max * plot_h  # noqa: E731
    parts = [f'<svg viewBox="0 0 {CHART_W} {CHART_H}" class="chart" role="img">']
    for y, label in refs:
        parts.append(f'<line x1="{PAD_L}" x2="{CHART_W - 8}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" '
                     f'class="ref"/><text x="{PAD_L - 6}" y="{sy(y) + 4:.1f}" class="axis" '
                     f'text-anchor="end">{html.escape(label)}</text>')
    parts.append(f'<line x1="{PAD_L}" x2="{CHART_W - 8}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" class="base"/>')
    parts.append(f'<text x="{PAD_L - 6}" y="{sy(0) + 4:.1f}" class="axis" text-anchor="end">0</text>')
    ticks = _time_ticks(x_max)
    for t in ticks:
        anchor = "end" if sx(t) > CHART_W - 30 else "middle"
        parts.append(f'<text x="{sx(t):.1f}" y="{CHART_H - 6}" class="axis" '
                     f'text-anchor="{anchor}">{_fmt_time(t)}</text>')
    if points:
        coords = []
        for i, (x, y) in enumerate(points):
            if step and i:
                coords.append(f"{sx(x):.1f},{sy(points[i - 1][1]):.1f}")
            coords.append(f"{sx(x):.1f},{sy(y):.1f}")
        parts.append(f'<polyline points="{" ".join(coords)}" class="{css_class}"/>')
    parts.append(f'<text x="{CHART_W - 8}" y="{PAD_T + 12}" class="axis" text-anchor="end">{unit}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _time_ticks(x_max):
    if x_max <= 0:
        return []
    for step in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600):
        if x_max / step <= 10:
            break
    return [i * step for i in range(int(x_max // step) + 1)]


def _fmt_time(seconds):
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def build_report_html(video_path, a, recorder, generated=None):
    s = a.summary()
    t = a.target_frametime
    fps = a.target_fps
    duration = s["duration_s"]
    title = os.path.basename(video_path) if isinstance(video_path, str) else str(video_path)

    # Frametime over time
    ft_points = _max_per_column(recorder.times, recorder.frametimes, duration or 1, CHART_W - PAD_L)
    ft_max = round(max(t * 5, min(max(recorder.frametimes, default=t), t * 8)))
    ft_refs = [(t, f"{t:.1f}")]
    if a.detect_stutters:
        ft_refs += [(a.moderate_threshold, f"{a.moderate_threshold:.1f}"),
                    (a.severe_threshold, f"{a.severe_threshold:.1f}")]
    else:
        ft_refs.append((t * 2, f"{t * 2:.1f}"))
    ft_refs.append((ft_max, f"{ft_max:.0f}"))
    ft_chart = _svg_chart(ft_points, duration or 1, ft_max, ft_refs, "line ft", "ms")

    # FPS per second of video
    seconds = max(1, math.ceil(duration))
    per_second = [0] * seconds
    for ts in recorder.all_times:
        per_second[min(seconds - 1, int(ts))] += 1
    fps_points = [(i + 1, v) for i, v in enumerate(per_second)]
    if fps_points:
        fps_points.insert(0, (0, fps_points[0][1]))
    fps_refs = [(r, str(r)) for r in (30, 60, 120, 240) if r <= fps]
    fps_chart = _svg_chart(fps_points, seconds, fps * 1.05, fps_refs, "line fps", "fps", step=True)

    buckets = get_histogram_buckets(fps)
    counts = a.window.bucket_counts(buckets)
    total = sum(counts) or 1
    bucket_rows = "".join(
        f'<tr><td><span class="swatch" style="background:rgb({c[2]},{c[1]},{c[0]})"></span>'
        f'{html.escape(label)}</td><td>{format_bucket_range(lo, hi)} ms</td><td class="num">{n}</td>'
        f'<td class="num">{n / total * 100:.1f}%</td><td class="barcell"><div class="bar" '
        f'style="width:{n / total * 100:.1f}%;background:rgb({c[2]},{c[1]},{c[0]})"></div></td></tr>'
        for (lo, hi, label, c), n in zip(buckets, counts))

    tiles = [("Average FPS", f"{s['avg_fps']:.1f}"), ("1% low", f"{s['low_1_fps']:.1f}"),
             ("0.1% low", f"{s['low_01_fps']:.1f}"), ("Avg frametime", f"{s['avg_frametime']:.1f} ms"),
             ("99th pct frametime", f"{s['p99_frametime']:.1f} ms"),
             ("Unique frames", f"{s['unique_frames']:,}")]
    if a.detect_stutters:
        tiles.append(("Stutters", f"{len(a.stutters)} ({a.severe_stutters} severe)"))
    if a.tear_detector is not None:
        tiles.append(("Torn frames", f"{s['torn_pct']:.1f}%"))
    if s["static_scenes"]:
        tiles.append(("Static time excluded", f"{s['static_s']:.1f} s"))
    tiles_html = "".join(f'<div class="tile"><div class="label">{k}</div><div class="value">{v}</div></div>'
                         for k, v in tiles)

    stutter_html = ""
    if a.detect_stutters and a.stutters:
        worst = sorted(a.stutters, key=lambda st: -st.frametime_ms)[:25]
        rows = "".join(f'<tr><td>{_fmt_time(st.video_frame / fps)}</td><td class="num">{st.video_frame}</td>'
                       f'<td class="num">{st.frametime_ms:.1f} ms</td><td>{st.severity}</td></tr>'
                       for st in sorted(worst, key=lambda st: st.video_frame))
        stutter_html = (f'<h2>Worst stutters</h2><div class="scroll"><table><thead><tr><th>Time</th>'
                        f'<th class="num">Video frame</th><th class="num">Frametime</th><th>Severity</th>'
                        f'</tr></thead><tbody>{rows}</tbody></table></div>')

    generated = generated or datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return REPORT_TEMPLATE.format(
        title=html.escape(title), video=html.escape(str(video_path)), fps=fps,
        duration=_fmt_time(duration), generated=generated, version=__version__,
        tiles=tiles_html, ft_chart=ft_chart, fps_chart=fps_chart, bucket_rows=bucket_rows,
        stutters=stutter_html, window=WINDOW_MINUTES)


def write_report(path, video_path, a, recorder):
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_report_html(video_path, a, recorder))


REPORT_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} - Frametime Report</title>
<style>
:root {{ --bg:#f7f7f5; --panel:#fff; --text:#1d1d1b; --muted:#6b6b66; --line:#e2e2dc;
        --ft:#1f8a4c; --fps:#2563eb; --ref:#c9c9c2; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#141413; --panel:#1e1e1c; --text:#ececea; --muted:#9b9b95; --line:#33332f;
          --ft:#5de882; --fps:#6ea8ff; --ref:#44443f; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text);
       font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:22px; margin:0 0 4px; overflow-wrap:anywhere; }}
h2 {{ font-size:16px; margin:28px 0 10px; }}
.meta {{ color:var(--muted); font-size:13px; overflow-wrap:anywhere; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px; margin-top:20px; }}
.tile {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
.tile .label {{ color:var(--muted); font-size:12px; }}
.tile .value {{ font-size:20px; font-weight:600; font-variant-numeric:tabular-nums; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }}
.chart {{ width:100%; height:auto; display:block; }}
.chart .line {{ fill:none; stroke-width:1.5; vector-effect:non-scaling-stroke; }}
.chart .ft {{ stroke:var(--ft); }} .chart .fps {{ stroke:var(--fps); }}
.chart .ref {{ stroke:var(--ref); stroke-dasharray:4 4; }} .chart .base {{ stroke:var(--ref); }}
.chart .axis {{ fill:var(--muted); font-size:11px; }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line);
        border-radius:8px; overflow:hidden; font-size:14px; }}
th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }}
th {{ color:var(--muted); font-weight:500; font-size:12px; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.swatch {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:8px; }}
.barcell {{ width:35%; }} .bar {{ height:10px; border-radius:2px; }}
.note {{ color:var(--muted); font-size:12px; margin-top:6px; }}
.scroll {{ overflow-x:auto; }}
</style>
</head>
<body>
<main>
<h1>{title}</h1>
<div class="meta">{video} &middot; {fps} fps target &middot; {duration} analysed &middot;
generated {generated} by Frametime Analyser {version}</div>
<div class="tiles">{tiles}</div>
<h2>Frametime</h2>
<div class="panel">{ft_chart}</div>
<div class="note">Highest frametime per pixel column, so short spikes stay visible. Static scenes are left out.</div>
<h2>Frame rate</h2>
<div class="panel">{fps_chart}</div>
<div class="note">New frames in each second of video.</div>
<h2>Frametime distribution</h2>
<div class="scroll"><table><thead><tr><th>Bucket</th><th>Range</th><th class="num">Frames</th>
<th class="num">Share</th><th></th></tr></thead><tbody>{bucket_rows}</tbody></table></div>
<div class="note">Averages, lows and the distribution cover the last {window} minutes of video.</div>
{stutters}
</main>
</body>
</html>
"""
