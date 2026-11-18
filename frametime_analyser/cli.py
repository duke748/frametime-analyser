"""Command line interface and the main analysis loops."""
import argparse
import datetime
import os
import sys
import time
from collections import namedtuple

import cv2

from . import __version__
from .analyser import DEFAULT_MIN_DIFF, DEFAULT_STATIC_AFTER, FrametimeAnalyser
from .detection import ANALYSIS_SIZE, AnalysisMask, analysis_image, frame_change, parse_rect
from .overlay import CompareOverlay, Overlay, load_watermark
from .report import Recorder, export_summary_csv, write_report
from .video import (
    CANVAS_SIZE,
    ENCODER_CHOICES,
    FfmpegVideoStream,
    FileVideoStream,
    find_ffmpeg,
    fit_to_canvas,
    open_writer,
)

MIN_FPS, MAX_FPS = 1, 1000
DEFAULT_FPS = 60
WINDOW_NAME = "Frametime Analyser"

OutputPaths = namedtuple("OutputPaths", "video summary_csv frametimes_csv report")
Prepared = namedtuple("Prepared", "canvas gray src_size")


# --- Arguments -----------------------------------------------------------------

def target_fps_type(text):
    try:
        value = int(text)
    except ValueError:
        value = None
    if value is None or not MIN_FPS <= value <= MAX_FPS:
        raise argparse.ArgumentTypeError(
            f"FPS must be a whole number between {MIN_FPS} and {MAX_FPS}, got '{text}'")
    return value


def rect_type(text):
    try:
        return parse_rect(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid region '{text}': {e}") from None


def size_type(text):
    try:
        w, h = (int(v) for v in text.lower().split("x"))
        if w > 0 and h > 0:
            return (w, h)
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"invalid size '{text}', expected WIDTHxHEIGHT")


def non_negative_float(text):
    try:
        value = float(text)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a number of 0 or more, got '{text}'")
    return value


def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="frametime-analyser",
        description="Detect repeated frames in game footage and measure frame rate, "
                    "frametimes, stutter and tearing.",
        epilog="Single-dash spellings (-delta-plot, -detect-stutters, -watermark, -title) "
               "are also accepted. Regions are X,Y,W,H in source video pixels.",
    )
    p.add_argument("video", nargs="?", help="video file to analyse")
    p.add_argument("fps", nargs="?", type=target_fps_type,
                   help="frame rate the video was recorded at (default: read from the video, else 60)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    g = p.add_argument_group("analysis")
    g.add_argument("--fps", dest="fps_option", type=target_fps_type, metavar="FPS",
                   help="same as the fps argument (useful with --capture)")
    g.add_argument("--detect-stutters", "-detect-stutters", action="store_true",
                   help="detect and mark frametime spikes (shader/traversal stutter)")
    g.add_argument("--detect-tears", action="store_true",
                   help="detect screen tearing (V-Sync off footage); experimental")
    g.add_argument("--roi", type=rect_type, metavar="X,Y,W,H",
                   help="only analyse this region of the video")
    g.add_argument("--ignore", type=rect_type, action="append", default=[], metavar="X,Y,W,H",
                   help="ignore this region, e.g. a webcam or chat overlay (repeatable)")
    g.add_argument("--min-diff", type=non_negative_float, default=DEFAULT_MIN_DIFF, metavar="N",
                   help=f"smallest change (0-255, peak of the difference) that counts as a new frame; raise it if "
                        f"compression noise in still scenes counts as new frames (default {DEFAULT_MIN_DIFF})")
    g.add_argument("--static-after", type=non_negative_float, default=DEFAULT_STATIC_AFTER,
                   metavar="SECONDS",
                   help=f"repeats longer than this are a static scene and left out of the statistics "
                        f"(default {DEFAULT_STATIC_AFTER:g}; 0 to always count them)")

    g = p.add_argument_group("overlay")
    g.add_argument("--delta-plot", "-delta-plot", action="store_true",
                   help="show the frame-to-frame delta plot (micro-stutter)")
    g.add_argument("--watermark", "-watermark", metavar="IMAGE",
                   help="image to show at 32x32, 50%% opacity")
    g.add_argument("--title", "-title", metavar="TEXT", help="title shown at the top of the video")

    g = p.add_argument_group("output")
    g.add_argument("--no-display", action="store_true",
                   help="don't open a preview window or play in real time; analyse as fast as possible")
    g.add_argument("--no-video", action="store_true",
                   help="don't write the annotated video (much faster); CSV files and the report are still written")
    g.add_argument("--no-report", action="store_true", help="don't write the HTML report")
    g.add_argument("--encoder", choices=ENCODER_CHOICES, default="auto",
                   help="video encoder: auto (FFmpeg libx264 if available, else opencv), opencv (mp4v), "
                        "or an FFmpeg encoder such as h264_nvenc (NVIDIA), h264_qsv (Intel), h264_amf (AMD)")
    g.add_argument("--output-dir", metavar="DIR", help="folder for output files (default: next to the video)")

    g = p.add_argument_group("input")
    g.add_argument("--decoder", choices=["auto", "ffmpeg", "opencv"], default="auto",
                   help="video decoder: auto (FFmpeg if available, much faster), ffmpeg or opencv")
    g.add_argument("--hw-decode", action="store_true",
                   help="decode on the GPU if possible (often slower than a fast CPU; can help with 4K HEVC)")
    g.add_argument("--compare", metavar="VIDEO_B", help="compare with a second video, side by side")
    g.add_argument("--labels", nargs=2, metavar=("A", "B"), help="names for the two videos in --compare")
    g.add_argument("--capture", type=int, metavar="DEVICE",
                   help="analyse live from a capture device (0 = first device) instead of a file")
    g.add_argument("--capture-size", type=size_type, metavar="WxH",
                   help="resolution to request from the capture device, e.g. 1920x1080")
    return p


def parse_args(argv=None):
    parser = build_arg_parser()
    # Intermixed parsing lets the fps value come after options, e.g. "video -delta-plot 120"
    args = parser.parse_intermixed_args(argv)

    if args.capture is not None:
        if args.video is not None:
            if args.video.isdigit() and args.fps is None:
                args.fps, args.video = target_fps_type(args.video), None  # "--capture 0 120"
            else:
                parser.error("use either a video file or --capture, not both")
        if args.compare:
            parser.error("--compare can't be used with --capture")
        if args.hw_decode:
            parser.error("--hw-decode only applies to video files")
    elif args.video is None:
        parser.error("a video file is required (or use --capture)")
    elif not os.path.isfile(args.video):
        parser.error(f"video file '{args.video}' not found")

    if args.fps is not None and args.fps_option is not None and args.fps != args.fps_option:
        parser.error("the fps argument and --fps disagree")
    args.fps = args.fps if args.fps is not None else args.fps_option

    if args.compare and not os.path.isfile(args.compare):
        parser.error(f"video file '{args.compare}' not found")
    if args.labels and not args.compare:
        parser.error("--labels needs --compare")
    if args.watermark and not os.path.isfile(args.watermark):
        parser.error(f"watermark file '{args.watermark}' not found")
    if args.decoder == "ffmpeg" and not find_ffmpeg():
        parser.error("--decoder ffmpeg: ffmpeg not found (install FFmpeg or `pip install imageio-ffmpeg`)")
    if args.output_dir and not os.path.isdir(args.output_dir):
        parser.error(f"output folder '{args.output_dir}' not found")
    return args


def output_paths(video_path, output_dir=None):
    folder, name = os.path.split(video_path)
    base = os.path.join(output_dir if output_dir is not None else folder, os.path.splitext(name)[0])
    return OutputPaths(
        video=base + "_analyzed.mp4",
        summary_csv=base + "_histogram.csv",
        frametimes_csv=base + "_frametimes.csv",
        report=base + "_report.html",
    )


def capture_basename(output_dir=None):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(output_dir or os.getcwd(), f"capture_{stamp}.mp4")


def choose_target_fps(requested, stream, label="video"):
    """The target FPS: as requested, else from the video's metadata, else 60."""
    reported = stream.fps()
    if requested:
        if reported and abs(reported - requested) > 0.5:
            print(f"Warning: the {label} is {reported:.2f} fps but the target is {requested} fps. "
                  f"Results are only meaningful when the video was recorded at the target frame rate.")
        return requested
    if reported and MIN_FPS <= round(reported) <= MAX_FPS:
        fps = round(reported)
        note = f" (read from the {label}: {reported:.3f})" if abs(reported - fps) > 0.001 else \
            f" (read from the {label})"
        print(f"Target FPS: {fps}{note}")
        return fps
    print(f"Target FPS: {DEFAULT_FPS} (the {label} doesn't say; pass the fps argument to change)")
    return DEFAULT_FPS


def make_prepare(canvas_size):
    """Per-frame work done on the decoding thread (OpenCV decoder): the canvas
    for drawing, and the analysis image."""
    def prepare(frame):
        h, w = frame.shape[:2]
        canvas = fit_to_canvas(frame, *canvas_size) if canvas_size else None
        return Prepared(canvas, analysis_image(frame), (w, h))
    return prepare


def open_stream(source, canvas_size, args):
    """Open a file with FFmpeg (fast) or OpenCV, or a capture device with OpenCV.
    read() returns Prepared frames; canvas is None when canvas_size is None."""
    live = isinstance(source, int)
    use_ffmpeg = not live and (args.decoder == "ffmpeg" or (args.decoder == "auto" and find_ffmpeg()))
    if not use_ffmpeg:
        return FileVideoStream(source, transform=make_prepare(canvas_size), size=None,
                               hw_decode=args.hw_decode, capture_size=args.capture_size,
                               capture_fps=args.fps if live else None)
    if canvas_size:
        stream = FfmpegVideoStream(source, canvas_size, hw_decode=args.hw_decode)
        stream.transform = lambda f: Prepared(f, analysis_image(f), stream.src_size)
    else:  # Analysis only: FFmpeg outputs just the small greyscale image
        stream = FfmpegVideoStream(source, ANALYSIS_SIZE, gray=True, hw_decode=args.hw_decode)
        stream.transform = lambda f: Prepared(None, f, stream.src_size)
    return stream


def report_stream_error(stream):
    error = getattr(stream, "error", None)
    if error:
        print(f"Error: decoding failed: {error}", file=sys.stderr)


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


class Progress:
    def __init__(self, total):
        self.total = total
        self.last = time.perf_counter()

    def tick(self, done):
        now = time.perf_counter()
        if now - self.last >= 5:
            self.last = now
            pct = f" ({done / self.total:.0%})" if self.total else ""
            print(f"Processed {done} frames{pct}")


def open_output_video(path, fps, encoder):
    try:
        writer = open_writer(path, fps, CANVAS_SIZE, encoder)
        print(f"Output will be saved to: {path} [{writer.name}]")
        return writer
    except OSError as e:
        print(f"Warning: could not create output video: {e}")
        return None


def close_output_video(writer, path):
    if writer is None:
        return
    try:
        writer.release()
        print(f"Video saved to: {path}")
    except Exception as e:
        print(f"Warning: video encoding failed: {e}")


def save_results(video_label, paths, analyser, recorder, write_html=True):
    print(f"Per-frame frametimes exported to: {paths.frametimes_csv}")
    if not analyser.unique_frames:
        print("No new frames were found, so no summary was written")
        return
    try:
        export_summary_csv(paths.summary_csv, video_label, analyser)
        print(f"Histogram data exported to: {paths.summary_csv}")
        if write_html:
            write_report(paths.report, video_label, analyser, recorder)
            print(f"Report saved to: {paths.report}")
    except OSError as e:
        print(f"Warning: could not write results: {e}")


def build_mask(args, src_size):
    try:
        return AnalysisMask(src_size[0], src_size[1], args.roi, args.ignore)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return None


def new_analyser(args, fps, mask, **kwargs):
    return FrametimeAnalyser(fps, detect_stutters=args.detect_stutters, detect_tears=args.detect_tears,
                             picture_rows=mask.picture_rows, min_diff=args.min_diff,
                             static_after=args.static_after, **kwargs)


def measure(analyser, frame, prev, mask):
    change = frame_change(frame.gray, prev.gray, mask, rows=True)
    return analyser.process(change.mean, change.rows, change.peak)


TEARING_WARNING = ("Warning: {label} looks like it was recorded with V-Sync off ({pct:.0f}% of frames torn). "
                   "Each torn frame is counted as a new frame, so FPS is overstated. "
                   "Re-run with --detect-tears for accurate results.")


class TearingWarning:
    """Warn once, a few seconds in, if the video tears without --detect-tears."""

    def __init__(self, label):
        self.label = label
        self.shown = False

    def check(self, a, final=False):
        if self.shown or not a.looks_torn() or (not final and a.frames_processed < 5 * a.target_fps):
            return
        self.shown = True
        pct = 100.0 * a.tearing_share()
        print(TEARING_WARNING.format(label=self.label, pct=pct))


# --- Single video ----------------------------------------------------------------

def run(args):
    live = args.capture is not None
    source = args.capture if live else args.video
    label = f"capture device {args.capture}" if live else args.video
    paths = output_paths(capture_basename(args.output_dir) if live else args.video, args.output_dir)
    display = not args.no_display
    draw = display or not args.no_video

    print(f"Loading {'capture device' if live else 'video'}: {source}")
    fvs = open_stream(source, CANVAS_SIZE if draw else None, args)
    if not fvs.is_opened():
        print(f"Error: could not open {label}", file=sys.stderr)
        return 1
    print(f"Decoder: {fvs.name}")
    if fvs.note:
        print(f"Note: {fvs.note}")
    target_fps = choose_target_fps(args.fps, fvs, "capture device" if live else "video")
    if args.fps:
        print(f"Target FPS: {target_fps}")
    for name, on in (("Delta Plot", args.delta_plot), ("Stutter Detection", args.detect_stutters),
                     ("Tear Detection", args.detect_tears)):
        print(f"{name}: {'Enabled' if on else 'Disabled'}")
    if args.title:
        print(f"Title: {args.title}")

    fvs.start()
    prev = fvs.read()
    if prev is None:
        fvs.stop()
        report_stream_error(fvs)
        print("Error: the video contains no frames", file=sys.stderr)
        return 1
    mask = build_mask(args, prev.src_size)
    if mask is None:
        fvs.stop()
        return 1

    analyser = new_analyser(args, target_fps, mask)
    overlay = None
    if draw:
        watermark = load_watermark(args.watermark) if args.watermark else None
        overlay = Overlay(target_fps, args.delta_plot, args.detect_stutters, args.detect_tears,
                          args.title, watermark, mask if (mask.roi or mask.ignore) else None)
    writer = None if args.no_video else open_output_video(paths.video, target_fps, args.encoder)
    recorder = Recorder(paths.frametimes_csv)
    pacer = Pacer(target_fps) if display and not live else None
    progress = Progress(fvs.frame_count())
    tearing = TearingWarning("this video")

    start = time.perf_counter()
    try:
        while True:
            frame = fvs.read()
            if frame is None:
                break
            for f in measure(analyser, frame, prev, mask):
                recorder.add(f)
            prev = frame
            tearing.check(analyser)

            if overlay is not None:
                annotated = frame.canvas  # Each decoded frame is a new array, so draw on it directly
                overlay.draw(annotated, analyser)
                if writer:
                    writer.write(annotated)
                if display:
                    if pacer:
                        pacer.wait()
                    cv2.imshow(WINDOW_NAME, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Stopped early (q pressed)")
                        break
            if not display:
                progress.tick(analyser.frames_processed)
    except KeyboardInterrupt:
        print("Interrupted; saving results so far")
    finally:
        fvs.stop()
        recorder.close()
        if display:
            cv2.destroyAllWindows()

    report_stream_error(fvs)
    elapsed = time.perf_counter() - start
    print(f"Processed {analyser.frames_processed} frames in {elapsed:.1f}s "
          f"({analyser.frames_processed / elapsed if elapsed else 0:.0f} fps)")
    if fvs.dropped:
        print(f"Warning: {fvs.dropped} captured frames were dropped because processing fell behind. "
              f"Results are unreliable; try --no-display or --no-video.")
    close_output_video(writer, paths.video)
    tearing.check(analyser, final=True)
    save_results(label, paths, analyser, recorder, not args.no_report)
    return 0


# --- Compare two videos --------------------------------------------------------

def compare_paths(video_a, video_b, output_dir=None):
    folder, name_a = os.path.split(video_a)
    base_a = os.path.splitext(name_a)[0]
    base_b = os.path.splitext(os.path.basename(video_b))[0]
    return os.path.join(output_dir if output_dir is not None else folder, f"{base_a}_vs_{base_b}_compare.mp4")


def run_compare(args):
    display = not args.no_display
    draw = display or not args.no_video
    videos = (args.video, args.compare)
    labels = args.labels or tuple(os.path.splitext(os.path.basename(v))[0] for v in videos)
    canvas = CompareOverlay.VIDEO_SIZE if draw else None

    streams = []
    for v in videos:
        fvs = open_stream(v, canvas, args)
        if not fvs.is_opened():
            print(f"Error: could not open video '{v}'", file=sys.stderr)
            return 1
        streams.append(fvs)
    target_fps = choose_target_fps(args.fps, streams[0], "first video")
    other = streams[1].fps()
    if other and abs(other - target_fps) > 0.5:
        print(f"Warning: the second video is {other:.2f} fps but the target is {target_fps} fps")

    for s in streams:
        s.start()
    prevs = [s.read() for s in streams]
    if any(p is None for p in prevs):
        print("Error: a video contains no frames", file=sys.stderr)
        for s in streams:
            s.stop()
        return 1
    masks = [build_mask(args, p.src_size) for p in prevs]
    if any(m is None for m in masks):
        for s in streams:
            s.stop()
        return 1

    analysers = [new_analyser(args, target_fps, m, frametime_samples=CompareOverlay.FRAMETIME_SAMPLES)
                 for m in masks]
    paths = [output_paths(v, args.output_dir) for v in videos]
    recorders = [Recorder(p.frametimes_csv) for p in paths]
    overlay = None
    if draw:
        watermark = load_watermark(args.watermark) if args.watermark else None
        overlay = CompareOverlay(target_fps, labels, args.detect_stutters, args.detect_tears,
                                 args.title, watermark)
    video_path = compare_paths(*videos, args.output_dir)
    writer = None if args.no_video else open_output_video(video_path, target_fps, args.encoder)
    pacer = Pacer(target_fps) if display else None
    progress = Progress(min(s.frame_count() for s in streams))

    start = time.perf_counter()
    try:
        while True:
            frames = [s.read() for s in streams]
            if any(f is None for f in frames):
                break  # Stop at the end of the shorter video
            for i in range(2):
                for f in measure(analysers[i], frames[i], prevs[i], masks[i]):
                    recorders[i].add(f)
            prevs = frames

            if overlay is not None:
                annotated = overlay.compose(frames[0].canvas, frames[1].canvas)
                overlay.draw(annotated, *analysers)
                if writer:
                    writer.write(annotated)
                if display:
                    pacer.wait()
                    cv2.imshow(WINDOW_NAME, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Stopped early (q pressed)")
                        break
            if not display:
                progress.tick(analysers[0].frames_processed)
    except KeyboardInterrupt:
        print("Interrupted; saving results so far")
    finally:
        for s in streams:
            s.stop()
        for r in recorders:
            r.close()
        if display:
            cv2.destroyAllWindows()

    for s in streams:
        report_stream_error(s)
    elapsed = time.perf_counter() - start
    print(f"Processed {analysers[0].frames_processed} frame pairs in {elapsed:.1f}s")
    close_output_video(writer, video_path)
    for v, p, a, r in zip(videos, paths, analysers, recorders):
        TearingWarning(f"'{os.path.basename(v)}'").check(a, final=True)
        save_results(v, p, a, r, not args.no_report)
    return 0


def main(argv=None):
    args = parse_args(argv)
    return run_compare(args) if args.compare else run(args)
