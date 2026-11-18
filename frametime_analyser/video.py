"""Video input (files and capture devices) and output (OpenCV or FFmpeg)."""
import os
import shutil
import subprocess
import sys
import time
from collections import namedtuple
from queue import Empty, Full, Queue
from threading import Thread

import cv2
import numpy as np

# Every frame is fitted onto a canvas of this size for the overlay, so graphs
# (drawn at fixed positions) always line up whatever the source size.
CANVAS_SIZE = (1280, 720)

Fit = namedtuple("Fit", "scale x y width height")


def canvas_fit(src_width, src_height, width, height):
    """Where a src_width x src_height image lands when fitted inside width x height."""
    scale = min(width / src_width, height / src_height)
    new_w = max(1, min(width, round(src_width * scale)))
    new_h = max(1, min(height, round(src_height * scale)))
    return Fit(scale, (width - new_w) // 2, (height - new_h) // 2, new_w, new_h)


def fit_to_canvas(image, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1], inter=cv2.INTER_LINEAR):
    """Scale an image to fit inside width x height, keeping its aspect ratio,
    and centre it on a black canvas of exactly that size (letterbox/pillarbox)."""
    (h, w) = image.shape[:2]
    if (w, h) == (width, height):
        return image
    fit = canvas_fit(w, h, width, height)
    resized = cv2.resize(image, (fit.width, fit.height), interpolation=inter)
    if (fit.width, fit.height) == (width, height):
        return resized
    canvas = np.zeros((height, width) + image.shape[2:], dtype=image.dtype)
    canvas[fit.y:fit.y + fit.height, fit.x:fit.x + fit.width] = resized
    return canvas


def open_capture(source, hw_decode=False, capture_size=None, capture_fps=None):
    """Open a video file (str) or capture device (int). Returns (VideoCapture, note)."""
    if isinstance(source, int):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, backend)
        if capture_size:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_size[1])
        if capture_fps:
            cap.set(cv2.CAP_PROP_FPS, capture_fps)
        return cap, None

    if hw_decode:
        params = [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
        if cap.isOpened():
            accel = int(cap.get(cv2.CAP_PROP_HW_ACCELERATION))
            if accel:
                return cap, f"hardware decoding enabled ({_ACCEL_NAMES.get(accel, accel)})"
            return cap, "hardware decoding not available for this video; using software decoding"
        cap.release()
    return cv2.VideoCapture(source), None


_ACCEL_NAMES = {
    getattr(cv2, name): name.replace("VIDEO_ACCELERATION_", "")
    for name in ("VIDEO_ACCELERATION_D3D11", "VIDEO_ACCELERATION_VAAPI", "VIDEO_ACCELERATION_MFX")
    if hasattr(cv2, name)
}


class FileVideoStream:
    name = "opencv"

    """Decode a video file or capture device on a background thread.

    Each frame is fitted onto a `size` canvas (if given) and then passed
    through `transform` (if given) on the decoding thread; read() returns the
    result. Pass an int source for a capture device: then frames are dropped
    (and counted in `dropped`) rather than waiting when the queue is full,
    since a live device can't be paused.
    """

    def __init__(self, source, transform=None, queue_size=196, size=CANVAS_SIZE,
                 hw_decode=False, capture_size=None, capture_fps=None):
        self.live = isinstance(source, int)
        self.stream, self.note = open_capture(source, hw_decode, capture_size, capture_fps)
        self.stopped = False
        self.transform = transform
        self.size = size
        self.dropped = 0
        self.Q = Queue(maxsize=queue_size * (3 if self.live else 1))
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def is_opened(self):
        return self.stream.isOpened()

    def fps(self):
        """Frame rate reported by the container or device (0 if unknown)."""
        return self.stream.get(cv2.CAP_PROP_FPS) or 0

    def frame_count(self):
        """Number of frames reported by the container (0 if unknown or live)."""
        return 0 if self.live else int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            if self.Q.full() and not self.live:
                time.sleep(0.01)  # Queue is full: rest for 10ms
                continue

            grabbed, frame = self.stream.read()
            if not grabbed:
                break  # End of the video (or the device went away)

            if self.size:
                frame = fit_to_canvas(frame, *self.size)
            if self.transform:
                frame = self.transform(frame)
            try:
                self.Q.put_nowait(frame)
            except Full:
                self.dropped += 1  # Live only: processing has fallen behind

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

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()


class FfmpegVideoStream:
    """Decode a video file with an FFmpeg process, which also scales each frame.

    Much faster than OpenCV's decoder: FFmpeg decodes with all CPU cores and
    converts straight from the video's YUV pixels to the size needed. Frames
    are fitted onto an `out_size` canvas, as BGR (gray=False) or greyscale
    (gray=True). Same interface as FileVideoStream.
    """

    live = False
    name = "ffmpeg"

    def __init__(self, path, out_size, gray=False, transform=None, queue_size=64,
                 hw_decode=False, ffmpeg=None):
        self.transform = transform
        self.stopped = False
        self.dropped = 0
        self.note = None
        self.error = None
        self.proc = None
        cap = cv2.VideoCapture(path)
        self._opened = cap.isOpened()
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 0
        self._count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.src_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        if self._opened and min(self.src_size) <= 0:
            self._opened = False

        self.out_size = out_size
        self.channels = 1 if gray else 3
        self.frame_bytes = out_size[0] * out_size[1] * self.channels
        if self._opened:
            fit = canvas_fit(*self.src_size, *out_size)
            vf = (f"scale={fit.width}:{fit.height}:flags={'area' if gray else 'bilinear'},"
                  f"pad={out_size[0]}:{out_size[1]}:{fit.x}:{fit.y}:black,"
                  f"format={'gray' if gray else 'bgr24'}")
            cmd = [ffmpeg or find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin"]
            if hw_decode:
                cmd += ["-hwaccel", "auto"]
                self.note = "hardware decoding requested (FFmpeg -hwaccel auto)"
            cmd += ["-i", path, "-an", "-sn", "-vf", vf, "-fps_mode", "passthrough",
                    "-f", "rawvideo", "-"]
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            # Unbuffered + readinto() is ~3x faster than buffered read() for big frames
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         bufsize=0, creationflags=flags)
        self.Q = Queue(maxsize=queue_size)
        self.thread = Thread(target=self.update, daemon=True)

    def is_opened(self):
        return self._opened

    def fps(self):
        return self._fps

    def frame_count(self):
        return self._count

    def start(self):
        self.thread.start()
        return self

    def update(self):
        shape = (self.out_size[1], self.out_size[0]) + ((self.channels,) if self.channels > 1 else ())
        out = self.proc.stdout
        while not self.stopped:
            # A new array per frame: the reader owns it and may draw on it
            frame = np.empty(shape, np.uint8)
            view = memoryview(frame).cast("B")
            got = 0
            while got < self.frame_bytes:
                n = out.readinto(view[got:])
                if not n:
                    break
                got += n
            if got < self.frame_bytes:
                break
            if self.transform:
                frame = self.transform(frame)
            while not self.stopped:
                try:
                    self.Q.put(frame, timeout=0.1)
                    break
                except Full:
                    pass
        self._finish()
        self.stopped = True

    def _finish(self):
        if self.proc.poll() is None:
            if self.stopped:
                self.proc.kill()
            self.proc.wait()
        err = self.proc.stderr.read().decode(errors="replace").strip()
        if self.proc.returncode not in (0, None) and not self.stopped:
            self.error = err or f"ffmpeg exited with code {self.proc.returncode}"

    def read(self):
        while True:
            try:
                return self.Q.get(timeout=0.1)
            except Empty:
                if self.stopped and self.Q.empty():
                    return None

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        elif self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()


# --- Writers -----------------------------------------------------------------

# Extra FFmpeg arguments per encoder: fast presets with good quality
FFMPEG_ENCODERS = {
    "libx264": ["-preset", "veryfast", "-crf", "20"],
    "h264_nvenc": ["-preset", "p4", "-rc", "vbr", "-cq", "23"],
    "hevc_nvenc": ["-preset", "p4", "-rc", "vbr", "-cq", "25"],
    "h264_qsv": ["-preset", "faster", "-global_quality", "23"],
    "h264_amf": ["-quality", "speed", "-rc", "cqp", "-qp_i", "22", "-qp_p", "22"],
}
ENCODER_CHOICES = ["auto", "opencv"] + list(FFMPEG_ENCODERS)


def find_ffmpeg():
    """Path to an ffmpeg executable: on PATH, else from the imageio-ffmpeg package."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class OpenCVWriter:
    """MPEG-4 Part 2 ('mp4v') through OpenCV. Always available, but slow and lower quality."""

    name = "opencv (mp4v)"

    def __init__(self, path, fps, size):
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
        if not self.writer.isOpened():
            raise OSError(f"could not create '{path}'")

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()


class FfmpegWriter:
    """Pipe raw frames to an FFmpeg process (H.264/H.265, including GPU encoders)."""

    def __init__(self, path, fps, size, encoder="libx264", ffmpeg=None):
        ffmpeg = ffmpeg or find_ffmpeg()
        if not ffmpeg:
            raise OSError("ffmpeg not found (install FFmpeg or `pip install imageio-ffmpeg`)")
        self.name = f"ffmpeg ({encoder})"
        self.size = size
        cmd = [ffmpeg, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
               "-r", str(fps), "-i", "-",
               "-c:v", encoder, *FFMPEG_ENCODERS.get(encoder, []),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, creationflags=flags)

    def write(self, frame):
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).data)
        except (BrokenPipeError, OSError):
            raise OSError(self._error() or "ffmpeg stopped unexpectedly") from None

    def _error(self):
        try:
            self.proc.wait(timeout=10)
            return self.proc.stderr.read().decode(errors="replace").strip()
        except Exception:
            return ""

    def release(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.wait()
        err = self.proc.stderr.read().decode(errors="replace").strip()
        self.proc.stderr.close()
        if self.proc.returncode != 0:
            raise OSError(err or f"ffmpeg exited with code {self.proc.returncode}")


class AsyncWriter:
    """Run another writer on a background thread so encoding overlaps with drawing.

    Frames passed to write() must not be modified afterwards. If the writer
    fails, later writes are ignored and the error is raised from release().
    """

    def __init__(self, writer, queue_size=32):
        self.writer = writer
        self.name = writer.name
        self.error = None
        self.Q = Queue(maxsize=queue_size)
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            frame = self.Q.get()
            if frame is None:
                return
            if self.error is None:
                try:
                    self.writer.write(frame)
                except Exception as e:  # Keep draining the queue so write() never blocks forever
                    self.error = e

    def write(self, frame):
        if self.error is None:
            self.Q.put(frame)

    def release(self):
        self.Q.put(None)
        self.thread.join()
        try:
            self.writer.release()
        except Exception as e:
            self.error = self.error or e
        if self.error:
            raise self.error


def open_writer(path, fps, size, encoder="auto"):
    """Open a video writer, run on a background thread.

    "auto" uses FFmpeg with libx264 when FFmpeg is available, else OpenCV.
    """
    if encoder == "auto":
        encoder = "libx264" if find_ffmpeg() else "opencv"
    if encoder == "opencv":
        writer = OpenCVWriter(path, fps, size)
    else:
        writer = FfmpegWriter(path, fps, size, encoder)
    return AsyncWriter(writer)


def remove_if_empty(path):
    try:
        if os.path.getsize(path) == 0:
            os.remove(path)
    except OSError:
        pass
