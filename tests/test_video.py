"""Tests for frametime_analyser.video: decoding and encoding"""
import unittest

import cv2
import numpy as np

from frametime_analyser.video import (
    AsyncWriter,
    FfmpegVideoStream,
    FfmpegWriter,
    FileVideoStream,
    OpenCVWriter,
    canvas_fit,
    fit_to_canvas,
    open_writer,
)
from tests.helpers import TempDirTestCase, gray_level, needs_ffmpeg, write_video


def read_all(stream):
    frames = []
    while True:
        f = stream.read()
        if f is None:
            return frames
        frames.append(f)


def frame_count(path):
    cap = cv2.VideoCapture(path)
    n = 0
    while cap.read()[0]:
        n += 1
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return n, size


class TestFitToCanvas(unittest.TestCase):
    """Frames of any shape are fitted onto the fixed overlay canvas"""

    def test_sizes(self):
        for h, w in [(1080, 1920), (720, 1280), (1440, 3440), (1080, 1440), (1920, 1080), (100, 100)]:
            out = fit_to_canvas(np.full((h, w, 3), 200, np.uint8), 1280, 720)
            self.assertEqual(out.shape, (720, 1280, 3), (h, w))

    def test_ultrawide_is_letterboxed(self):
        out = fit_to_canvas(np.full((1440, 3440, 3), 200, np.uint8), 1280, 720)
        self.assertEqual(out[0, 640, 0], 0)      # black bar at the top
        self.assertEqual(out[360, 640, 0], 200)  # picture in the middle

    def test_4_3_is_pillarboxed(self):
        out = fit_to_canvas(np.full((1080, 1440, 3), 200, np.uint8), 1280, 720)
        self.assertEqual(out[360, 0, 0], 0)
        self.assertEqual(out[360, 640, 0], 200)

    def test_canvas_size_is_unchanged(self):
        img = np.zeros((720, 1280, 3), np.uint8)
        self.assertIs(fit_to_canvas(img, 1280, 720), img)

    def test_canvas_fit(self):
        self.assertEqual(canvas_fit(1920, 1080, 1280, 720)[1:], (0, 0, 1280, 720))
        self.assertEqual(canvas_fit(1440, 1080, 1280, 720)[1:], (160, 0, 960, 720))


class TestFileVideoStream(TempDirTestCase):
    """OpenCV decoding on a background thread"""

    def test_invalid_path(self):
        fvs = FileVideoStream("nonexistent_file.mp4")
        self.assertFalse(fvs.is_opened())
        self.assertFalse(fvs.live)

    def test_queue_and_thread(self):
        fvs = FileVideoStream("test.mp4", queue_size=100)
        self.assertEqual(fvs.Q.maxsize, 100)
        self.assertTrue(fvs.thread.daemon)
        self.assertFalse(fvs.stopped)

    def test_reads_all_frames_then_none(self):
        path = self.path("clip.mp4")
        write_video(path, list(range(30)))
        # A small queue makes the decoder wait for the reader
        fvs = FileVideoStream(path, queue_size=4).start()
        frames = read_all(fvs)
        fvs.stop()
        self.assertEqual(len(frames), 30)
        self.assertEqual(frames[0].shape, (720, 1280, 3))
        self.assertFalse(fvs.thread.is_alive())

    def test_transform_runs_on_decoder_thread(self):
        path = self.path("clip.mp4")
        write_video(path, list(range(5)))
        fvs = FileVideoStream(path, transform=lambda f: f.shape, size=(64, 36)).start()
        self.assertEqual(read_all(fvs), [(36, 64, 3)] * 5)
        fvs.stop()

    def test_stop_early(self):
        path = self.path("clip.mp4")
        write_video(path, list(range(60)))
        fvs = FileVideoStream(path, queue_size=2).start()
        fvs.read()
        fvs.stop()
        self.assertFalse(fvs.thread.is_alive())


@needs_ffmpeg
class TestFfmpegVideoStream(TempDirTestCase):
    """FFmpeg decoding, scaled by FFmpeg"""

    def test_bgr_canvas(self):
        path = self.path("clip.mp4")
        ids = list(range(40))
        write_video(path, ids)
        s = FfmpegVideoStream(path, (1280, 720)).start()
        frames = read_all(s)
        s.stop()
        self.assertEqual(len(frames), 40)
        self.assertEqual(frames[0].shape, (720, 1280, 3))
        self.assertTrue(frames[0].flags.writeable)
        self.assertAlmostEqual(int(frames[3][360, 640, 0]), gray_level(3), delta=4)
        self.assertEqual((s.src_size, round(s.fps()), s.frame_count()), ((320, 180), 60, 40))
        self.assertIsNone(s.error)

    def test_gray_letterboxed(self):
        path = self.path("tall.mp4")
        write_video(path, list(range(10)), size=(180, 320))  # Portrait
        s = FfmpegVideoStream(path, (128, 72), gray=True).start()
        frames = read_all(s)
        s.stop()
        self.assertEqual(frames[0].shape, (72, 128))
        self.assertEqual(frames[0][36, 0], 0)  # Pillarbox bar
        self.assertGreater(frames[0][36, 64], 20)

    def test_invalid_file(self):
        bad = self.path("bad.mp4")
        with open(bad, "wb") as f:
            f.write(b"not a video")
        s = FfmpegVideoStream(bad, (128, 72), gray=True)
        self.assertFalse(s.is_opened())

    def test_stop_early(self):
        path = self.path("clip.mp4")
        write_video(path, list(range(120)))
        s = FfmpegVideoStream(path, (1280, 720), queue_size=2).start()
        s.read()
        s.stop()
        self.assertFalse(s.thread.is_alive())
        self.assertIsNone(s.error)


class FailingWriter:
    name = "failing"

    def __init__(self):
        self.written = 0

    def write(self, frame):
        self.written += 1
        if self.written == 3:
            raise OSError("disk full")

    def release(self):
        pass


class TestWriters(TempDirTestCase):
    frames = [np.full((720, 1280, 3), v, np.uint8) for v in (0, 80, 160, 240) * 5]

    def check_video(self, path):
        n, size = frame_count(path)
        self.assertEqual((n, size), (len(self.frames), (1280, 720)))

    def test_opencv_writer(self):
        path = self.path("out.mp4")
        w = AsyncWriter(OpenCVWriter(path, 60, (1280, 720)))
        for f in self.frames:
            w.write(f)
        w.release()
        self.check_video(path)

    @needs_ffmpeg
    def test_ffmpeg_libx264(self):
        path = self.path("out.mp4")
        w = open_writer(path, 60, (1280, 720), "libx264")
        self.assertIn("libx264", w.name)
        for f in self.frames:
            w.write(f)
        w.release()
        self.check_video(path)

    @needs_ffmpeg
    def test_ffmpeg_bad_encoder_reports_error(self):
        w = AsyncWriter(FfmpegWriter(self.path("out.mp4"), 60, (1280, 720), "no_such_encoder"))
        for f in self.frames:
            w.write(f)
        with self.assertRaises(OSError):
            w.release()

    def test_auto_writer(self):
        path = self.path("out.mp4")
        w = open_writer(path, 60, (1280, 720), "auto")
        for f in self.frames:
            w.write(f)
        w.release()
        self.check_video(path)

    def test_async_writer_error_is_raised_on_release(self):
        inner = FailingWriter()
        w = AsyncWriter(inner, queue_size=2)
        for f in self.frames:
            w.write(f)  # Never blocks, even after the failure
        with self.assertRaisesRegex(OSError, "disk full"):
            w.release()
        self.assertEqual(inner.written, 3)


if __name__ == "__main__":
    unittest.main()
