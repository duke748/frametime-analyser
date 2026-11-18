"""Integration tests for the frametime analyser"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_SCRIPT = os.path.join(REPO_ROOT, "Frametime Analyser.py")

sys.path.insert(0, REPO_ROOT)

import frametime_analyser as fa  # noqa: E402


def write_video(path, frame_ids, fps=60, size=(320, 180)):
    """Write a video where frame i shows picture frame_ids[i].

    Consecutive equal ids are exact repeats, like the duplicate frames a
    capture card records when the game misses a refresh.
    """
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
    for pid in frame_ids:
        level = 30 + (pid * 37) % 200
        out.write(np.full((size[1], size[0], 3), level, np.uint8))
    out.release()


def expected_frametime_counts(frame_ids):
    """Frametimes (in video frames) the analyser should report for frame_ids."""
    counts, last_unique = [], 0
    for i in range(1, len(frame_ids)):
        if frame_ids[i] != frame_ids[i - 1]:
            counts.append(i - last_unique)
            last_unique = i
    return counts


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)


class TestVideoGeneration(TempDirTestCase):
    """The synthetic videos used below are readable by OpenCV"""

    def test_video_creation(self):
        path = os.path.join(self.temp_dir, "test.mp4")
        write_video(path, list(range(60)))
        cap = cv2.VideoCapture(path)
        self.assertTrue(cap.isOpened())
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 60)
        self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 60, places=1)
        cap.release()


class TestArgumentParsing(TempDirTestCase):
    """Command line parsing (in-process, via fa.parse_args)"""

    def setUp(self):
        super().setUp()
        self.video = os.path.join(self.temp_dir, "clip.mp4")
        open(self.video, "w").close()

    def parse(self, *args):
        return fa.parse_args(list(args))

    def assert_rejected(self, *args):
        with self.assertRaises(SystemExit) as ctx:
            self.parse(*args)
        self.assertEqual(ctx.exception.code, 2)

    def test_defaults(self):
        a = self.parse(self.video)
        self.assertEqual(a.fps, 60)
        self.assertFalse(a.delta_plot or a.detect_stutters or a.no_display)
        self.assertIsNone(a.watermark)
        self.assertIsNone(a.title)

    def test_any_target_fps(self):
        for fps in (30, 60, 120, 144, 240):
            self.assertEqual(self.parse(self.video, str(fps)).fps, fps)

    def test_invalid_fps(self):
        for fps in ("0", "-5", "1001", "fast", "59.94"):
            self.assert_rejected(self.video, fps)

    def test_legacy_single_dash_flags(self):
        wm = os.path.join(self.temp_dir, "wm.png")
        cv2.imwrite(wm, np.zeros((8, 8, 3), np.uint8))
        a = self.parse(self.video, "120", "-delta-plot", "-detect-stutters",
                       "-watermark", wm, "-title", "My Game")
        self.assertTrue(a.delta_plot and a.detect_stutters)
        self.assertEqual((a.fps, a.watermark, a.title), (120, wm, "My Game"))

    def test_double_dash_flags(self):
        a = self.parse(self.video, "--delta-plot", "--detect-stutters", "--no-display",
                       "--title", "T")
        self.assertTrue(a.delta_plot and a.detect_stutters and a.no_display)

    def test_fps_after_options(self):
        """The fps value can come after options, as in the original script"""
        a = self.parse(self.video, "-delta-plot", "120")
        self.assertEqual(a.fps, 120)
        self.assertTrue(a.delta_plot)

    def test_unknown_option(self):
        self.assert_rejected(self.video, "--bogus")

    def test_missing_video_file(self):
        self.assert_rejected(os.path.join(self.temp_dir, "missing.mp4"))

    def test_missing_watermark_file(self):
        self.assert_rejected(self.video, "-watermark", os.path.join(self.temp_dir, "nope.png"))

    def test_watermark_requires_path(self):
        self.assert_rejected(self.video, "-watermark")

    def test_title_requires_text(self):
        self.assert_rejected(self.video, "-title")


class TestLegacyEntryPoint(TempDirTestCase):
    """'Frametime Analyser.py' still works as the entry point"""

    def run_script(self, *args):
        return subprocess.run([sys.executable, LEGACY_SCRIPT, *args], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=120)

    def test_help(self):
        result = self.run_script("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--no-display", result.stdout)

    def test_rejects_bad_arguments(self):
        result = self.run_script(os.path.join(self.temp_dir, "missing.mp4"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("not found", result.stderr)

    def test_headless_run(self):
        video = os.path.join(self.temp_dir, "clip.mp4")
        write_video(video, list(range(40)))
        result = self.run_script(video, "--no-display")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.path.exists(fa.output_paths(video).summary_csv))


class TestOutputPaths(unittest.TestCase):
    def test_output_paths(self):
        p = fa.output_paths("C:/Videos/gameplay.mp4")
        self.assertEqual(p.video, "C:/Videos/gameplay_analyzed.mp4")
        self.assertEqual(p.summary_csv, "C:/Videos/gameplay_histogram.csv")
        self.assertEqual(p.frametimes_csv, "C:/Videos/gameplay_frametimes.csv")


class TestFrametimeAnalyser(unittest.TestCase):
    """Duplicate detection and statistics, fed with frame differences directly"""

    def feed(self, analyser, pattern):
        """pattern: 'u' for a new frame, 'd' for a repeat. Returns reported frametimes."""
        return [ft for ft in (analyser.process(20.0 if c == "u" else 0.1) for c in pattern)
                if ft is not None]

    def test_smooth_video(self):
        a = fa.FrametimeAnalyser(60)
        fts = self.feed(a, "u" * 120)
        self.assertEqual(len(fts), 120)
        self.assertTrue(all(abs(ft - 1000 / 60) < 1e-9 for ft in fts))
        self.assertEqual(a.fps, 60)
        self.assertAlmostEqual(a.window.average_fps(), 60)

    def test_repeated_frames_lengthen_frametime(self):
        a = fa.FrametimeAnalyser(60)
        fts = self.feed(a, "uuduuddu")
        self.assertEqual([round(ft / (1000 / 60)) for ft in fts], [1, 1, 2, 1, 3])
        self.assertEqual(a.unique_frames, 5)
        self.assertEqual(a.frames_processed, 8)

    def test_half_rate_video(self):
        a = fa.FrametimeAnalyser(60)
        self.feed(a, "ud" * 60)
        self.assertEqual(a.fps, 30)
        # The first unique frame is only one frame long, so allow a little slack
        self.assertAlmostEqual(a.window.average_fps(), 30, delta=0.3)
        self.assertAlmostEqual(a.window.mean(), 2000 / 60, delta=0.3)

    def test_120fps(self):
        a = fa.FrametimeAnalyser(120)
        fts = self.feed(a, "u" * 240)
        self.assertAlmostEqual(fts[-1], 1000 / 120)
        self.assertEqual(a.fps, 120)

    def test_stutter_detection(self):
        """2.5x target is moderate, 4x is severe (60fps: 41.7ms / 66.7ms)"""
        a = fa.FrametimeAnalyser(60, detect_stutters=True)
        # 2 frames (33ms): none; 3 frames (50ms): moderate; 4 frames (66.7ms): severe
        self.feed(a, "uu" + "du" + "ddu" + "dddu" + "u")
        self.assertEqual([s.severity for s in a.stutters], ["moderate", "severe"])
        self.assertEqual((a.moderate_stutters, a.severe_stutters), (1, 1))
        self.assertEqual([s.unique_index for s in a.stutters], [3, 4])
        self.assertEqual([s.video_frame for s in a.stutters], [7, 11])

    def test_stutters_not_recorded_when_disabled(self):
        a = fa.FrametimeAnalyser(60)
        self.feed(a, "u" + "d" * 10 + "u")
        self.assertEqual(a.stutters, [])

    def test_frame_delta(self):
        a = fa.FrametimeAnalyser(60)
        self.feed(a, "uudu")
        ft = 1000 / 60
        self.assertEqual(list(a.delta_graph)[-3:], [0, 0, ft])

    def test_summary(self):
        a = fa.FrametimeAnalyser(60)
        self.feed(a, ("u" * 98 + "du") * 2)
        s = a.summary()
        self.assertEqual(s["unique_frames"], 198)
        self.assertAlmostEqual(s["duration_s"], 200 / 60)
        self.assertAlmostEqual(s["p99_frametime"], 2000 / 60)
        self.assertAlmostEqual(s["low_1_fps"], 30)

    def test_overlay_draws_on_canvas(self):
        """Every overlay option draws without error, with and without data"""
        img_wm = np.full((32, 32, 4), 255, np.uint8)
        for delta in (False, True):
            for fps in (30, 60, 120, 144):
                a = fa.FrametimeAnalyser(fps, detect_stutters=True)
                overlay = fa.Overlay(fps, delta, True, "Title", img_wm)
                img = np.zeros((720, 1280, 3), np.uint8)
                overlay.draw(img, a)
                self.feed(a, "uuuddddu" * 100)
                overlay.draw(img, a)
                self.assertGreater(img.sum(), 0)


class TestEndToEnd(TempDirTestCase):
    """Run the full analyser headless on a generated video with known drops"""

    def test_known_frame_drops(self):
        # 300 frames at 60fps: mostly smooth, with single, double and a long repeat
        ids, pid = [], 0
        for i in range(300):
            if i % 10 == 5 or (i % 50 == 20) or 200 <= i < 206:
                pass  # repeat the previous picture
            else:
                pid += 1
            ids.append(pid)
        video = os.path.join(self.temp_dir, "game capture.mp4")
        write_video(video, ids)

        code = fa.main([video, "60", "--no-display", "--detect-stutters", "--delta-plot",
                        "--title", "Test"])
        self.assertEqual(code, 0)

        paths = fa.output_paths(video)
        for path in paths:
            self.assertTrue(os.path.exists(path), path)

        # Per-frame frametimes match the drops written into the video
        with open(paths.frametimes_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        got = [round(float(r["frametime_ms"]) / (1000 / 60)) for r in rows]
        self.assertEqual(got, expected_frametime_counts(ids))
        self.assertEqual([int(r["unique_frame"]) for r in rows], list(range(len(rows))))

        # Summary CSV agrees
        with open(paths.summary_csv, newline="") as f:
            summary = {row[0]: row[1:] for row in csv.reader(f) if row}
        self.assertEqual(int(summary["Total Unique Frames Analyzed"][0]), len(rows))
        self.assertEqual(int(summary["Severe Stutters (>= 4x target frametime)"][0]), 1)
        self.assertIn("Bad", summary)

        # Output video is the full analysed length on the fixed canvas
        cap = cv2.VideoCapture(paths.video)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 299)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 1280)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 720)
        cap.release()

    def test_fps_mismatch_warning(self):
        video = os.path.join(self.temp_dir, "clip30.mp4")
        write_video(video, list(range(40)), fps=30)
        result = subprocess.run([sys.executable, LEGACY_SCRIPT, video, "60", "--no-display"],
                                cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Warning: the video is 30.00 fps", result.stdout)


if __name__ == "__main__":
    unittest.main()
