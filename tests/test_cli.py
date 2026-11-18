"""Command line and end-to-end tests: real videos through the whole analyser"""
import csv
import io
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout

import cv2
import numpy as np

from frametime_analyser import cli
from tests.helpers import (
    HAS_FFMPEG,
    TempDirTestCase,
    dropped_frame_ids,
    expected_frametime_counts,
    needs_ffmpeg,
    tearing_schedule,
    write_torn_video,
    write_video,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_SCRIPT = os.path.join(REPO_ROOT, "Frametime Analyser.py")


def run_main(*args):
    out = io.StringIO()
    with redirect_stdout(out):
        code = cli.main([str(a) for a in args])
    return code, out.getvalue()


def read_frametimes(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_summary(path):
    with open(path, newline="") as f:
        return {row[0]: row[1:] for row in csv.reader(f) if row}


class TestArgumentParsing(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.video = self.path("clip.mp4")
        open(self.video, "w").close()

    def parse(self, *args):
        return cli.parse_args(list(args))

    def assert_rejected(self, *args):
        with self.assertRaises(SystemExit) as ctx, redirect_stdout(io.StringIO()):
            import contextlib
            with contextlib.redirect_stderr(io.StringIO()):
                self.parse(*args)
        self.assertEqual(ctx.exception.code, 2)

    def test_defaults(self):
        a = self.parse(self.video)
        self.assertIsNone(a.fps)  # Read from the video later
        self.assertFalse(a.delta_plot or a.detect_stutters or a.no_display or a.no_video)
        self.assertEqual((a.encoder, a.decoder, a.ignore, a.min_diff, a.static_after),
                         ("auto", "auto", [], 0.1, 2.0))

    def test_any_target_fps(self):
        for fps in (30, 60, 120, 144, 240):
            self.assertEqual(self.parse(self.video, str(fps)).fps, fps)
        self.assertEqual(self.parse(self.video, "--fps", "90").fps, 90)

    def test_invalid_fps(self):
        for fps in ("0", "-5", "1001", "fast", "59.94"):
            self.assert_rejected(self.video, fps)
        self.assert_rejected(self.video, "60", "--fps", "120")

    def test_legacy_single_dash_flags(self):
        wm = self.path("wm.png")
        cv2.imwrite(wm, np.zeros((8, 8, 3), np.uint8))
        a = self.parse(self.video, "120", "-delta-plot", "-detect-stutters", "-watermark", wm, "-title", "My Game")
        self.assertTrue(a.delta_plot and a.detect_stutters)
        self.assertEqual((a.fps, a.watermark, a.title), (120, wm, "My Game"))

    def test_fps_after_options(self):
        a = self.parse(self.video, "-delta-plot", "120")
        self.assertEqual(a.fps, 120)

    def test_regions(self):
        a = self.parse(self.video, "--roi", "0,0,1920,1000", "--ignore", "0,0,10,10", "--ignore", "5,5,20,20")
        self.assertEqual(len(a.ignore), 2)
        self.assertEqual(a.roi.w, 1920)
        self.assert_rejected(self.video, "--ignore", "1,2,3")

    def test_capture(self):
        a = self.parse("--capture", "0", "120")
        self.assertEqual((a.capture, a.fps, a.video), (0, 120, None))
        a = self.parse("--capture", "1", "--fps", "60", "--capture-size", "1920x1080")
        self.assertEqual(a.capture_size, (1920, 1080))
        self.assert_rejected("--capture", "0", self.video)
        self.assert_rejected("--capture", "0", "--capture-size", "big")
        self.assert_rejected("--capture", "0", "--hw-decode")

    def test_compare(self):
        a = self.parse(self.video, "--compare", self.video, "--labels", "PC", "Console")
        self.assertEqual(a.labels, ["PC", "Console"])
        self.assert_rejected(self.video, "--compare", self.path("missing.mp4"))
        self.assert_rejected(self.video, "--labels", "a", "b")
        self.assert_rejected("--capture", "0", "--compare", self.video)

    def test_missing_files(self):
        self.assert_rejected(self.path("missing.mp4"))
        self.assert_rejected()
        self.assert_rejected(self.video, "-watermark", self.path("nope.png"))
        self.assert_rejected(self.video, "--output-dir", self.path("nope"))

    def test_missing_values(self):
        self.assert_rejected(self.video, "-watermark")
        self.assert_rejected(self.video, "-title")
        self.assert_rejected(self.video, "--bogus")
        self.assert_rejected(self.video, "--encoder", "divx")

    def test_output_paths(self):
        p = cli.output_paths("C:/Videos/gameplay.mp4")
        self.assertEqual(p.video, os.path.join("C:/Videos", "gameplay_analyzed.mp4"))
        self.assertEqual(p.summary_csv, os.path.join("C:/Videos", "gameplay_histogram.csv"))
        self.assertEqual(p.frametimes_csv, os.path.join("C:/Videos", "gameplay_frametimes.csv"))
        self.assertEqual(p.report, os.path.join("C:/Videos", "gameplay_report.html"))
        self.assertEqual(cli.output_paths("C:/Videos/g.mp4", "D:/out").video, os.path.join("D:/out", "g_analyzed.mp4"))
        self.assertEqual(os.path.basename(cli.compare_paths("a/x.mp4", "b/y.mkv")), "x_vs_y_compare.mp4")


class TestLegacyEntryPoint(TempDirTestCase):
    """'Frametime Analyser.py' and `python -m frametime_analyser` still work"""

    def run_cmd(self, *cmd):
        return subprocess.run([sys.executable, *cmd], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)

    def test_help(self):
        result = self.run_cmd(LEGACY_SCRIPT, "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--no-video", result.stdout)

    def test_version(self):
        result = self.run_cmd("-m", "frametime_analyser", "--version")
        self.assertEqual(result.returncode, 0)
        self.assertIn("2.", result.stdout)

    def test_rejects_bad_arguments(self):
        result = self.run_cmd(LEGACY_SCRIPT, self.path("missing.mp4"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("not found", result.stderr)

    def test_headless_run(self):
        video = self.path("clip.mp4")
        write_video(video, list(range(40)))
        result = self.run_cmd(LEGACY_SCRIPT, video, "--no-display", "--no-video")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.path.exists(cli.output_paths(video).summary_csv))


class TestEndToEnd(TempDirTestCase):
    """Run the analyser headless on generated videos with known timing"""

    def check_drops(self, *extra):
        ids = dropped_frame_ids()
        video = self.path("game capture.mp4")
        write_video(video, ids)
        code, out = run_main(video, "60", "--no-display", "--detect-stutters", "--delta-plot",
                             "--title", "Test", *extra)
        self.assertEqual(code, 0, out)
        paths = cli.output_paths(video)
        rows = read_frametimes(paths.frametimes_csv)
        got = [round(float(r["frametime_ms"]) / (1000 / 60)) for r in rows]
        self.assertEqual(got, expected_frametime_counts(ids))
        self.assertEqual([int(r["unique_frame"]) for r in rows], list(range(len(rows))))
        summary = read_summary(paths.summary_csv)
        self.assertEqual(int(summary["Total Unique Frames Analyzed"][0]), len(rows))
        self.assertEqual(int(summary["Severe Stutters (>= 4x target frametime)"][0]), 1)
        return paths, out

    def test_known_frame_drops_opencv(self):
        paths, out = self.check_drops("--decoder", "opencv", "--encoder", "opencv")
        cap = cv2.VideoCapture(paths.video)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 299)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 1280)
        cap.release()
        self.assertIn("Decoder: opencv", out)

    @needs_ffmpeg
    def test_known_frame_drops_ffmpeg(self):
        paths, out = self.check_drops("--decoder", "ffmpeg", "--encoder", "libx264")
        self.assertIn("Decoder: ffmpeg", out)
        self.assertIn("libx264", out)
        self.assertTrue(os.path.exists(paths.video))

    def test_known_frame_drops_analysis_only(self):
        paths, _ = self.check_drops("--no-video")
        self.assertFalse(os.path.exists(paths.video))

    def test_report(self):
        paths, _ = self.check_drops("--no-video")
        with open(paths.report, encoding="utf-8") as f:
            page = f.read()
        self.assertIn("<svg", page)
        self.assertIn("Average FPS", page)
        self.assertIn("Worst stutters", page)
        self.assertIn("game capture.mp4", page)

    def test_no_report(self):
        paths, _ = self.check_drops("--no-video", "--no-report")
        self.assertFalse(os.path.exists(paths.report))

    def test_output_dir(self):
        out_dir = self.path("out")
        os.mkdir(out_dir)
        video = self.path("clip.mp4")
        write_video(video, list(range(30)))
        code, _ = run_main(video, "--no-display", "--no-video", "--output-dir", out_dir)
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "clip_histogram.csv")))

    def test_target_fps_read_from_video(self):
        video = self.path("clip30.mp4")
        write_video(video, list(range(40)), fps=30)
        code, out = run_main(video, "--no-display", "--no-video")
        self.assertEqual(code, 0)
        self.assertIn("Target FPS: 30 (read from the video)", out)
        self.assertEqual(read_summary(cli.output_paths(video).summary_csv)["Target FPS"], ["30"])

    def test_fps_mismatch_warning(self):
        video = self.path("clip30.mp4")
        write_video(video, list(range(40)), fps=30)
        code, out = run_main(video, "60", "--no-display", "--no-video")
        self.assertEqual(code, 0)
        self.assertIn("Warning: the video is 30.00 fps", out)

    def test_ignore_webcam_region(self):
        """A webcam overlay changing every frame would hide the game's 30fps"""
        ids = [i // 2 for i in range(120)]  # Game at 30fps on a 60fps capture

        def webcam(i, frame):
            frame[0:90, 200:320] = (i * 97) % 256  # Top-right facecam changes every frame

        video = self.path("stream.mp4")
        write_video(video, ids, overlay=webcam)
        code, _ = run_main(video, "--no-display", "--no-video")
        self.assertEqual(code, 0)
        fps_without = float(read_summary(cli.output_paths(video).summary_csv)["Average FPS"][0])
        code, _ = run_main(video, "--no-display", "--no-video", "--ignore", "195,0,125,95")
        self.assertEqual(code, 0)
        fps_with = float(read_summary(cli.output_paths(video).summary_csv)["Average FPS"][0])
        self.assertAlmostEqual(fps_without, 60, delta=1)
        self.assertAlmostEqual(fps_with, 30, delta=1)

    def test_static_scene_excluded(self):
        ids = list(range(60)) + [60] * 180 + list(range(61, 121))  # 3s pause in the middle
        video = self.path("pause.mp4")
        write_video(video, ids)
        code, _ = run_main(video, "--no-display", "--no-video")
        self.assertEqual(code, 0)
        summary = read_summary(cli.output_paths(video).summary_csv)
        self.assertEqual(summary["Static Scenes Excluded"], ["1"])
        self.assertAlmostEqual(float(summary["Average FPS"][0]), 60, delta=0.5)
        rows = read_frametimes(cli.output_paths(video).frametimes_csv)
        self.assertEqual(sum(int(r["after_static"]) for r in rows), 1)

    def test_torn_video(self):
        """50fps with V-Sync off on a 60Hz capture: every new frame tears"""
        flips = tearing_schedule(game_fps=50, frames=120)
        video = self.path("torn.mp4")
        write_torn_video(video, flips, 120)
        code, out = run_main(video, "60", "--no-display", "--no-video", "--detect-tears", "--decoder", "opencv")
        self.assertEqual(code, 0, out)
        rows = read_frametimes(cli.output_paths(video).frametimes_csv)
        times = [float(r["time_s"]) * 60 for r in rows]  # In refreshes
        expected = [t for t in flips if 2 <= t < 119]
        matched = sum(1 for t in expected if any(abs(t - g) < 0.04 for g in times))
        self.assertGreaterEqual(matched / len(expected), 0.95)
        self.assertLessEqual(len(rows), len([t for t in flips if t < 120]) + 1)
        fts = sorted(float(r["frametime_ms"]) for r in rows[3:])
        self.assertAlmostEqual(fts[len(fts) // 2], 20.0, delta=0.5)  # Median 1000/50
        summary = read_summary(cli.output_paths(video).summary_csv)
        self.assertGreater(float(summary["Torn Frame Percentage"][0].rstrip("%")), 90)

    def test_vsync_video_has_no_tears(self):
        ids = dropped_frame_ids(120)
        video = self.path("vsync.mp4")
        write_video(video, ids)
        code, _ = run_main(video, "60", "--no-display", "--no-video", "--detect-tears")
        self.assertEqual(code, 0)
        rows = read_frametimes(cli.output_paths(video).frametimes_csv)
        self.assertEqual([round(float(r["frametime_ms"]) / (1000 / 60)) for r in rows],
                         expected_frametime_counts(ids))
        self.assertTrue(all(r["tear_position"] == "" for r in rows))

    def test_bad_capture_device(self):
        code, out = run_main("--capture", "97", "--no-display", "--no-video")
        self.assertEqual(code, 1)

    def test_compare(self):
        a, b = self.path("pc.mp4"), self.path("console.mp4")
        ids_a = list(range(150))
        ids_b = [i // 2 for i in range(140)]  # 30fps, and shorter
        write_video(a, ids_a)
        write_video(b, ids_b)
        code, out = run_main(a, "--compare", b, "--labels", "PC", "Console", "--no-display",
                             "--detect-stutters", "--encoder", "opencv")
        self.assertEqual(code, 0, out)
        video = cli.compare_paths(a, b)
        cap = cv2.VideoCapture(video)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 139)  # Shorter video
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 1280)
        cap.release()
        fps_a = float(read_summary(cli.output_paths(a).summary_csv)["Average FPS"][0])
        fps_b = float(read_summary(cli.output_paths(b).summary_csv)["Average FPS"][0])
        self.assertAlmostEqual(fps_a, 60, delta=0.5)
        self.assertAlmostEqual(fps_b, 30, delta=0.5)

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg not available")
    def test_compare_analysis_only(self):
        a, b = self.path("a.mp4"), self.path("b.mp4")
        write_video(a, list(range(60)))
        write_video(b, list(range(60)))
        code, _ = run_main(a, "--compare", b, "--no-display", "--no-video")
        self.assertEqual(code, 0)
        self.assertFalse(os.path.exists(cli.compare_paths(a, b)))
        self.assertTrue(os.path.exists(cli.output_paths(b).report))


if __name__ == "__main__":
    unittest.main()
