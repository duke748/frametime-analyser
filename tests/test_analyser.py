"""Tests for frametime_analyser.analyser, fed with frame differences directly"""
import unittest

import numpy as np

from frametime_analyser.analyser import FrametimeAnalyser

FT60 = 1000 / 60


def feed(analyser, pattern, new=20.0, repeat=0.0):
    """pattern: 'u' for a new frame, 'd' for a repeat. Returns reported frametimes."""
    out = []
    for c in pattern:
        out += [f.frametime_ms for f in analyser.process(new if c == "u" else repeat)]
    return out


class TestDuplicateDetection(unittest.TestCase):
    def test_smooth_video(self):
        a = FrametimeAnalyser(60)
        fts = feed(a, "u" * 120)
        self.assertEqual(len(fts), 120)
        self.assertTrue(all(abs(ft - FT60) < 1e-9 for ft in fts))
        self.assertEqual(a.fps, 60)
        self.assertAlmostEqual(a.window.average_fps(), 60)

    def test_repeated_frames_lengthen_frametime(self):
        a = FrametimeAnalyser(60)
        fts = feed(a, "uuduuddu")
        self.assertEqual([round(ft / FT60) for ft in fts], [1, 1, 2, 1, 3])
        self.assertEqual((a.unique_frames, a.frames_processed), (5, 8))

    def test_half_rate_video(self):
        a = FrametimeAnalyser(60)
        feed(a, "ud" * 60)
        self.assertEqual(a.fps, 30)
        # The first new frame is only one frame long, so allow a little slack
        self.assertAlmostEqual(a.window.average_fps(), 30, delta=0.3)

    def test_120fps(self):
        a = FrametimeAnalyser(120)
        fts = feed(a, "u" * 240)
        self.assertAlmostEqual(fts[-1], 1000 / 120)
        self.assertEqual(a.fps, 120)

    def test_compression_noise_below_floor_is_a_repeat(self):
        """Tiny differences (noise on a repeated frame) don't count as new frames"""
        a = FrametimeAnalyser(60, min_diff=0.1)
        feed(a, "u" * 10)
        fts = feed(a, "d" * 20, repeat=0.05)
        self.assertEqual(fts, [])
        self.assertEqual(a.unique_frames, 10)

    def test_peak_decides_when_given(self):
        """With a peak measure, a tiny mean change can still be a new frame"""
        a = FrametimeAnalyser(60)
        for i in range(20):
            a.process(0.05, peak=8.0 if i % 2 == 0 else 0.5)
        self.assertEqual(a.unique_frames, 10)

    def test_floor_can_be_disabled(self):
        a = FrametimeAnalyser(60, min_diff=0)
        feed(a, "d" * 5, repeat=0.05)  # Noise only: every frame looks new
        self.assertEqual(a.unique_frames, 5)


class TestStaticScenes(unittest.TestCase):
    def test_long_pause_is_not_counted(self):
        a = FrametimeAnalyser(60, static_after=2.0)
        feed(a, "u" * 60)
        feed(a, "d" * 130)
        self.assertTrue(a.in_static)
        new = a.process(20.0)
        self.assertTrue(new[0].after_static)
        self.assertFalse(a.in_static)
        feed(a, "u" * 59)
        self.assertEqual(a.static_scenes, 1)
        self.assertAlmostEqual(a.static_ms, 131 * FT60)
        self.assertEqual(len(a.window), 119)
        self.assertAlmostEqual(a.window.average_fps(), 60)
        self.assertEqual(a.counted_frames, 119)

    def test_short_freeze_is_counted(self):
        a = FrametimeAnalyser(60, static_after=2.0, detect_stutters=True)
        feed(a, "u" * 10 + "d" * 59 + "u")  # A one second freeze
        self.assertEqual(a.static_scenes, 0)
        self.assertAlmostEqual(a.window.percentile(100), 60 * FT60)
        self.assertEqual(a.severe_stutters, 1)

    def test_disabled(self):
        a = FrametimeAnalyser(60, static_after=0)
        feed(a, "u" + "d" * 300 + "u")
        self.assertEqual(a.static_scenes, 0)
        self.assertFalse(a.in_static)


class TestStutters(unittest.TestCase):
    def test_stutter_detection(self):
        """2.5x target is moderate, 4x is severe (60fps: 41.7ms / 66.7ms)"""
        a = FrametimeAnalyser(60, detect_stutters=True)
        # 2 frames (33ms): none; 3 frames (50ms): moderate; 4 frames (66.7ms): severe
        feed(a, "uu" + "du" + "ddu" + "dddu" + "u")
        self.assertEqual([s.severity for s in a.stutters], ["moderate", "severe"])
        self.assertEqual((a.moderate_stutters, a.severe_stutters), (1, 1))
        self.assertEqual([s.unique_index for s in a.stutters], [3, 4])
        self.assertEqual([s.video_frame for s in a.stutters], [7, 11])
        self.assertEqual([s.graph_index for s in a.stutters], [3, 4])

    def test_threshold_is_exact_late_in_a_video(self):
        """A 4-frame stall is severe (4x) however far into the video it happens,
        with no floating-point drift putting it just under the threshold"""
        a = FrametimeAnalyser(60, detect_stutters=True)
        for start in range(5000):
            a.process(20.0)
            if start % 997 == 0:
                feed(a, "dddu")
        self.assertEqual(a.moderate_stutters, 0)
        self.assertEqual(a.severe_stutters, 6)
        self.assertTrue(all(s.frametime_ms == 4 * FT60 for s in a.stutters))

    def test_stutters_not_recorded_when_disabled(self):
        a = FrametimeAnalyser(60)
        feed(a, "u" + "d" * 10 + "u")
        self.assertEqual(a.stutters, [])


class TestTearing(unittest.TestCase):
    def rows(self, start, n=72):
        r = np.zeros(n)
        r[start:] = 40.0
        return r

    def test_torn_frametimes(self):
        """A new frame starting halfway down the screen is half a refresh later"""
        a = FrametimeAnalyser(60, detect_tears=True)
        a.process(40.0, self.rows(0))
        new = a.process(20.0, self.rows(36))
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0].tear, 0.5)
        self.assertAlmostEqual(new[0].frametime_ms, 1.5 * FT60, places=2)
        self.assertEqual(a.torn_frames, 1)
        self.assertAlmostEqual(a.summary()["torn_pct"], 50)

    def test_tearing_probe_warns_without_tear_detection(self):
        a = FrametimeAnalyser(60)
        for i in range(200):
            tear = 10 + (i * 17) % 50
            a.process(20.0, self.rows(tear))
            r = np.zeros(72)
            r[:tear] = 40.0
            a.process(20.0, r)
        self.assertTrue(a.looks_torn())
        self.assertEqual(a.torn_frames, 0)  # Not counted as torn: tear detection is off

    def test_no_warning_for_static_sky(self):
        """V-Sync on, with a still sky above a moving foreground: each new frame
        changes only the bottom, which looks like a tear but never catches up"""
        a = FrametimeAnalyser(60)
        for i in range(200):
            a.process(20.0, self.rows(30) if i % 2 else np.zeros(72))
        self.assertFalse(a.looks_torn())

    def test_no_warning_for_vsync_footage(self):
        a = FrametimeAnalyser(60)
        for i in range(200):
            a.process(20.0, self.rows(0) if i % 3 else np.zeros(72))
        self.assertFalse(a.looks_torn())

    def test_without_rows_falls_back(self):
        a = FrametimeAnalyser(60, detect_tears=True)
        self.assertEqual(len(a.process(20.0)), 1)


class TestGraphsAndSummary(unittest.TestCase):
    def test_frame_delta(self):
        a = FrametimeAnalyser(60)
        feed(a, "uudu")
        self.assertEqual(list(a.delta_graph)[-3:], [0, 0, FT60])
        self.assertEqual(a.graph_samples, 3)

    def test_summary(self):
        a = FrametimeAnalyser(60)
        feed(a, ("u" * 98 + "du") * 2)
        s = a.summary()
        self.assertEqual(s["unique_frames"], 198)
        self.assertAlmostEqual(s["duration_s"], 200 / 60)
        self.assertAlmostEqual(s["p99_frametime"], 2 * FT60)
        self.assertAlmostEqual(s["low_1_fps"], 30)
        self.assertEqual((s["torn_frames"], s["static_scenes"]), (0, 0))


if __name__ == "__main__":
    unittest.main()
