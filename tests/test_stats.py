"""Unit tests for frametime_analyser.stats"""
import math
import unittest

from frametime_analyser.stats import (
    BUCKET_MULTIPLIERS,
    GREEN,
    ORANGE,
    RED,
    YELLOW,
    FpsCounter,
    FrametimeWindow,
    categorize_frametime,
    count_frames_per_bucket,
    format_bucket_range,
    fps_color,
    get_histogram_buckets,
)


class TestHistogramBuckets(unittest.TestCase):
    """The histogram buckets used by the overlay, CSV export and report"""

    def assert_bucket_edges(self, target_fps, expected):
        buckets = get_histogram_buckets(target_fps)
        self.assertEqual([(lo, hi, label) for lo, hi, label, _ in buckets], expected)

    def test_60fps_histogram_buckets(self):
        self.assert_bucket_edges(60, [(0, 18, "Perfect"), (18, 25, "Good"), (25, 35, "Fair"),
                                      (35, 50, "Stutter"), (50, math.inf, "Bad")])

    def test_120fps_histogram_buckets(self):
        self.assert_bucket_edges(120, [(0, 9, "Perfect"), (9, 12, "Good"), (12, 17, "Fair"),
                                       (17, 25, "Stutter"), (25, math.inf, "Bad")])

    def test_buckets_are_contiguous(self):
        """Each bucket starts where the previous one ends, so no frametime is skipped"""
        for target_fps in (30, 60, 90, 120, 144, 240):
            buckets = get_histogram_buckets(target_fps)
            self.assertEqual(buckets[0][0], 0)
            self.assertTrue(math.isinf(buckets[-1][1]))
            for prev, nxt in zip(buckets, buckets[1:]):
                self.assertEqual(prev[1], nxt[0])
                self.assertLess(prev[0], prev[1])

    def test_ideal_frametime_is_perfect(self):
        for target_fps in (30, 60, 90, 120, 144, 240):
            buckets = get_histogram_buckets(target_fps)
            i = categorize_frametime(1000.0 / target_fps, buckets)
            self.assertEqual(buckets[i][2], "Perfect", target_fps)

    def test_performance_category_thresholds(self):
        """Frametimes on and around the bucket edges get the right category"""
        buckets = get_histogram_buckets(60)
        cases = {0: "Perfect", 17.99: "Perfect", 18: "Good", 24.99: "Good", 25: "Fair",
                 35: "Stutter", 49.99: "Stutter", 50: "Bad", 500: "Bad"}
        for ft, expected in cases.items():
            self.assertEqual(buckets[categorize_frametime(ft, buckets)][2], expected, ft)

    def test_long_freezes_are_counted(self):
        """The last bucket is open-ended, so multi-second freezes are still counted"""
        buckets = get_histogram_buckets(60)
        self.assertEqual(buckets[categorize_frametime(5000, buckets)][2], "Bad")
        self.assertIsNone(categorize_frametime(-1, buckets))

    def test_count_frames_per_bucket(self):
        buckets = get_histogram_buckets(60)
        frametimes = [16.67, 16.67, 20.0, 33.33, 40.0, 66.67, 1500.0]
        self.assertEqual(count_frames_per_bucket(frametimes, buckets), [2, 1, 1, 1, 2])

    def test_derived_buckets_match_60fps_table(self):
        edges = [round(1000.0 / 60 * m, 1) for m in BUCKET_MULTIPLIERS]
        self.assertEqual(edges, [18, 25, 35, 50])

    def test_invalid_target_fps(self):
        with self.assertRaises(ValueError):
            get_histogram_buckets(0)

    def test_format_bucket_range(self):
        self.assertEqual(format_bucket_range(18, 25), "18-25")
        self.assertEqual(format_bucket_range(4.5, 6.2), "4.5-6.2")
        self.assertEqual(format_bucket_range(50, math.inf), "50+")


class TestFpsColor(unittest.TestCase):
    def test_thresholds_scale_with_target(self):
        cases = [(60, 60, GREEN), (58, 60, GREEN), (55, 60, YELLOW), (45, 60, ORANGE),
                 (30, 60, RED), (120, 120, GREEN), (100, 120, YELLOW), (90, 120, ORANGE),
                 (60, 120, RED), (144, 144, GREEN)]
        for avg, target, expected in cases:
            self.assertEqual(fps_color(avg, target), expected, (avg, target))


class TestFpsCounter(unittest.TestCase):
    """The rolling one-second FPS counter"""

    def test_window_spans_one_second(self):
        for target_fps in (60, 120):
            self.assertEqual(len(FpsCounter(target_fps).window), target_fps)

    def test_all_unique_frames_reach_target_fps(self):
        for target_fps in (60, 120):
            counter = FpsCounter(target_fps)
            for _ in range(target_fps):
                fps = counter.record(True)
            self.assertEqual(fps, target_fps)

    def test_repeated_frames_lower_fps(self):
        counter = FpsCounter(60)
        for i in range(60):
            fps = counter.record(i % 2 == 0)
        self.assertEqual(fps, 30)

    def test_old_frames_leave_window(self):
        counter = FpsCounter(60)
        for _ in range(60):
            counter.record(True)
        for _ in range(60):
            fps = counter.record(False)
        self.assertEqual(fps, 0)

    def test_several_new_frames_in_one_refresh(self):
        """With tearing, one captured frame can contain two new frames"""
        counter = FpsCounter(60)
        for i in range(60):
            fps = counter.record(2 if i % 2 else 1)
        self.assertEqual(fps, 90)


class TestFrametimeWindow(unittest.TestCase):
    """The rolling frametime statistics window"""

    def test_empty_window(self):
        w = FrametimeWindow(1000)
        self.assertEqual((len(w), w.mean(), w.average_fps(), w.percentile(99), w.low_fps(1)),
                         (0, 0, 0, 0, 0))
        self.assertEqual(w.cdf([10, 20]), [0, 0])

    def test_mean_and_average_fps(self):
        w = FrametimeWindow(60_000)
        for ft in [16.67, 16.67, 16.67, 33.34, 16.67]:
            w.add(ft)
        self.assertAlmostEqual(w.mean(), 20.004, places=3)
        self.assertAlmostEqual(w.average_fps(), 1000 / 20.004, places=2)

    def test_window_keeps_only_recent_duration(self):
        w = FrametimeWindow(100)
        for _ in range(10):
            w.add(10.0)
        w.add(50.0)
        self.assertEqual(len(w), 6)  # 5 x 10ms + 50ms
        self.assertAlmostEqual(w.total_ms, 100)
        self.assertEqual(w.bucket_counts([(0, 20, "a", None), (20, math.inf, "b", None)]), [5, 1])

    def test_single_long_frame_is_kept(self):
        w = FrametimeWindow(100)
        w.add(500.0)
        self.assertEqual(len(w), 1)

    def test_thirty_minute_window_at_60fps(self):
        w = FrametimeWindow(30 * 60 * 1000)
        for _ in range(60 * 60 * 31):
            w.add(1000.0 / 60)
        self.assertAlmostEqual(len(w), 60 * 60 * 30, delta=1)

    def test_percentiles_and_lows(self):
        w = FrametimeWindow(10**9)
        for _ in range(98):
            w.add(10.0)
        w.add(20.0)
        w.add(50.0)
        self.assertEqual(w.percentile(50), 10.0)
        self.assertEqual(w.percentile(99), 20.0)
        self.assertEqual(w.percentile(100), 50.0)
        self.assertAlmostEqual(w.low_fps(1), 50.0)
        self.assertAlmostEqual(w.low_fps(0.1), 20.0)

    def test_percentile_cache_updates(self):
        w = FrametimeWindow(10**9)
        w.add(10.0)
        self.assertEqual(w.percentile(100), 10.0)
        w.add(30.0)
        self.assertEqual(w.percentile(100), 30.0)

    def test_cdf(self):
        w = FrametimeWindow(10**9)
        for ft in [10, 10, 20, 40]:
            w.add(ft)
        self.assertEqual(w.cdf([0, 10, 15, 20, 39, 40, 100]), [0, 50, 50, 75, 75, 100, 100])

    def test_bucket_counts_match_count_frames_per_bucket(self):
        buckets = get_histogram_buckets(60)
        frametimes = [16.67, 16.67, 20.0, 33.33, 40.0, 66.67, 1500.0]
        w = FrametimeWindow(10**9)
        for ft in frametimes:
            w.add(ft)
        self.assertEqual(w.bucket_counts(buckets), count_frames_per_bucket(frametimes, buckets))


if __name__ == "__main__":
    unittest.main()
