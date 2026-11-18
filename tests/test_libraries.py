"""Unit tests for libraries module"""
import math
import shutil
import unittest
import sys
import os
import tempfile
import numpy as np
import cv2

# Make the repo root importable regardless of the working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libraries import (
    ResizeWithAspectRatio,
    FileVideoStream,
    get_histogram_buckets,
    categorize_frametime,
    count_frames_per_bucket,
    FpsCounter,
    FrametimeWindow,
    BUCKET_MULTIPLIERS,
    fit_to_canvas,
    format_bucket_range,
    fps_color,
    frame_difference,
    GREEN, YELLOW, ORANGE, RED,
)


class TestResizeWithAspectRatio(unittest.TestCase):
    """Test cases for ResizeWithAspectRatio function"""
    
    def setUp(self):
        """Create a test image"""
        # Create a 1920x1080 test image
        self.test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
    def test_resize_with_width(self):
        """Test resizing by width while maintaining aspect ratio"""
        result = ResizeWithAspectRatio(self.test_image, width=1280)
        self.assertEqual(result.shape[1], 1280)  # Width should be 1280
        self.assertEqual(result.shape[0], 720)   # Height should scale proportionally
        
    def test_resize_with_height(self):
        """Test resizing by height while maintaining aspect ratio"""
        result = ResizeWithAspectRatio(self.test_image, height=720)
        self.assertEqual(result.shape[0], 720)   # Height should be 720
        self.assertEqual(result.shape[1], 1280)  # Width should scale proportionally
        
    def test_no_resize_parameters(self):
        """Test that image is returned unchanged when no parameters are provided"""
        result = ResizeWithAspectRatio(self.test_image)
        np.testing.assert_array_equal(result, self.test_image)
        
    def test_resize_small_image(self):
        """Test resizing a small image"""
        small_image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = ResizeWithAspectRatio(small_image, width=400)
        self.assertEqual(result.shape[1], 400)
        self.assertEqual(result.shape[0], 200)  # Should double in height too
        
    def test_resize_maintains_channels(self):
        """Test that number of color channels is preserved"""
        result = ResizeWithAspectRatio(self.test_image, width=640)
        self.assertEqual(len(result.shape), 3)
        self.assertEqual(result.shape[2], 3)  # Should still have 3 channels


class TestFileVideoStream(unittest.TestCase):
    """Test cases for FileVideoStream class"""
    
    def test_initialization_with_invalid_path(self):
        """Test initialization with non-existent video file"""
        fvs = FileVideoStream("nonexistent_file.mp4")
        # Should initialize without error, but stream should not be opened
        self.assertIsNotNone(fvs.stream)
        self.assertFalse(fvs.stopped)
        
    def test_queue_initialization(self):
        """Test that queue is initialized with correct size"""
        fvs = FileVideoStream("test.mp4", queue_size=100)
        self.assertEqual(fvs.Q.maxsize, 100)
        
    def test_default_queue_size(self):
        """Test that default queue size is set correctly"""
        fvs = FileVideoStream("test.mp4")
        self.assertEqual(fvs.Q.maxsize, 196)
        
    def test_thread_daemon_flag(self):
        """Test that thread is created as daemon"""
        fvs = FileVideoStream("test.mp4")
        self.assertTrue(fvs.thread.daemon)
        
    def test_initial_state(self):
        """Test initial state of FileVideoStream"""
        fvs = FileVideoStream("test.mp4")
        self.assertFalse(fvs.stopped)
        self.assertIsNone(fvs.transform)


class TestFrametimeCalculations(unittest.TestCase):
    """Test cases for frametime calculation logic"""
    
    def test_60fps_frametime(self):
        """Test frametime calculation for 60fps"""
        target_fps = 60
        target_frametime = 1000.0 / target_fps
        self.assertAlmostEqual(target_frametime, 16.666666666666668, places=5)
        
    def test_120fps_frametime(self):
        """Test frametime calculation for 120fps"""
        target_fps = 120
        target_frametime = 1000.0 / target_fps
        self.assertAlmostEqual(target_frametime, 8.333333333333334, places=5)
        
    def test_stutter_threshold_60fps(self):
        """Test stutter detection threshold for 60fps"""
        target_fps = 60
        target_frametime = 1000.0 / target_fps
        stutter_threshold = target_frametime * 2.5
        self.assertAlmostEqual(stutter_threshold, 41.666666666666664, places=5)
        
    def test_severe_stutter_threshold_120fps(self):
        """Test severe stutter detection threshold for 120fps"""
        target_fps = 120
        target_frametime = 1000.0 / target_fps
        severe_stutter_threshold = target_frametime * 4.0
        self.assertAlmostEqual(severe_stutter_threshold, 33.333333333333336, places=5)


class TestHistogramBuckets(unittest.TestCase):
    """Test cases for the histogram buckets used by the display and CSV export"""

    def assert_bucket_edges(self, target_fps, expected):
        buckets = get_histogram_buckets(target_fps)
        self.assertEqual([(lo, hi, label) for lo, hi, label, _ in buckets], expected)

    def test_60fps_histogram_buckets(self):
        """Test histogram bucket ranges for 60fps"""
        self.assert_bucket_edges(60, [
            (0, 18, "Perfect"),
            (18, 25, "Good"),
            (25, 35, "Fair"),
            (35, 50, "Stutter"),
            (50, math.inf, "Bad"),
        ])

    def test_120fps_histogram_buckets(self):
        """Test histogram bucket ranges for 120fps"""
        self.assert_bucket_edges(120, [
            (0, 9, "Perfect"),
            (9, 12, "Good"),
            (12, 17, "Fair"),
            (17, 25, "Stutter"),
            (25, math.inf, "Bad"),
        ])

    def test_buckets_are_contiguous(self):
        """Each bucket starts where the previous one ends, so no frametime is skipped"""
        for target_fps in (60, 120):
            buckets = get_histogram_buckets(target_fps)
            for prev, nxt in zip(buckets, buckets[1:]):
                self.assertEqual(prev[1], nxt[0])

    def test_ideal_frametime_is_perfect(self):
        """The target frametime falls in the Perfect bucket"""
        for target_fps in (60, 120):
            buckets = get_histogram_buckets(target_fps)
            i = categorize_frametime(1000.0 / target_fps, buckets)
            self.assertEqual(buckets[i][2], "Perfect")

    def test_performance_category_thresholds(self):
        """Frametimes on and around the bucket edges get the right category"""
        buckets = get_histogram_buckets(60)
        cases = {
            0: "Perfect", 17.99: "Perfect", 18: "Good", 24.99: "Good",
            25: "Fair", 35: "Stutter", 49.99: "Stutter", 50: "Bad", 500: "Bad",
        }
        for ft, expected in cases.items():
            self.assertEqual(buckets[categorize_frametime(ft, buckets)][2], expected, ft)

    def test_long_freezes_are_counted(self):
        """The last bucket is open-ended, so multi-second freezes are still counted"""
        buckets = get_histogram_buckets(60)
        self.assertEqual(buckets[categorize_frametime(999, buckets)][2], "Bad")
        self.assertEqual(buckets[categorize_frametime(5000, buckets)][2], "Bad")
        self.assertIsNone(categorize_frametime(-1, buckets))

    def test_count_frames_per_bucket(self):
        """Frametimes are counted into the right buckets, including a 1.5s freeze"""
        buckets = get_histogram_buckets(60)
        frametimes = [16.67, 16.67, 20.0, 33.33, 40.0, 66.67, 1500.0]
        self.assertEqual(count_frames_per_bucket(frametimes, buckets), [2, 1, 1, 1, 2])

    def test_other_target_fps_buckets(self):
        """Other target FPS values get buckets scaled from the target frametime"""
        for target_fps in (30, 50, 90, 144, 240):
            buckets = get_histogram_buckets(target_fps)
            self.assertEqual(len(buckets), 5)
            self.assertEqual(buckets[0][0], 0)
            self.assertTrue(math.isinf(buckets[-1][1]))
            for prev, nxt in zip(buckets, buckets[1:]):
                self.assertEqual(prev[1], nxt[0])
                self.assertLess(prev[0], prev[1])
            i = categorize_frametime(1000.0 / target_fps, buckets)
            self.assertEqual(buckets[i][2], "Perfect", target_fps)

    def test_derived_buckets_match_60fps_table(self):
        """The scaling rule reproduces the hand-written 60fps edges"""
        edges = [round(1000.0 / 60 * m, 1) for m in BUCKET_MULTIPLIERS]
        self.assertEqual(edges, [18, 25, 35, 50])

    def test_invalid_target_fps(self):
        """A target FPS of zero or less has no buckets"""
        with self.assertRaises(ValueError):
            get_histogram_buckets(0)

    def test_format_bucket_range(self):
        """Bucket ranges are formatted for the overlay and CSV"""
        self.assertEqual(format_bucket_range(18, 25), "18-25")
        self.assertEqual(format_bucket_range(4.5, 6.2), "4.5-6.2")
        self.assertEqual(format_bucket_range(50, math.inf), "50+")


class TestFpsColor(unittest.TestCase):
    """Average FPS colour thresholds scale with the target FPS"""

    def test_thresholds(self):
        cases = [(60, 60, GREEN), (58, 60, GREEN), (55, 60, YELLOW), (45, 60, ORANGE),
                 (30, 60, RED), (120, 120, GREEN), (100, 120, YELLOW), (90, 120, ORANGE),
                 (60, 120, RED), (144, 144, GREEN)]
        for avg, target, expected in cases:
            self.assertEqual(fps_color(avg, target), expected, (avg, target))


class TestFpsCounter(unittest.TestCase):
    """Test cases for the rolling one-second FPS counter"""

    def test_window_spans_one_second(self):
        """The window holds one second of frames at the target FPS"""
        for target_fps in (60, 120):
            self.assertEqual(len(FpsCounter(target_fps).window), target_fps)

    def test_all_unique_frames_reach_target_fps(self):
        """A smooth video reports the full target FPS, including 120fps"""
        for target_fps in (60, 120):
            counter = FpsCounter(target_fps)
            for _ in range(target_fps):
                fps = counter.record(unique=True)
            self.assertEqual(fps, target_fps)

    def test_repeated_frames_lower_fps(self):
        """Repeated frames count as dropped frames"""
        counter = FpsCounter(60)
        for i in range(60):
            fps = counter.record(unique=(i % 2 == 0))
        self.assertEqual(fps, 30)

    def test_old_frames_leave_window(self):
        """Frames older than one second no longer count"""
        counter = FpsCounter(60)
        for _ in range(60):
            counter.record(unique=True)
        for _ in range(60):
            fps = counter.record(unique=False)
        self.assertEqual(fps, 0)


class TestImageProcessing(unittest.TestCase):
    """Test cases for image processing functions"""
    
    def test_grayscale_conversion(self):
        """Test BGR to grayscale conversion"""
        # Create a colored test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :, 0] = 255  # Blue channel
        
        grayscale = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        self.assertEqual(len(grayscale.shape), 2)  # Should be 2D
        self.assertEqual(grayscale.shape, (100, 100))
        
    def test_frame_difference_calculation(self):
        """Test frame difference calculation"""
        # Create two similar frames with slight differences
        frame1 = np.ones((100, 100), dtype=np.uint8) * 100
        frame2 = np.ones((100, 100), dtype=np.uint8) * 110
        
        diff = cv2.absdiff(frame1, frame2)
        expected_diff = 10
        
        self.assertEqual(diff.mean(), expected_diff)
        
    def test_identical_frames_difference(self):
        """Test that identical frames have zero difference"""
        frame = np.ones((100, 100), dtype=np.uint8) * 128
        
        diff = cv2.absdiff(frame, frame)

        self.assertEqual(diff.mean(), 0)


class TestFrametimeWindow(unittest.TestCase):
    """Test cases for the rolling frametime statistics window"""

    def test_empty_window(self):
        w = FrametimeWindow(1000)
        self.assertEqual(len(w), 0)
        self.assertEqual(w.mean(), 0)
        self.assertEqual(w.average_fps(), 0)
        self.assertEqual(w.percentile(99), 0)
        self.assertEqual(w.low_fps(1), 0)
        self.assertEqual(w.cdf([10, 20]), [0, 0])

    def test_mean_and_average_fps(self):
        w = FrametimeWindow(60_000)
        for ft in [16.67, 16.67, 16.67, 33.34, 16.67]:
            w.add(ft)
        self.assertAlmostEqual(w.mean(), 20.004, places=3)
        self.assertAlmostEqual(w.average_fps(), 1000 / 20.004, places=2)

    def test_window_keeps_only_recent_duration(self):
        """Frames older than the window duration are dropped"""
        w = FrametimeWindow(100)
        for _ in range(10):
            w.add(10.0)
        self.assertEqual(len(w), 10)
        w.add(50.0)
        self.assertEqual(len(w), 6)  # 5 x 10ms + 50ms
        self.assertAlmostEqual(w.total_ms, 100)
        self.assertEqual(w.bucket_counts([(0, 20, "a", None), (20, math.inf, "b", None)]), [5, 1])

    def test_single_long_frame_is_kept(self):
        """A frame longer than the whole window still counts"""
        w = FrametimeWindow(100)
        w.add(500.0)
        self.assertEqual(len(w), 1)

    def test_thirty_minute_window_at_60fps(self):
        """A 30 minute window holds 30 minutes of smooth 60fps frames"""
        w = FrametimeWindow(30 * 60 * 1000)
        ft = 1000.0 / 60
        for _ in range(60 * 60 * 31):
            w.add(ft)
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


class TestFrameDifference(unittest.TestCase):
    """Test cases for the downscaled greyscale frame difference"""

    def test_identical_frames(self):
        frame = np.full((720, 1280, 3), 90, np.uint8)
        self.assertEqual(frame_difference(frame, frame.copy()), 0)

    def test_different_frames(self):
        a = np.full((720, 1280, 3), 100, np.uint8)
        b = np.full((720, 1280, 3), 110, np.uint8)
        self.assertAlmostEqual(frame_difference(a, b), 10, places=0)


class TestFileVideoStreamReading(unittest.TestCase):
    """Reading a real (generated) video through the background thread"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_reads_all_frames_then_none(self):
        path = os.path.join(self.temp_dir, "clip.mp4")
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (320, 240))
        for i in range(30):
            out.write(np.full((240, 320, 3), i * 5, np.uint8))
        out.release()

        # A small queue makes the decoder wait for the reader
        fvs = FileVideoStream(path, queue_size=4).start()
        frames = []
        while True:
            frame = fvs.read()
            if frame is None:
                break
            frames.append(frame)
        fvs.stop()
        self.assertEqual(len(frames), 30)
        self.assertEqual(frames[0].shape, (720, 1280, 3))
        self.assertFalse(fvs.thread.is_alive())


if __name__ == '__main__':
    unittest.main()
