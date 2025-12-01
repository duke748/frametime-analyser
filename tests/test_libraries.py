"""Unit tests for libraries module"""
import unittest
import numpy as np
import cv2
from libraries import ResizeWithAspectRatio, FileVideoStream


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
    """Test cases for histogram bucket calculations"""
    
    def test_60fps_histogram_buckets(self):
        """Test histogram bucket ranges for 60fps"""
        target_fps = 60
        target_frametime = 1000.0 / target_fps
        
        # Test bucket thresholds
        perfect_threshold = target_frametime * 1.08  # ~18ms
        good_threshold = target_frametime * 1.5      # ~25ms
        fair_threshold = target_frametime * 2.1      # ~35ms
        stutter_threshold = target_frametime * 3.0   # ~50ms
        
        self.assertAlmostEqual(perfect_threshold, 18.0, places=1)
        self.assertAlmostEqual(good_threshold, 25.0, places=1)
        self.assertAlmostEqual(fair_threshold, 35.0, places=1)
        self.assertAlmostEqual(stutter_threshold, 50.0, places=1)
        
    def test_120fps_histogram_buckets(self):
        """Test histogram bucket ranges for 120fps"""
        target_fps = 120
        target_frametime = 1000.0 / target_fps
        
        # Test bucket thresholds
        perfect_threshold = target_frametime * 1.08  # ~9ms
        good_threshold = target_frametime * 1.44     # ~12ms
        fair_threshold = target_frametime * 2.04     # ~17ms
        stutter_threshold = target_frametime * 3.0   # ~25ms
        
        self.assertAlmostEqual(perfect_threshold, 9.0, places=1)
        self.assertAlmostEqual(good_threshold, 12.0, places=1)
        self.assertAlmostEqual(fair_threshold, 17.0, places=1)
        self.assertAlmostEqual(stutter_threshold, 25.0, places=1)


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


if __name__ == '__main__':
    unittest.main()
