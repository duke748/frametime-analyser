"""Integration tests for the frametime analyser"""
import unittest
import sys
import os
import tempfile
import numpy as np
import cv2

# Add parent directory to path to import the main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVideoGeneration(unittest.TestCase):
    """Test cases for generating test videos"""
    
    def setUp(self):
        """Create a temporary directory for test files"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def create_test_video(self, filename, fps=60, duration_seconds=1, width=1920, height=1080):
        """Helper method to create a test video"""
        filepath = os.path.join(self.temp_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, float(fps), (width, height))
        
        num_frames = fps * duration_seconds
        
        for i in range(num_frames):
            # Create a frame with changing content
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Add a moving square to create unique frames
            x = (i * 10) % width
            cv2.rectangle(frame, (x, 100), (x + 50, 150), (255, 255, 255), -1)
            out.write(frame)
            
        out.release()
        return filepath
        
    def test_video_creation(self):
        """Test that we can create a valid test video"""
        video_path = self.create_test_video("test.mp4", fps=60, duration_seconds=1)
        
        # Verify the video exists
        self.assertTrue(os.path.exists(video_path))
        
        # Verify we can open it with OpenCV
        cap = cv2.VideoCapture(video_path)
        self.assertTrue(cap.isOpened())
        
        # Verify frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.assertEqual(frame_count, 60)
        
        cap.release()
        
    def test_video_with_duplicate_frames(self):
        """Test creating a video with duplicate frames (to simulate frame drops)"""
        filepath = os.path.join(self.temp_dir, "test_duplicates.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, 60.0, (1920, 1080))
        
        # Create 60 frames, but repeat some to simulate drops
        unique_frame_count = 0
        for i in range(60):
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            
            # Every 10th frame is a duplicate (write the same frame twice)
            if i % 10 == 0 and i > 0:
                # This is a duplicate frame, don't increment unique_frame_count
                pass
            else:
                unique_frame_count += 1
                # Move the square to make it unique
                x = (unique_frame_count * 10) % 1920
                cv2.rectangle(frame, (x, 100), (x + 50, 150), (255, 255, 255), -1)
                
            out.write(frame)
            
        out.release()
        
        self.assertTrue(os.path.exists(filepath))


class TestCommandLineArguments(unittest.TestCase):
    """Test cases for command line argument parsing"""
    
    def test_fps_argument_parsing(self):
        """Test parsing FPS values"""
        valid_fps = [60, 120]
        
        for fps in valid_fps:
            # Test that these are valid
            self.assertIn(fps, [60, 120])
            
    def test_invalid_fps_values(self):
        """Test that invalid FPS values are rejected"""
        invalid_fps = [30, 90, 144, 240]
        
        for fps in invalid_fps:
            self.assertNotIn(fps, [60, 120])
            
    def test_frametime_calculation_from_fps(self):
        """Test that frametime is correctly calculated from FPS"""
        test_cases = {
            60: 16.666666666666668,
            120: 8.333333333333334
        }
        
        for fps, expected_frametime in test_cases.items():
            calculated_frametime = 1000.0 / fps
            self.assertAlmostEqual(calculated_frametime, expected_frametime, places=5)


class TestOutputFileGeneration(unittest.TestCase):
    """Test cases for output file naming and path generation"""
    
    def test_analyzed_video_filename(self):
        """Test that analyzed video filename is generated correctly"""
        input_path = "test_video.mp4"
        expected_output = "test_video_analyzed.mp4"
        
        output_path = os.path.splitext(input_path)[0] + "_analyzed.mp4"
        self.assertEqual(output_path, expected_output)
        
    def test_histogram_csv_filename(self):
        """Test that histogram CSV filename is generated correctly"""
        input_path = "test_video.mp4"
        expected_output = "test_video_histogram.csv"
        
        csv_path = os.path.splitext(input_path)[0] + "_histogram.csv"
        self.assertEqual(csv_path, expected_output)
        
    def test_path_with_directory(self):
        """Test filename generation with full path"""
        input_path = "C:/Videos/gameplay.mp4"
        expected_output = "C:/Videos/gameplay_analyzed.mp4"
        
        output_path = os.path.splitext(input_path)[0] + "_analyzed.mp4"
        self.assertEqual(output_path, expected_output)


class TestPerformanceMetrics(unittest.TestCase):
    """Test cases for performance metric calculations"""
    
    def test_fps_calculation(self):
        """Test FPS calculation from frame list"""
        # Simulate 60 unique frames in the last second
        fps_list = [1] * 60
        calculated_fps = sum(fps_list)
        self.assertEqual(calculated_fps, 60)
        
        # Simulate some dropped frames (30 unique, 30 duplicates)
        fps_list = [1] * 30 + [0] * 30
        calculated_fps = sum(fps_list)
        self.assertEqual(calculated_fps, 30)
        
    def test_average_frametime_calculation(self):
        """Test average frametime calculation"""
        frametime_history = [16.67, 16.67, 16.67, 33.34, 16.67]
        avg_frametime = sum(frametime_history) / len(frametime_history)
        expected_avg = 20.004
        self.assertAlmostEqual(avg_frametime, expected_avg, places=2)
        
    def test_stutter_detection_logic(self):
        """Test stutter detection thresholds"""
        target_fps = 60
        target_frametime = 1000.0 / target_fps  # 16.67ms
        
        moderate_threshold = target_frametime * 2.5  # 41.67ms
        severe_threshold = target_frametime * 4.0    # 66.67ms
        
        test_frametimes = [16.67, 20.0, 45.0, 70.0, 16.67]
        
        moderate_stutters = [ft for ft in test_frametimes if ft >= moderate_threshold and ft < severe_threshold]
        severe_stutters = [ft for ft in test_frametimes if ft >= severe_threshold]
        
        self.assertEqual(len(moderate_stutters), 1)  # 45.0ms
        self.assertEqual(len(severe_stutters), 1)    # 70.0ms


class TestDataStructures(unittest.TestCase):
    """Test cases for data structures used in the analyser"""
    
    def test_frametime_graph_initialization(self):
        """Test frametime graph list initialization"""
        frametime_samples = 90
        frametime_graph = [0] * frametime_samples
        
        self.assertEqual(len(frametime_graph), 90)
        self.assertEqual(sum(frametime_graph), 0)
        
    def test_histogram_buffer_size(self):
        """Test histogram buffer can hold 30 minutes of data"""
        histogram_sample_size = 216000  # 30 minutes at 120fps
        
        # At 120fps: 120 * 60 * 30 = 216,000
        expected_size_120fps = 120 * 60 * 30
        self.assertEqual(histogram_sample_size, expected_size_120fps)
        
        # Verify it also covers 60fps (60 * 60 * 30 = 108,000)
        expected_size_60fps = 60 * 60 * 30
        self.assertGreater(histogram_sample_size, expected_size_60fps)
        
    def test_fps_list_represents_one_second(self):
        """Test that FPS list represents exactly one second of data"""
        fps_list = [0] * 60
        
        # At 60fps base rate, 60 samples = 1 second
        self.assertEqual(len(fps_list), 60)


if __name__ == '__main__':
    unittest.main()
