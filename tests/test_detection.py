"""Tests for frametime_analyser.detection: analysis image, masks, tearing"""
import unittest

import numpy as np

from frametime_analyser.detection import (
    ANALYSIS_SIZE,
    AnalysisMask,
    Rect,
    TearDetector,
    analysis_image,
    frame_difference,
    parse_rect,
)


class TestAnalysisImage(unittest.TestCase):
    def test_size_and_grey(self):
        img = analysis_image(np.full((1080, 1920, 3), 100, np.uint8))
        self.assertEqual(img.shape, (72, 128))
        self.assertEqual(int(img[36, 64]), 100)

    def test_letterbox_matches_canvas(self):
        img = analysis_image(np.full((1440, 3440, 3), 200, np.uint8))
        self.assertEqual(int(img[0, 64]), 0)
        self.assertEqual(int(img[36, 64]), 200)


class TestFrameDifference(unittest.TestCase):
    def test_identical_frames(self):
        g = np.full((72, 128), 90, np.uint8)
        self.assertEqual(frame_difference(g, g.copy()), 0)

    def test_different_frames(self):
        a = np.full((72, 128), 100, np.uint8)
        b = np.full((72, 128), 110, np.uint8)
        self.assertAlmostEqual(frame_difference(a, b), 10)

    def test_rows(self):
        a = np.zeros((72, 128), np.uint8)
        b = a.copy()
        b[40:] = 50
        mean, rows = frame_difference(a, b, rows=True)
        self.assertEqual(rows[39], 0)
        self.assertEqual(rows[40], 50)
        self.assertAlmostEqual(mean, 50 * 32 / 72)


class TestAnalysisMask(unittest.TestCase):
    def test_parse_rect(self):
        self.assertEqual(parse_rect("10, 20,300,400"), Rect(10, 20, 300, 400))
        for bad in ("1,2,3", "a,b,c,d", "0,0,0,10", "-1,0,10,10"):
            with self.assertRaises(ValueError):
                parse_rect(bad)

    def test_full_picture(self):
        m = AnalysisMask(1920, 1080)
        self.assertEqual(m.count, 128 * 72)
        self.assertTrue(m.full)
        self.assertEqual(m.picture_rows, (0, 72))

    def test_letterbox_bars_are_excluded(self):
        m = AnalysisMask(3440, 1440)
        top = m.picture_rows[0]
        self.assertGreater(top, 0)
        self.assertEqual(m.mask[0].max(), 0)
        self.assertEqual(m.mask[36].min(), 255)

    def test_ignore_region(self):
        """Changes inside an ignored region (e.g. a webcam) don't count"""
        m = AnalysisMask(1920, 1080, ignore=[Rect(1440, 0, 480, 270)])  # Top-right quarter
        self.assertFalse(m.full)
        a = np.zeros((72, 128), np.uint8)
        b = a.copy()
        b[0:18, 96:128] = 255
        self.assertEqual(frame_difference(a, b, m), 0)
        self.assertGreater(frame_difference(a, b), 0)

    def test_roi(self):
        m = AnalysisMask(1920, 1080, roi=Rect(0, 540, 1920, 540))  # Bottom half only
        self.assertEqual(m.count, 128 * 36)
        a = np.zeros((72, 128), np.uint8)
        b = a.copy()
        b[:36] = 255  # Top half changes
        self.assertEqual(frame_difference(a, b, m), 0)
        _, rows = frame_difference(a, b, m, rows=True)
        self.assertTrue(np.isnan(rows[0]))
        self.assertEqual(rows[60], 0)

    def test_canvas_rects(self):
        m = AnalysisMask(1920, 1080, roi=Rect(0, 0, 960, 540), ignore=[Rect(0, 0, 192, 108)])
        roi, ignore = m.canvas_rects()
        self.assertEqual(roi, (0, 0, 640, 360))
        self.assertEqual(ignore, [(0, 0, 130, 80)])

    def test_too_small(self):
        with self.assertRaises(ValueError):
            AnalysisMask(1920, 1080, ignore=[Rect(0, 0, 1920, 1080)])


class TestTearDetector(unittest.TestCase):
    """Row patterns -> where new frames start"""

    def rows(self, changed_ranges, n=72, level=40.0):
        r = np.zeros(n)
        for s, e in changed_ranges:
            r[s:e] = level
        return r

    def test_whole_frame_change_is_new_frame_at_top(self):
        d = TearDetector((0, 72))
        self.assertEqual(d.detect(self.rows([(0, 72)]), 10), [0.0])

    def test_no_change(self):
        d = TearDetector((0, 72))
        self.assertEqual(d.detect(self.rows([]), 10), [])

    def test_tear_and_catch_up(self):
        d = TearDetector((0, 72))
        self.assertEqual(d.detect(self.rows([(36, 72)]), 10), [0.5])  # New frame from halfway
        self.assertEqual(d.last_tear_rows, [36])
        self.assertEqual(d.detect(self.rows([(0, 36)]), 10), [])      # Top half catching up
        self.assertEqual(d.detect(self.rows([(0, 72)]), 10), [0.0])   # Then a normal new frame

    def test_catch_up_plus_new_tear(self):
        d = TearDetector((0, 72))
        d.detect(self.rows([(18, 72)]), 10)
        self.assertEqual(d.detect(self.rows([(0, 18), (54, 72)]), 10), [0.75])

    def test_single_row_blip_ignored(self):
        d = TearDetector((0, 72))
        self.assertEqual(d.detect(self.rows([(40, 41)]), 10), [])

    def test_letterboxed_position(self):
        d = TearDetector((12, 60))
        rows = self.rows([(36, 60)])
        rows[:12] = np.nan
        rows[60:] = np.nan
        self.assertEqual(d.detect(rows, 10), [0.5])

    def test_static_rows_are_learned(self):
        """A HUD that never changes at the top isn't mistaken for a tear"""
        d = TearDetector((0, 72))
        results = [d.detect(self.rows([(8, 72)]), 10) for _ in range(40)]
        self.assertEqual(results[-1], [0.0])
        self.assertTrue(all(len(r) == 1 for r in results))

    def test_rows_in_analysis_size(self):
        self.assertEqual(ANALYSIS_SIZE, (128, 72))


if __name__ == "__main__":
    unittest.main()
