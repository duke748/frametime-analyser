"""Tests for frametime_analyser.overlay"""
import unittest

import numpy as np

from frametime_analyser.analyser import FrametimeAnalyser
from frametime_analyser.detection import AnalysisMask, Rect
from frametime_analyser.overlay import KEY, CompareOverlay, GraphBox, Layer, Overlay, draw_step_graph


def feed(a, pattern):
    for c in pattern:
        a.process(20.0 if c == "u" else 0.0)


class TestLayer(unittest.TestCase):
    def test_only_drawn_pixels_are_copied(self):
        layer = Layer((64, 32))
        layer.img[5:10, 5:10] = (0, 0, 0)       # Black counts as drawn
        layer.img[20:25, 40:50] = (200, 100, 50)
        layer.finish()
        dst = np.full((32, 64, 3), 77, np.uint8)
        layer.apply(dst)
        self.assertEqual(tuple(dst[7, 7]), (0, 0, 0))
        self.assertEqual(tuple(dst[22, 45]), (200, 100, 50))
        self.assertEqual(tuple(dst[0, 0]), (77, 77, 77))

    def test_clear(self):
        layer = Layer((8, 8))
        layer.img[:] = 0
        layer.clear()
        self.assertEqual(tuple(layer.img[4, 4]), KEY)
        self.assertIsNone(layer.mask)


class TestStepGraph(unittest.TestCase):
    def test_draws_only_valid_samples(self):
        img = np.zeros((200, 400, 3), np.uint8)
        box = GraphBox(10, 150, 270, 100)
        values = [0.0] * 80 + [16.7] * 10
        n = draw_step_graph(img, values, 10, box, 41.7, (0, 255, 0))
        self.assertEqual(n, 90)
        drawn = np.flatnonzero(img[:, :, 1].max(axis=0))
        self.assertGreaterEqual(drawn.min(), box.x + 80 * 3)


class TestOverlay(unittest.TestCase):
    def test_all_options_draw(self):
        """Every overlay option draws, with and without data, at several target FPS"""
        wm = np.full((32, 32, 4), 255, np.uint8)
        mask = AnalysisMask(1920, 1080, roi=Rect(0, 0, 1900, 1000), ignore=[Rect(0, 0, 200, 100)])
        for delta in (False, True):
            for fps in (30, 60, 120, 144):
                a = FrametimeAnalyser(fps, detect_stutters=True, detect_tears=True)
                overlay = Overlay(fps, delta, True, True, "Title", wm, mask)
                img = np.zeros((720, 1280, 3), np.uint8)
                overlay.draw(img, a)
                feed(a, "uuuddddu" * 100 + "d" * fps * 3)
                img2 = np.zeros((720, 1280, 3), np.uint8)
                overlay.draw(img2, a)
                self.assertGreater(img2.sum(), img.sum())

    def test_slow_layer_is_cached(self):
        a = FrametimeAnalyser(60)
        overlay = Overlay(60)
        feed(a, "u" * 100)
        overlay.draw(np.zeros((720, 1280, 3), np.uint8), a)
        layer_mask = overlay.slow.mask
        a.process(20.0)  # Same 1/10 second, same FPS text
        overlay.draw(np.zeros((720, 1280, 3), np.uint8), a)
        self.assertIs(overlay.slow.mask, layer_mask)

    def test_compare_overlay(self):
        a, b = FrametimeAnalyser(60, frametime_samples=180), FrametimeAnalyser(60, frametime_samples=180)
        overlay = CompareOverlay(60, ("A", "B"), detect_stutters=True, detect_tears=True, title="T",
                                 watermark=np.full((32, 32, 3), 255, np.uint8))
        feed(a, "u" * 200)
        feed(b, "ud" * 100)
        img = overlay.compose(np.full((360, 640, 3), 50, np.uint8), np.full((360, 640, 3), 90, np.uint8))
        self.assertEqual(img.shape, (720, 1280, 3))
        overlay.draw(img, a, b)
        self.assertEqual(int(img[300, 300, 0]), 50)
        self.assertEqual(int(img[300, 1000, 0]), 90)


if __name__ == "__main__":
    unittest.main()
