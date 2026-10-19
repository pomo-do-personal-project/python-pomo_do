"""
test_ai_offline.py - Automated tests for pomo-ai computer vision logic and session scoring.
Runs completely offline without needing a physical hardware webcam.
"""

import os
import sys
import unittest
import asyncio

# Ensure pomo-ai directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detector import FocusDetector
from server import analyze_session


class TestAiDetectorOffline(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Use an isolated temporary config file to prevent overwriting user's real config.json
        self.temp_config = "test_config_temp.json"
        self.detector = FocusDetector(config_filename=self.temp_config)

    def tearDown(self):
        if os.path.exists(self.temp_config):
            try:
                os.remove(self.temp_config)
            except Exception:
                pass

    def test_ai_sensitivity_levels(self):
        """AI-H04: Verify sensitivity configuration updates tolerances properly."""
        # 1. Test Relaxed
        res_relaxed = self.detector.set_sensitivity("relaxed")
        self.assertEqual(res_relaxed["sensitivity"], "relaxed")
        self.assertEqual(self.detector.tolerance_x, 0.28)
        self.assertEqual(self.detector.phone_confirm_delay, 2.0)

        # 2. Test Strict
        res_strict = self.detector.set_sensitivity("strict")
        self.assertEqual(res_strict["sensitivity"], "strict")
        self.assertEqual(self.detector.tolerance_x, 0.16)
        self.assertEqual(self.detector.phone_confirm_delay, 1.0)

        # 3. Test Standard
        res_std = self.detector.set_sensitivity("standard")
        self.assertEqual(res_std["sensitivity"], "standard")
        self.assertEqual(self.detector.tolerance_x, 0.22)
        self.assertEqual(self.detector.phone_confirm_delay, 1.5)

    def test_ai_screen_calibration_logic(self):
        """AI-H04: Verify adding screens and resetting screens in ScreenManager."""
        self.detector.reset_screens()
        self.assertEqual(len(self.detector.screens), 0)

        # Explicitly add screen coordinates
        s1 = self.detector.add_screen(x=0.0, y=0.0)
        self.assertIsNotNone(s1)
        self.assertEqual(len(self.detector.screens), 1)

        s2 = self.detector.add_screen(x=0.5, y=-0.1)
        self.assertIsNotNone(s2)
        self.assertEqual(len(self.detector.screens), 2)

        # Reset
        self.detector.reset_screens()
        self.assertEqual(len(self.detector.screens), 0)

    async def test_analyze_session_perfect_score(self):
        """AI-H05: 100% focus ratio should yield Grade S."""
        payload = {
            "total_seconds": 1500,
            "focused_seconds": 1500,
            "phone_count": 0,
            "sleep_count": 0,
            "look_away_count": 0,
            "task_name": "Deep Work"
        }
        res = await analyze_session(payload)
        self.assertEqual(res["score"], 100)
        self.assertEqual(res["grade"], "S")
        self.assertIn("Xuất sắc", res["conclusion"])

    async def test_analyze_session_distracted_score(self):
        """AI-B03: High distraction should drop score and yield Grade D with advice."""
        payload = {
            "total_seconds": 1500,
            "focused_seconds": 500,
            "phone_count": 8,
            "sleep_count": 2,
            "look_away_count": 10,
            "task_name": "Math Homework"
        }
        res = await analyze_session(payload)
        self.assertEqual(res["score"], 33)
        self.assertEqual(res["grade"], "D")
        self.assertTrue(len(res["advice"]) > 0)

    async def test_analyze_session_zero_division_guard(self):
        """AI-B06: Zero total_seconds should not trigger ZeroDivisionError."""
        payload = {
            "total_seconds": 0,
            "focused_seconds": 0,
            "phone_count": 0,
            "sleep_count": 0,
            "look_away_count": 0
        }
        res = await analyze_session(payload)
        self.assertEqual(res["score"], 0)
        self.assertEqual(res["grade"], "D")

    def test_check_focus_frame_inference_offline(self):
        """AI-H02: Pass a synthetic 640x480 frame to verify MediaPipe and YOLO inference pipeline runs cleanly without crash."""
        import numpy as np
        # 640x480 3-channel black image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.check_focus(frame)

        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
        self.assertIn("is_focused", result)
        self.assertIn("has_face", result)
        self.assertFalse(result["has_face"])
        self.assertEqual(result["status"], "AFK")


if __name__ == "__main__":
    unittest.main()
