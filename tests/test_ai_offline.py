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
from vision_pipeline import VisionPipeline
from routes_session import analyze_session


class TestAiDetectorOffline(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.shared_vision = VisionPipeline(base_dir)

    @classmethod
    def tearDownClass(cls):
        for attr in ("face_landmarker", "hand_landmarker", "face_mesh"):
            obj = getattr(cls.shared_vision, attr, None)
            if obj is not None and hasattr(obj, "close"):
                try:
                    obj.close()
                except Exception:
                    pass

    def setUp(self):
        # Use an isolated temporary config file to prevent overwriting user's real config.json
        self.temp_config = "test_config_temp.json"
        self.detector = FocusDetector(config_filename=self.temp_config, vision=self.shared_vision)

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
        self.assertIn("event_code", result)
        self.assertIn("is_focused", result)
        self.assertIn("has_face", result)
        self.assertFalse(result["has_face"])
        self.assertEqual(result["status"], "AFK")
        self.assertEqual(result["event_code"], "AFK")

    def test_hand_on_cheek_and_background_false_positives_rejected(self):
        """Verify hand resting on cheek and unheld background objects do not trigger false phone detection."""
        import numpy as np

        pd = self.detector.vision.phone_detector
        frame = np.full((330, 440, 3), 60, dtype=np.uint8)
        face_box = [190, 39, 326, 214]

        # Hand landmarks matching the user's cheek-resting pose
        cheek_hand = {
            "points": [
                (218, 287), (195, 271), (182, 245), (174, 221), (170, 203),
                (212, 218), (209, 184), (208, 165), (207, 148),
                (233, 219), (233, 183), (234, 162), (235, 145),
                (250, 227), (262, 200), (272, 183), (280, 170),
                (264, 237), (281, 218), (294, 204), (304, 192),
            ],
            "box": [170, 145, 304, 287],
            "center": (237, 216),
            "radius": 71,
            "holding_phone": False,
        }

        self.assertTrue(pd.is_flat_open_hand(cheek_hand))
        self.assertTrue(pd.is_hand_resting_on_face(cheek_hand, face_box))
        self.assertFalse(pd.is_calling_pose(cheek_hand, face_box, look_x=0.0, head_tilt_deg=0.0))

        # Mock YOLO returning the 65% background laundry basket and 76% cheek/glasses artifact
        orig_onnx = pd.detect_phone_onnx
        try:
            pd.detect_phone_onnx = lambda *a, **kw: [
                {"box": [88, 141, 168, 280], "conf": 0.65},
                {"box": [152, 160, 192, 211], "conf": 0.76},
            ]
            has_phone, boxes = pd.detect_phone(frame, face_box, look_x=0.0, hands=[cheek_hand])
            self.assertFalse(has_phone)
            self.assertEqual(len(boxes), 0)
            self.assertFalse(cheek_hand["holding_phone"])
        finally:
            pd.detect_phone_onnx = orig_onnx

    def test_c_grip_and_ear_grip_phone_detection(self):
        """Verify holding a phone in front of face (C-grip) or tilted at ear is reliably detected."""
        import numpy as np
        import cv2

        pd = self.detector.vision.phone_detector
        face_box = [232, 79, 349, 232]

        # 1. C-grip hand holding phone vertically in front of face
        c_grip_hand = {
            "points": [
                (142, 329), (168, 297), (166, 255), (154, 223), (144, 198),
                (120, 246), (105, 199), (100, 174), (96, 154),
                (100, 257), (85, 220), (88, 198), (94, 183),
                (90, 267), (76, 232), (83, 214), (91, 204),
                (85, 276), (76, 247), (82, 229), (89, 218),
            ],
            "box": [76, 154, 168, 329],
            "center": (106, 234),
            "radius": 87,
            "holding_phone": False,
        }
        gm_front = pd._hand_grip_metrics(c_grip_hand["points"])
        self.assertTrue(gm_front["is_c_grip"])
        self.assertFalse(pd.is_flat_open_hand(c_grip_hand))
        self.assertFalse(pd.is_hand_resting_on_face(c_grip_hand, face_box))

        # Synthetic frame with grey-blue phone slab between fingers and thumb
        frame_front = np.full((330, 440, 3), 200, dtype=np.uint8)
        cv2.rectangle(frame_front, (85, 105), (150, 255), (135, 120, 115), -1)
        cv2.rectangle(frame_front, (95, 115), (135, 160), (80, 75, 70), 2)
        orig_onnx = pd.detect_phone_onnx
        try:
            pd.detect_phone_onnx = lambda *a, **kw: [{"box": [86, 102, 147, 225], "conf": 0.24}]
            slab_front = pd.detect_grip_phone_slab(frame_front, c_grip_hand, face_box)
            self.assertIsNotNone(slab_front)
        finally:
            pd.detect_phone_onnx = orig_onnx

        # 2. Angled ear grip hand holding phone tilted against ear
        ear_face_box = [237, 25, 332, 217]
        ear_grip_hand = {
            "points": [
                (295, 278), (305, 244), (294, 216), (275, 199), (257, 192),
                (285, 198), (267, 162), (249, 162), (239, 169),
                (275, 199), (244, 173), (225, 188), (218, 204),
                (260, 207), (226, 193), (214, 210), (212, 224),
                (244, 221), (218, 211), (209, 221), (207, 229),
            ],
            "box": [207, 162, 305, 278],
            "center": (252, 209),
            "radius": 58,
            "holding_phone": False,
        }
        gm_ear = pd._hand_grip_metrics(ear_grip_hand["points"])
        self.assertTrue(gm_ear["is_angled_ear_grip"])
        self.assertFalse(pd.is_flat_open_hand(ear_grip_hand))
        self.assertFalse(pd.is_hand_resting_on_face(ear_grip_hand, ear_face_box))
        self.assertTrue(pd.is_calling_pose(ear_grip_hand, ear_face_box))

        # 3. Bare hand held sideways/diagonally in front of background wall must NOT trigger phone or call
        side_face_box = [228, 111, 321, 221]
        bare_side_hand = {
            "points": [
                (220, 229), (223, 200), (213, 176), (199, 163), (186, 157),
                (192, 169), (169, 159), (155, 154), (144, 149),
                (187, 180), (163, 169), (149, 162), (137, 156),
                (183, 194), (163, 185), (151, 177), (141, 170),
                (181, 209), (167, 204), (159, 198), (151, 192),
            ],
            "box": [137, 149, 223, 229],
            "center": (175, 180),
            "radius": 43,
            "holding_phone": False,
        }
        gm_side = pd._hand_grip_metrics(bare_side_hand["points"])
        self.assertFalse(gm_side["is_c_grip"])
        self.assertFalse(gm_side["is_angled_ear_grip"])
        self.assertTrue(pd.is_flat_open_hand(bare_side_hand))
        self.assertFalse(pd.is_calling_pose(bare_side_hand, side_face_box))
        self.assertIsNone(pd.detect_grip_phone_slab(frame_front, bare_side_hand, side_face_box))

        # 4. Pointing 1 index finger overlapping a background clothes rack box (PHONE 56%) must be rejected
        pointing_hand = {
            "points": [
                (290, 250), (288, 222), (280, 198), (265, 185), (255, 180),
                (260, 182), (255, 155), (252, 138), (250, 122),
                (245, 190), (235, 195), (238, 205), (242, 210),
                (235, 200), (225, 205), (228, 214), (232, 218),
                (228, 210), (218, 215), (222, 222), (226, 225),
            ],
            "box": [218, 122, 290, 250],
            "center": (254, 186),
            "radius": 64,
            "holding_phone": False,
        }
        clothes_rack_box = [228, 72, 288, 178]
        self.assertTrue(pd.is_non_holding_gesture(pointing_hand))
        self.assertFalse(pd.is_hand_attached(clothes_rack_box, [pointing_hand], side_face_box))

        # 5. Verify background clothes-on-rack crop (bright wall top third + dark cloth bottom) is rejected by chassis check
        rack_crop = np.zeros((100, 60, 3), dtype=np.uint8)
        rack_crop[:32, :] = (180, 180, 180)  # bright neutral wall at top
        rack_crop[32:, :] = (30, 30, 30)     # dark hanging clothes below
        has_chassis, _, _ = pd.detect_phone_chassis_features(rack_crop)
        self.assertFalse(has_chassis)

    def test_hybrid_vlm_verifier_parsing_and_fusion(self):
        """Verify VLMPhoneVerifier parses security camera JSON schema strictly and caches results."""
        import time
        from vlm_verifier import VLMPhoneVerifier

        vlm = VLMPhoneVerifier()
        raw_json = """{
          "person_present": true,
          "phone_detected": true,
          "action_type": "calling",
          "confidence_score": 0.96,
          "analysis_vietnamese": "Người dùng đang cầm điện thoại áp sát vào tai bên phải với cụm camera lộ rõ."
        }"""
        parsed = vlm._parse_vlm_json(raw_json)
        self.assertIsNotNone(parsed)
        self.assertTrue(parsed["person_present"])
        self.assertTrue(parsed["phone_detected"])
        self.assertEqual(parsed["action_type"], "calling")
        self.assertEqual(parsed["confidence_score"], 0.96)
        self.assertIn("camera", parsed["analysis_vietnamese"])

        # Verify negative VLM response normalizes action_type to 'none'
        raw_neg = """```json
        {
          "person_present": true,
          "phone_detected": false,
          "action_type": "calling",
          "confidence_score": 0.92,
          "analysis_vietnamese": "Người dùng chỉ đang đặt bàn tay trống lên má nghỉ ngơi."
        }
        ```"""
        parsed_neg = vlm._parse_vlm_json(raw_neg)
        self.assertFalse(parsed_neg["phone_detected"])
        self.assertEqual(parsed_neg["action_type"], "none")

        # Verify TTL cache behavior
        vlm._last_result = parsed
        vlm._last_result_time = time.time()
        self.assertIsNotNone(vlm.get_recent_result())
        vlm.clear_result()
        self.assertIsNone(vlm.get_recent_result())

    def test_background_blur_and_unheld_bag_rejection(self):
        """AI-H06: Verify background blur mode works and unheld bags behind user are strictly rejected."""
        import numpy as np

        # 1. Verify blur background flag toggle
        res = self.detector.set_blur_background(True)
        self.assertTrue(res["blur_background"])
        self.assertTrue(self.detector.blur_background)

        # 2. Verify synthetic bag placed in background (offset to the left, no hand holding)
        pd = self.detector.vision.phone_detector
        face_box = [220, 60, 340, 210]
        # Bag behind user (x: 60-140, far to the left of the face_box)
        bag_box = [60, 100, 140, 240]
        frame = np.full((320, 400, 3), 100, dtype=np.uint8)

        orig_onnx = pd.detect_phone_onnx
        try:
            pd.detect_phone_onnx = lambda *a, **kw: [{"box": bag_box, "conf": 0.65}]
            # No hands present, user looking forward (look_x=0, look_y=0)
            has_phone, boxes = pd.detect_phone(frame, face_box, look_x=0.0, hands=[], look_y=0.0)
            self.assertFalse(has_phone, "Unheld background bag/chair must NOT be counted as phone")
            self.assertEqual(len(boxes), 0)

            # Test laundry basket right behind shoulder with hands on desk typing
            basket_box = [150, 90, 230, 210]
            desk_hand = {
                "box": [80, 260, 160, 310],
                "points": [(120, 280)] * 21,
                "holding_phone": False,
            }
            pd.detect_phone_onnx = lambda *a, **kw: [{"box": basket_box, "conf": 0.88}]
            has_phone_basket, boxes_basket = pd.detect_phone(
                frame, face_box, look_x=0.0, hands=[desk_hand], look_y=0.0
            )
            self.assertFalse(has_phone_basket, "Laundry hamper behind shoulder must NOT be detected as phone")
            self.assertEqual(len(boxes_basket), 0)
        finally:
            pd.detect_phone_onnx = orig_onnx

        # 3. Verify apply_background_blur returns composed frame and mask
        composed, mask = self.detector.vision.apply_background_blur(frame, ksize=25)
        self.assertEqual(composed.shape, frame.shape)
        self.assertEqual(composed.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
