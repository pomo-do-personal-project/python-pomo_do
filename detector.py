"""
detector.py - Module nhận diện tập trung thông minh (Focus Detector)
Dành cho dự án: pomo-ai (Pomo-do)
"""

import os
import time
from collections import deque, Counter
from screen_manager import ScreenManager
from vision_pipeline import VisionPipeline


class FocusDetector:
    """Class chính điều phối nhận diện tập trung và tính toán trạng thái."""

    def __init__(self, config_filename="config.json"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.screen_manager = ScreenManager(config_filename)
        self.vision = VisionPipeline(current_dir)

        self.tolerance_x = 0.22
        self.tolerance_y = 0.18
        self.eye_closed_threshold = 0.013
        self.eye_closed_start_time = None

        self.last_look_x = 0.0
        self.last_look_y = 0.0
        self.has_face = False
        self.history = deque(maxlen=8)

        print("[pomo-ai] Khởi tạo FocusDetector thành công!")

    @property
    def screens(self):
        return self.screen_manager.screens

    def add_screen(self, x=None, y=None):
        return self.screen_manager.add_screen(x, y, self.has_face, self.last_look_x, self.last_look_y)

    def reset_screens(self):
        return self.screen_manager.reset_screens()

    def check_focus(self, frame):
        """Nhận diện trạng thái tập trung từ 1 khung hình camera."""
        raw_status = "AFK"
        active_screen = None
        look_x, look_y = None, None
        h, w = frame.shape[:2]

        # 1. Trích xuất MediaPipe landmarks
        landmarks = self.vision.extract_landmarks(frame)
        points, mesh_points, face_box = None, [], None

        if landmarks is not None:
            self.has_face = True
            mesh_points = [(int(p.x * w), int(p.y * h)) for p in landmarks]
            xs, ys = [pt[0] for pt in mesh_points], [pt[1] for pt in mesh_points]
            pad_x = int((max(xs) - min(xs)) * 0.1)
            pad_y = int((max(ys) - min(ys)) * 0.15)
            face_box = [max(0, min(xs) - pad_x), max(0, min(ys) - pad_y), min(w, max(xs) + pad_x), min(h, max(ys) + int(pad_y * 0.6))]

            nose, left_eye, right_eye = landmarks[1], landmarks[33], landmarks[263]
            upper_lid, lower_lid = landmarks[159], landmarks[145]

            points = {
                "nose": (int(nose.x * w), int(nose.y * h)),
                "left_eye": (int(left_eye.x * w), int(left_eye.y * h)),
                "right_eye": (int(right_eye.x * w), int(right_eye.y * h))
            }

            # Kiểm tra nhắm mắt / ngủ gật
            if abs(upper_lid.y - lower_lid.y) < self.eye_closed_threshold:
                now = time.time()
                if self.eye_closed_start_time is None:
                    self.eye_closed_start_time = now
                elif (now - self.eye_closed_start_time) >= 2.5:
                    raw_status = "SLEEPING"
            else:
                self.eye_closed_start_time = None

            eye_cx = (left_eye.x + right_eye.x) / 2.0
            eye_cy = (left_eye.y + right_eye.y) / 2.0
            look_x = round(nose.x - eye_cx, 3)
            look_y = round(nose.y - eye_cy, 3)
        else:
            # 2. Fallback sang YOLOv8 Pose khi quay mặt góc lớn
            pose_ok, p_lx, p_ly, p_pts, p_box = self.vision.estimate_pose_look(frame, w, h)
            if pose_ok:
                self.has_face = True
                self.eye_closed_start_time = None
                look_x, look_y, points = p_lx, p_ly, p_pts
                if face_box is None:
                    face_box = p_box
            else:
                self.has_face = False
                self.eye_closed_start_time = None
                raw_status = "AFK"

        # 3. Quét nhận diện điện thoại
        has_phone, phone_boxes = self.vision.detect_phone(frame, face_box, look_x)

        # 4. Phân loại trạng thái
        detail = ""
        if self.has_face:
            self.last_look_x, self.last_look_y = look_x, look_y

            if raw_status == "SLEEPING":
                raw_status = "DISTRACTED"
                detail = "Nhắm mắt ngủ gật (>2.5s)"
            elif has_phone:
                raw_status = "DISTRACTED"
                detail = "Phát hiện dùng điện thoại"
            else:
                matched_id = self.screen_manager.match_screen(look_x, look_y, self.tolerance_x, self.tolerance_y)
                if matched_id is not None:
                    raw_status = "FOCUSED"
                    detail = f"Màn hình {matched_id}"
                    active_screen = matched_id
                else:
                    raw_status = "DISTRACTED"
                    detail = "Nhìn ra ngoài vùng màn hình"
        else:
            raw_status = "AFK"
            detail = "Không thấy người trước camera"

        # 5. Làm mượt trạng thái với Deque
        self.history.append(raw_status)
        smoothed = Counter(self.history).most_common(1)[0][0]
        is_focused = (smoothed == "FOCUSED")
        if is_focused:
            detail = f"Màn hình {active_screen}" if active_screen else "Góc nhìn chuẩn xác"
        elif not detail:
            detail = "Mất tập trung"

        return {
            "status": smoothed,
            "detail": detail,
            "is_focused": is_focused,
            "active_screen": active_screen if is_focused else None,
            "total_screens": len(self.screens),
            "look_x": look_x,
            "look_y": look_y,
            "has_face": self.has_face,
            "has_phone": has_phone,
            "points": points,
            "mesh_points": mesh_points,
            "face_box": face_box,
            "phone_boxes": phone_boxes
        }
