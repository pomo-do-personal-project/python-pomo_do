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

        # Chế độ Thoải mái mặc định & bộ đệm
        self.tolerance_x, self.tolerance_y = 0.28, 0.24
        self.eye_closed_threshold = 0.013
        self.eye_closed_start_time = None
        self.last_look_x, self.last_look_y = 0.0, 0.0
        self.has_face = False
        self.history = deque(maxlen=8)

        # Bộ đệm Face Persistence chống chớp tắt Face khi chạm tay
        self.last_face_time = 0.0
        self.last_valid_look_x, self.last_valid_look_y = 0.0, 0.0
        self.last_valid_face_box, self.last_valid_points = None, None

        # Bộ đệm nhịp khung hình để tối ưu CPU
        self.frame_count = 0
        self.last_has_phone = False
        self.last_phone_boxes = []
        self.last_suspicious_hand, self.last_suspicious_roi = False, None
        self.sensitivity, self.phone_confirm_delay = "relaxed", 2.0
        self.baseline_calibrated, self.baseline_info = False, None

        print("[pomo-ai] Khởi tạo FocusDetector thành công!")

    @property
    def screens(self):
        return self.screen_manager.screens

    def add_screen(self, x=None, y=None):
        return self.screen_manager.add_screen(x, y, self.has_face, self.last_look_x, self.last_look_y)

    def reset_screens(self):
        return self.screen_manager.reset_screens()

    def set_sensitivity(self, level="standard"):
        """Điều chỉnh độ nhạy AI: relaxed | standard | strict."""
        self.sensitivity = level
        if level == "relaxed":
            self.tolerance_x, self.tolerance_y, self.phone_confirm_delay = 0.28, 0.24, 2.0
        elif level == "strict":
            self.tolerance_x, self.tolerance_y, self.phone_confirm_delay = 0.16, 0.14, 1.0
        else:
            self.tolerance_x, self.tolerance_y, self.phone_confirm_delay = 0.22, 0.18, 1.5
        print(f"[pomo-ai] Cập nhật độ nhạy: {self.sensitivity} (delay={self.phone_confirm_delay}s)")
        return {"sensitivity": self.sensitivity, "delay": self.phone_confirm_delay}

    def calibrate_baseline(self):
        """Lưu dáng ngồi chuẩn và nhận diện mắt kính."""
        if not self.has_face:
            return {"success": False, "message": "Không thấy khuôn mặt"}
        self.baseline_calibrated = True
        self.baseline_info = {"look_x": self.last_look_x, "look_y": self.last_look_y}
        if len(self.screens) == 0:
            self.add_screen(self.last_look_x, self.last_look_y)
        return {"success": True, "baseline": self.baseline_info}

    def check_focus(self, frame):
        """Nhận diện trạng thái tập trung từ 1 khung hình camera."""
        raw_status = "AFK"
        active_screen = None
        look_x, look_y = None, None
        h, w = frame.shape[:2]

        now = time.time()
        # 1. Trích xuất MediaPipe landmarks
        landmarks = self.vision.extract_landmarks(frame)
        points, mesh_points, face_box = None, [], None

        if landmarks is not None:
            self.has_face = True
            self.last_face_time = now
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
                if self.eye_closed_start_time is None:
                    self.eye_closed_start_time = now
                elif (now - self.eye_closed_start_time) >= 2.5:
                    raw_status = "SLEEPING"
            else:
                self.eye_closed_start_time = None

            eye_cx = (left_eye.x + right_eye.x) / 2.0
            eye_cy = (left_eye.y + right_eye.y) / 2.0
            raw_look_x = round(nose.x - eye_cx, 3)
            raw_look_y = round(nose.y - eye_cy, 3)

            # Làm mượt hướng nhìn bằng Exponential Moving Average (EMA)
            if hasattr(self, "ema_look_x") and self.ema_look_x is not None:
                look_x = round(0.6 * self.ema_look_x + 0.4 * raw_look_x, 3)
                look_y = round(0.6 * self.ema_look_y + 0.4 * raw_look_y, 3)
            else:
                look_x = raw_look_x
                look_y = raw_look_y
            self.ema_look_x = look_x
            self.ema_look_y = look_y

            # Lưu lại trạng thái khuôn mặt hợp lệ gần nhất
            self.last_valid_look_x = look_x
            self.last_valid_look_y = look_y
            self.last_valid_face_box = face_box
            self.last_valid_points = points
        else:
            # 2. Xử lý khi MediaPipe FaceMesh bị che khuất tạm thời (chạm tay lên đầu/trán, chỉnh kính, gãi mặt)
            has_person, head_box = self.vision.detect_person_and_head(frame)
            time_since_face = now - self.last_face_time

            # Nếu YOLO xác nhận người vẫn đang ngồi trước bàn hoặc vừa thấy mặt trong 2.0s
            if has_person or (time_since_face < 2.0 and self.last_valid_face_box is not None):
                self.has_face = True
                self.eye_closed_start_time = None
                look_x = self.last_valid_look_x
                look_y = self.last_valid_look_y
                face_box = head_box if head_box is not None else self.last_valid_face_box
                points = self.last_valid_points
            else:
                self.has_face = False
                self.eye_closed_start_time = None
                self.ema_look_x = None
                self.ema_look_y = None
                raw_status = "AFK"

        # 3. Quét nhận diện điện thoại 2 tầng (chạy mỗi 3 khung hình để video mượt mà 30 FPS)
        self.frame_count += 1
        if self.frame_count % 3 == 0:
            self.last_suspicious_hand, self.last_suspicious_roi = self.vision.check_suspicious_hand_near_ear(frame, face_box)
            self.last_has_phone, self.last_phone_boxes = self.vision.detect_phone(frame, face_box, look_x, self.last_suspicious_roi)

        raw_has_phone = self.last_has_phone
        phone_boxes = self.last_phone_boxes
        is_suspicious_hand = self.last_suspicious_hand

        # Nguyên lý 1: Bộ đệm xác nhận thời gian (chỉ kích hoạt khi giữ điện thoại >= delay liên tục)
        now = time.time()
        if raw_has_phone:
            if not hasattr(self, "phone_start_time") or self.phone_start_time is None:
                self.phone_start_time = now
            has_phone = (now - self.phone_start_time) >= self.phone_confirm_delay
        else:
            self.phone_start_time = None
            has_phone = False

        # 4. Phân loại trạng thái
        detail = ""
        is_deep_flow = False
        if self.has_face:
            self.last_look_x, self.last_look_y = look_x, look_y

            matched_id = self.screen_manager.match_screen(look_x, look_y, self.tolerance_x, self.tolerance_y)

            # Lọc phân biệt: Chỉ tính dùng điện thoại khi gọi điện (tay gần tai) hoặc nhìn xuống hướng điện thoại
            is_phone_distracted = has_phone and (is_suspicious_hand or matched_id is None or (look_y is not None and look_y >= 0.08))

            if raw_status == "SLEEPING":
                raw_status = "DISTRACTED"
                detail = "Nhắm mắt ngủ gật (>2.5s)"
            elif is_phone_distracted:
                raw_status = "DISTRACTED"
                detail = "Phát hiện nghe điện thoại" if is_suspicious_hand else "Phát hiện dùng điện thoại"
            else:
                # Kiểm tra chế độ Deep Flow: Cúi đầu đọc tài liệu / ghi chép / gõ phím với mắt mở bình thường
                is_deep_flow = (
                    raw_status != "SLEEPING"
                    and not is_phone_distracted
                    and (0.10 <= (look_y or 0.0) <= 0.45)
                    and abs((look_x or 0.0) - 0.0) <= (self.tolerance_x * 1.35)
                )

                if matched_id is not None:
                    raw_status = "FOCUSED"
                    detail = f"Màn hình {matched_id}"
                    active_screen = matched_id
                elif is_deep_flow:
                    raw_status = "FOCUSED"
                    detail = "Tập trung sâu (Thao tác / Ghi chép)"
                    active_screen = 1
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

        hands = getattr(self.vision, "last_hands", [])
        if not has_phone:
            for h in hands:
                h["holding_phone"] = False

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
            "is_suspicious_hand": is_suspicious_hand,
            "sensitivity": self.sensitivity,
            "baseline_calibrated": self.baseline_calibrated,
            "points": points,
            "mesh_points": mesh_points,
            "face_box": face_box,
            "phone_boxes": phone_boxes,
            "hands": hands
        }

