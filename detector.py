"""
detector.py - Module nhận diện tập trung thông minh (Focus Detector)
Dành cho dự án: pomo-ai (Pomo-do)
"""

import os
import time
import base64
import cv2
from collections import deque, Counter
from screen_manager import ScreenManager
from vision_pipeline import VisionPipeline
from vlm_verifier import VLMPhoneVerifier


class FocusDetector:
    """Class chính điều phối nhận diện tập trung và tính toán trạng thái."""

    def __init__(self, config_filename=None, vision=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.screen_manager = ScreenManager(config_filename)
        self.vision = vision if vision is not None else VisionPipeline(current_dir)
        self.vlm_verifier = VLMPhoneVerifier(current_dir)

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

        # Bộ đệm nhịp khung hình và chống chập chờn (Anti-Flickering & Hysteresis)
        self.frame_count = 0
        self.last_has_phone = False
        self.last_phone_boxes = []
        self.last_hands = []
        self.last_suspicious_hand, self.last_suspicious_roi = False, None
        self.sensitivity, self.phone_confirm_delay = "standard", 1.5
        self.baseline_calibrated, self.baseline_info = False, None
        self.desk_reading_mode = False

        # Bộ đệm thời gian duy trì trạng thái ổn định, chống chớp tắt
        self.last_phone_seen_time = 0.0
        self.phone_start_time = None
        self.phone_active = False
        self.last_valid_phone_boxes = []

        self.last_call_seen_time = 0.0
        self.call_start_time = None
        self.likely_call_start_time = None
        self.call_active = False
        self.last_valid_calling_hand = None
        self.last_valid_call_box = None
        self.blur_background = False

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

    def set_desk_reading_mode(self, enabled: bool):
        """Bật/tắt chế độ đọc sách / tài liệu giấy trên bàn."""
        self.desk_reading_mode = bool(enabled)
        print(f"[pomo-ai] Cập nhật Chế độ đọc sách/tài liệu: {self.desk_reading_mode}")
        return {"desk_reading_mode": self.desk_reading_mode}

    def set_blur_background(self, enabled: bool):
        """Bật/tắt chế độ làm mờ hậu cảnh (Background Blur)."""
        self.blur_background = bool(enabled)
        print(f"[pomo-ai] Cập nhật Chế độ làm mờ nền: {self.blur_background}")
        return {"blur_background": self.blur_background}

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
        img_h, img_w = frame.shape[:2]

        now = time.time()
        self.frame_count += 1
        # Tự động nâng cấp VisionPipeline đang chạy sang yolov8s_fast.onnx (320x416) + đa luồng nếu server hot-reload
        if not hasattr(self.vision, "analyze_frame_parallel") and hasattr(self.vision, "base_dir"):
            try:
                import importlib
                import vision_pipeline as vp_mod
                importlib.reload(vp_mod)
                new_vp = vp_mod.VisionPipeline(self.vision.base_dir)
                self.vision.__dict__.update(new_vp.__dict__)
                self.vision.__class__ = vp_mod.VisionPipeline
            except Exception as e:
                print(f"[pomo-ai] Lỗi nâng cấp VisionPipeline: {e}")
        if hasattr(self.vision, "yolo_input_shape") and getattr(self.vision.phone_detector, "input_shape", None) != self.vision.yolo_input_shape:
            self.vision.phone_detector.input_shape = self.vision.yolo_input_shape

        # 1. Chạy song song MediaPipe Face + MediaPipe Hand + YOLOv8s Fast trên đa luồng C++ (~25-33ms toàn trình)
        if hasattr(self.vision, "analyze_frame_parallel") and (
            getattr(self.vision.phone_detector.detect_phone_onnx, "__func__", None) is not None
        ):
            landmarks, hands = self.vision.analyze_frame_parallel(frame, run_yolo=True)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = self.vision.extract_landmarks(frame, rgb_frame=rgb_frame)
            hands = self.vision.extract_hand_landmarks(frame, rgb_frame=rgb_frame)
        self.last_hands = hands
        points, mesh_points, face_box = None, [], None

        if landmarks is not None:
            self.has_face = True
            self.last_face_time = now
            mesh_points = [(int(p.x * img_w), int(p.y * img_h)) for p in landmarks]
            xs, ys = [pt[0] for pt in mesh_points], [pt[1] for pt in mesh_points]
            pad_x = int((max(xs) - min(xs)) * 0.1)
            pad_y = int((max(ys) - min(ys)) * 0.15)
            face_box = [max(0, min(xs) - pad_x), max(0, min(ys) - pad_y), min(img_w, max(xs) + pad_x), min(img_h, max(ys) + int(pad_y * 0.6))]

            nose, left_eye, right_eye = landmarks[1], landmarks[33], landmarks[263]
            upper_lid, lower_lid = landmarks[159], landmarks[145]

            points = {
                "nose": (int(nose.x * img_w), int(nose.y * img_h)),
                "left_eye": (int(left_eye.x * img_w), int(left_eye.y * img_h)),
                "right_eye": (int(right_eye.x * img_w), int(right_eye.y * img_h))
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

            # Nếu bàn tay đang đặt lên má/mặt che một phần mũi/mắt, tăng hệ số làm mượt EMA để tránh nhảy góc nhìn ảo
            hand_on_face = any(self.vision.phone_detector.is_hand_resting_on_face(h, face_box) for h in hands)
            alpha = 0.85 if hand_on_face else 0.60

            # Làm mượt hướng nhìn bằng Exponential Moving Average (EMA)
            if hasattr(self, "ema_look_x") and self.ema_look_x is not None:
                look_x = round(alpha * self.ema_look_x + (1.0 - alpha) * raw_look_x, 3)
                look_y = round(alpha * self.ema_look_y + (1.0 - alpha) * raw_look_y, 3)
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
            # 2. Xử lý khi MediaPipe FaceMesh bị che khuất tạm thời (đặt tay lên má, chạm trán, chỉnh kính, chống cằm)
            has_person, head_box = self.vision.detect_person_and_head(frame)
            time_since_face = now - self.last_face_time

            # Nếu YOLO xác nhận người vẫn đang ngồi trước bàn, hoặc bàn tay đang tựa lên mặt, hoặc vừa thấy mặt trong 3.0s
            hand_on_last_face = bool(
                self.last_valid_face_box is not None
                and any(self.vision.phone_detector.is_hand_resting_on_face(h, self.last_valid_face_box) for h in hands)
            )
            if has_person or hand_on_last_face or (time_since_face < 3.0 and self.last_valid_face_box is not None):
                self.has_face = True
                self.eye_closed_start_time = None
                look_x = self.last_valid_look_x
                look_y = self.last_valid_look_y
                # Ưu tiên giữ nguyên last_valid_face_box khi người dùng chỉ che mặt bằng tay để tránh lệch vùng má/kính
                if self.last_valid_face_box is not None and (hand_on_last_face or time_since_face < 5.0):
                    face_box = self.last_valid_face_box
                else:
                    face_box = head_box if head_box is not None else self.last_valid_face_box
                points = self.last_valid_points
            else:
                self.has_face = False
                self.eye_closed_start_time = None
                self.ema_look_x = None
                self.ema_look_y = None
                raw_status = "AFK"

        # 2.5. Xử lý làm mờ hậu cảnh (Background Blur) và trích xuất mặt nạ người (Person Mask)
        blurred_frame_data = None
        person_mask = None
        frame_for_detection = frame
        if getattr(self, "blur_background", False):
            frame_blurred, person_mask = self.vision.apply_background_blur(frame)
            if frame_blurred is not None:
                frame_for_detection = frame_blurred
                try:
                    _, buf = cv2.imencode(".jpg", frame_blurred, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    blurred_frame_data = "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")
                except Exception:
                    pass
        elif hasattr(self.vision, "get_person_mask"):
            person_mask = self.vision.get_person_mask(frame)

        # 3. Nhận diện điện thoại thời gian thực (kết hợp person_mask để loại bỏ false positive ở background)
        raw_has_phone, detected_phone_boxes = self.vision.detect_phone(
            frame_for_detection, face_box, look_x, hands, person_mask=person_mask, look_y=look_y
        )
        self.last_has_phone = raw_has_phone
        self.last_phone_boxes = detected_phone_boxes

        phone_boxes = list(detected_phone_boxes)
        is_suspicious_hand = any(h.get("holding_phone", False) for h in hands)

        # 3.1. Phân loại tư thế cuộc gọi từ điện thoại thật được phát hiện (True Object Detection)
        is_calling_frame = any(p.get("is_call", False) for p in phone_boxes)
        calling_level = "CONFIRMED" if is_calling_frame else "NONE"
        call_phone_box = None
        for p in phone_boxes:
            if p.get("is_call", False):
                call_phone_box = p
                self.last_valid_call_box = p
                break

        # 3.1b. Hybrid VLM Verification (Tầng 2 - Tùy chọn khi có Gemini API key)
        vlm_verifier = getattr(self, "vlm_verifier", None)
        vlm_result = None
        if vlm_verifier is not None and vlm_verifier.is_configured:
            need_vlm_verify = (
                not raw_has_phone
                and hands
                and face_box
                and any(self.vision.phone_detector.is_grasping_pose(h) for h in hands)
            )
            if need_vlm_verify:
                hb = hands[0].get("box", face_box)
                vlm_roi = [min(face_box[0], hb[0]), min(face_box[1], hb[1]), max(face_box[2], hb[2]), max(face_box[3], hb[3])]
                vlm_verifier.trigger_async_verify(frame, roi_box=vlm_roi)
            vlm_result = vlm_verifier.get_recent_result()

            if vlm_result is not None and hands:
                vlm_conf = vlm_result.get("confidence_score", 0.0)
                if vlm_result.get("phone_detected") and vlm_conf >= 0.70:
                    hb = hands[0].get("box", face_box)
                    if vlm_result.get("action_type") == "calling":
                        is_calling_frame = True
                        calling_level = "CONFIRMED"
                        self.call_active = True
                        if call_phone_box is None:
                            call_phone_box = {"box": hb, "conf": vlm_conf, "is_call": True, "level": "CONFIRMED"}
                            phone_boxes.append(call_phone_box)
                        hands[0]["is_calling"] = True
                        hands[0]["holding_phone"] = True
                    else:
                        raw_has_phone = True
                        self.phone_active = True
                        is_suspicious_hand = True
                        if not phone_boxes:
                            phone_boxes.append({"box": hb, "conf": vlm_conf, "is_near_user": True, "is_call": False})
                        hands[0]["holding_phone"] = True

        now = time.time()
        resting_hand_on_face = any(self.vision.phone_detector.is_hand_resting_on_face(h, face_box) for h in hands)

        # Nếu bàn tay trống đang đặt trên má/cằm/mặt hoặc đã hạ tay khỏi khung hình và không có điện thoại thật -> xóa ngay bộ đệm
        if (resting_hand_on_face or not hands) and not raw_has_phone and not is_calling_frame:
            self.call_start_time = None
            self.likely_call_start_time = None
            self.call_active = False
            self.phone_start_time = None
            self.phone_active = False
            self.last_valid_phone_boxes = []

        # 3.2. Xác nhận cuộc gọi tức thì khi đã qua bộ lọc YOLO + Hand-Object + Chassis
        if is_calling_frame:
            self.last_call_seen_time = now
            if calling_level == "CONFIRMED":
                if self.call_start_time is None:
                    self.call_start_time = now
                self.call_active = True
        else:
            if (now - getattr(self, "last_call_seen_time", 0.0)) >= 0.25:
                self.call_start_time = None
                self.likely_call_start_time = None
                self.call_active = False

        has_call = bool(self.call_active or is_calling_frame)
        is_calling = has_call

        # 3.3. Hysteresis phản hồi tức thì cho Phone Detection
        if raw_has_phone:
            self.last_phone_seen_time = now
            if self.phone_start_time is None:
                self.phone_start_time = now
        else:
            if (now - getattr(self, "last_phone_seen_time", 0.0)) >= 0.25:
                self.phone_start_time = None

        time_since_phone = now - getattr(self, "last_phone_seen_time", 0.0)

        # Kiểm tra điện thoại có ở gần mặt / đầu hay không
        is_phone_near_face = has_call or (any(
            self.vision.phone_detector.is_phone_near_face(b["box"], face_box)
            for b in phone_boxes
        ) if (phone_boxes and face_box) else False)

        # Nếu đang nghe điện thoại và có box điện thoại thực sự được xác minh (không tạo box ảo nếu tay không)
        if has_call:
            active_box = call_phone_box or getattr(self, "last_valid_call_box", None)
            if active_box and not any(p.get("is_call", False) or p["box"] == active_box["box"] for p in phone_boxes):
                phone_boxes.append(active_box)
                raw_has_phone = True
            elif any(p.get("is_call", False) for p in phone_boxes):
                raw_has_phone = True

        effective_delay = 0.0 if (has_call or is_phone_near_face or is_suspicious_hand) else self.phone_confirm_delay
        if self.phone_start_time is not None and ((now - self.phone_start_time) >= effective_delay):
            self.phone_active = True
        elif time_since_phone >= 0.25 and not has_call:
            self.phone_active = False

        has_phone = self.phone_active or has_call

        # 3.4. Chế độ Gemini VLM (VLM_ONLY_MODE): Gemini API là thẩm quyền xác minh cao cấp
        if vlm_verifier is not None and vlm_verifier.is_configured and getattr(vlm_verifier, "vlm_only_mode", False):
            # Kích hoạt quét Gemini VLM khi có nghi vấn (thấy vật thể hoặc bàn tay ở gần mặt)
            need_vlm = raw_has_phone or need_vlm_verify or bool(hands and face_box) or (self.frame_count % 8 == 0)
            if need_vlm:
                vlm_roi_periodic = (phone_boxes[0]["box"] if phone_boxes else None) or (hands[0]["box"] if hands else None) or face_box
                vlm_verifier.trigger_async_verify(frame, roi_box=vlm_roi_periodic)

            if vlm_result is not None:
                vlm_phone = vlm_result.get("phone_detected", False)
                if vlm_phone:
                    # Gemini xác nhận có điện thoại!
                    has_phone = True
                    vlm_conf = vlm_result.get("confidence_score", 0.95)
                    is_call_action = (vlm_result.get("action_type") == "calling")
                    if is_call_action or is_phone_near_face:
                        has_call = True
                        is_calling = True
                        self.call_active = True
                    else:
                        has_call = False
                        is_calling = False
                        self.call_active = False

                    hb = (phone_boxes[0]["box"] if phone_boxes else None) or (hands[0].get("box", face_box) if hands else face_box) or [0, 0, 100, 100]
                    phone_boxes = [{
                        "box": hb,
                        "conf": vlm_conf,
                        "is_call": is_call_action,
                        "is_near_user": True,
                    }]
                else:
                    # Gemini VLM xác nhận KHÔNG có điện thoại (tay không che mặt/chống cằm) -> VETO toàn bộ!
                    has_phone = False
                    raw_has_phone = False
                    has_call = False
                    is_calling = False
                    self.call_active = False
                    self.phone_active = False
                    self.phone_start_time = None
                    self.call_start_time = None
                    self.likely_call_start_time = None
                    phone_boxes = []
                    self.last_valid_phone_boxes = []
            else:
                # Trong lúc VLM đang phân tích (chưa có kết quả đệm hoặc API rate limit):
                # Nếu YOLO cục bộ đã nhận diện rõ ràng điện thoại thì kích hoạt phát hiện ngay
                if raw_has_phone and len(phone_boxes) > 0:
                    has_phone = True
                    if is_phone_near_face or any(p.get("is_call", False) for p in phone_boxes):
                        has_call = True
                        is_calling = True
                else:
                    has_phone = False
                    has_call = False
                    is_calling = False
                    phone_boxes = []

        # Chống chớp tắt bounding box khi rớt frame ngắn
        if has_phone or has_call:
            if len(phone_boxes) > 0:
                self.last_valid_phone_boxes = list(phone_boxes)
            elif getattr(self, "last_valid_phone_boxes", None):
                phone_boxes = list(self.last_valid_phone_boxes)
        else:
            self.last_valid_phone_boxes = []

        # 4. Phân loại trạng thái
        detail = ""
        is_deep_flow = False
        if self.has_face:
            self.last_look_x, self.last_look_y = look_x, look_y

            matched_id = self.screen_manager.match_screen(look_x, look_y, self.tolerance_x, self.tolerance_y)

            # Lọc phân biệt:
            # - Khi cầm điện thoại trên tay (is_suspicious_hand), giơ gần mặt/tai HOẶC đang nghe điện thoại:
            #   BẮT BUỘC tính là mất tập trung (ghi đè hoàn toàn trạng thái Focused, ngay cả khi Gaze: Screen 1!)
            # - Khi điện thoại để trên bàn/ở xa: chỉ tính khi nhìn xuống (look_y >= 0.06) hoặc nhìn ra ngoài (matched_id is None)
            is_phone_distracted = has_phone and (
                is_phone_near_face or is_suspicious_hand or has_call or matched_id is None or (look_y is not None and look_y >= 0.06)
            )

            if has_call:
                raw_status = "DISTRACTED"
                detail = "Phát hiện đang nghe điện thoại"
            elif raw_status == "SLEEPING":
                raw_status = "DISTRACTED"
                detail = "Nhắm mắt ngủ gật (>2.5s)"
            elif is_phone_distracted:
                raw_status = "DISTRACTED"
                if is_phone_near_face:
                    detail = "Phát hiện dùng điện thoại gần mặt"
                else:
                    detail = "Phát hiện dùng điện thoại"
            else:
                # Kiểm tra chế độ Deep Flow: Cúi đầu đọc tài liệu / ghi chép chỉ khi người dùng BẬT chế độ đọc sách trên bàn
                is_deep_flow = (
                    self.desk_reading_mode
                    and raw_status != "SLEEPING"
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
                    detail = "Tập trung sâu (Đọc sách / Ghi chép)"
                    active_screen = 1
                else:
                    raw_status = "DISTRACTED"
                    detail = "Nhìn ra ngoài vùng màn hình"
        else:
            if has_phone or has_call:
                raw_status = "DISTRACTED"
                if has_call:
                    detail = "Phát hiện đang nghe điện thoại"
                elif is_phone_near_face:
                    detail = "Phát hiện dùng điện thoại gần mặt"
                else:
                    detail = "Phát hiện dùng điện thoại"
            else:
                raw_status = "AFK"
                detail = "Không thấy người trước camera"

        # 5. Làm mượt trạng thái với Deque:
        # Nếu đang nghe điện thoại hoặc dùng điện thoại, ghi đè hoàn toàn FOCUSED và dọn sạch buffer
        if has_phone or has_call:
            smoothed = "DISTRACTED"
            self.history.clear()
        else:
            self.history.append(raw_status)
            smoothed = Counter(self.history).most_common(1)[0][0]

        is_focused = (smoothed == "FOCUSED") and not (has_phone or has_call)
        if is_focused:
            detail = f"Màn hình {active_screen}" if active_screen else "Góc nhìn chuẩn xác"
            event_code = "NORMAL"
        elif has_phone or has_call or "điện thoại" in detail:
            event_code = "PHONE_DETECTED"
            if vlm_result and vlm_result.get("analysis_vietnamese") and (has_phone or has_call):
                detail = vlm_result.get("analysis_vietnamese")
            elif has_call:
                detail = "Phát hiện đang nghe điện thoại"
            elif is_phone_near_face:
                detail = "Phát hiện dùng điện thoại gần mặt"
            else:
                detail = "Phát hiện dùng điện thoại"
        elif not detail:
            detail = "Mất tập trung"
            event_code = "LOOK_AWAY"
        elif not self.has_face or smoothed == "AFK":
            event_code = "AFK"
        elif "ngủ" in detail:
            event_code = "SLEEPING"
        else:
            event_code = "LOOK_AWAY"

        return {
            "status": smoothed,
            "event_code": event_code,
            "detail": detail,
            "is_focused": is_focused,
            "active_screen": active_screen if is_focused else None,
            "total_screens": len(self.screens),
            "desk_reading_mode": self.desk_reading_mode,
            "blur_background": getattr(self, "blur_background", False),
            "blurred_image": blurred_frame_data,
            "look_x": look_x,
            "look_y": look_y,
            "has_face": self.has_face,
            "has_phone": has_phone,
            "raw_has_phone": raw_has_phone,
            "is_phone_near_face": is_phone_near_face,
            "is_calling": is_calling,
            "has_call": has_call,
            "is_suspicious_hand": is_suspicious_hand,
            "sensitivity": self.sensitivity,
            "baseline_calibrated": self.baseline_calibrated,
            "points": points,
            "mesh_points": mesh_points,
            "face_box": face_box,
            "phone_boxes": phone_boxes,
            "hands": hands,
            "vlm_verification": vlm_result,
        }

