"""
detector.py - Module nhận diện tập trung thông minh (Focus Detector)
Dành cho dự án: pomo-ai (Pomo-do)
Mục tiêu: Đơn giản, dễ hiểu, dễ mở rộng (Fresher/Intern level)

Các chức năng chính:
1. Nhận diện khuôn mặt, hướng nhìn và nhắm mắt bằng MediaPipe Face Mesh.
2. Hỗ trợ hiệu chuẩn linh hoạt từ 1 đến N màn hình (Dynamic Multi-Screen).
3. Nhận diện dùng điện thoại bằng YOLOv8n (chỉ kích hoạt khi không tập trung để tiết kiệm tài nguyên).
4. Làm mượt trạng thái bằng hàng đợi deque (Smoothing) tránh giật lắc khung hình.
"""

import os
import json
import time
import math
from collections import deque, Counter
import cv2

# Khởi tạo thư viện nhận diện khuôn mặt MediaPipe
import mediapipe as mp

# Khởi tạo thư viện YOLOv8 từ Ultralytics
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print(f"[pomo-ai] Chu y: Khong the import YOLO ({e}). Che do nhan dien dien thoai se tam tat.")


class FocusDetector:
    """
    Class chính xử lý toàn bộ logic nhận diện sự tập trung từ webcam.
    """

    def __init__(self, config_filename="config.json"):
        # Đường dẫn file config.json cùng thư mục với detector.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(current_dir, config_filename)

        # Danh sách màn hình đã cấu hình: [{"id": 1, "x": 0.02, "y": 0.01}, ...]
        self.screens = []
        self.load_screens()

        # Khoảng sai số (tolerance) cho phép khi người dùng nhìn vào màn hình
        # Đã tăng lên một chút để linh hoạt hơn khi cử động đầu tự nhiên
        self.tolerance_x = 0.22
        self.tolerance_y = 0.18

        # Ngưỡng nhắm mắt (khoảng cách mí mắt trên và dưới theo tỉ lệ 0.0 -> 1.0)
        self.eye_closed_threshold = 0.013
        # Thời điểm bắt đầu nhắm mắt liên tục (dùng để phát hiện ngủ gật > 2.5s)
        self.eye_closed_start_time = None

        # Tọa độ nhìn gần nhất của khuôn mặt (dùng khi bấm thêm màn hình mới)
        self.last_look_x = 0.0
        self.last_look_y = 0.0
        self.has_face = False

        # Hàng đợi 8 frame gần nhất để làm mượt trạng thái (Majority Voting)
        self.history = deque(maxlen=8)

        # Khởi tạo MediaPipe Face Mesh (Hỗ trợ tương thích cả phiên bản cũ và mới)
        self._init_mediapipe(current_dir)

        # Khởi tạo mô hình YOLOv8n (phát hiện điện thoại) và YOLOv8n-pose (hỗ trợ góc nghiêng)
        self.yolo = None
        self.pose = None
        if YOLO is not None:
            try:
                print("[pomo-ai] Đang nạp mô hình YOLOv8n và YOLOv8n-pose...")
                self.yolo = YOLO("yolov8n.pt")
                self.pose = YOLO("yolov8n-pose.pt")
                print("[pomo-ai] Nạp mô hình YOLO thành công!")
            except Exception as e:
                print(f"[pomo-ai] Cảnh báo nạp YOLO: {e}")
                self.yolo = None
                self.pose = None

        print("[pomo-ai] Khởi tạo FocusDetector thành công!")

    def _init_mediapipe(self, current_dir):
        """Khởi tạo MediaPipe Face Mesh hỗ trợ mọi phiên bản MediaPipe."""
        self.use_legacy_mesh = False
        self.use_tasks_mesh = False

        # Cách 1: Thử dùng MediaPipe Solutions (phiên bản truyền thống)
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_legacy_mesh = True
            print("[pomo-ai] Sử dụng MediaPipe Solutions FaceMesh.")
            return

        # Cách 2: Sử dụng MediaPipe Tasks Vision API (phiên bản hiện đại mới nhất)
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import urllib.request

            model_file = os.path.join(current_dir, "face_landmarker.task")
            if not os.path.exists(model_file):
                print("[pomo-ai] Đang tải mô hình face_landmarker.task từ Google...")
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                urllib.request.urlretrieve(url, model_file)
                print("[pomo-ai] Đã tải xong face_landmarker.task!")

            base_options = python.BaseOptions(model_asset_path=model_file)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.25,
                min_face_presence_confidence=0.25,
                min_tracking_confidence=0.25
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
            self.use_tasks_mesh = True
            print("[pomo-ai] Sử dụng MediaPipe Tasks FaceLandmarker.")
        except Exception as err:
            print(f"[pomo-ai] Lỗi khởi tạo MediaPipe: {err}")
            raise RuntimeError(f"Không thể khởi tạo MediaPipe: {err}")

    def load_screens(self):
        """Đọc danh sách màn hình từ file config.json."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.screens = json.load(f)
                    print(f"[pomo-ai] Đã tải {len(self.screens)} màn hình từ {self.config_path}")
            except Exception as e:
                print(f"[pomo-ai] Lỗi đọc config.json ({e}), khởi tạo danh sách trống.")
                self.screens = []
        else:
            self.screens = []

    def save_screens(self):
        """Lưu danh sách màn hình vào file config.json."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.screens, f, indent=2, ensure_ascii=False)
            print(f"[pomo-ai] Đã lưu {len(self.screens)} màn hình vào {self.config_path}")
            return True
        except Exception as e:
            print(f"[pomo-ai] Lỗi lưu file config.json: {e}")
            return False

    def add_screen(self, x=None, y=None):
        """
        Thêm một màn hình mới vào danh sách.
        Nếu không truyền x, y sẽ tự động lấy góc nhìn hiện tại (last_look_x, last_look_y).
        Yêu cầu: Phải có khuôn mặt trong khung hình hiện tại mới cho phép lưu.
        """
        if x is None and y is None and not self.has_face:
            print("[pomo-ai] Cảnh báo: Camera không thấy mặt! Vui lòng nhìn rõ vào camera/màn hình rồi thử lại.")
            return None

        target_x = x if x is not None else self.last_look_x
        target_y = y if y is not None else self.last_look_y

        new_screen = {
            "id": len(self.screens) + 1,
            "x": round(float(target_x), 3),
            "y": round(float(target_y), 3)
        }
        self.screens.append(new_screen)
        self.save_screens()
        print(f"[pomo-ai] Đã thêm Màn hình {new_screen['id']} tại tọa độ (x={new_screen['x']}, y={new_screen['y']})")
        return new_screen

    def reset_screens(self):
        """Xóa toàn bộ màn hình đã lưu để hiệu chuẩn lại từ đầu."""
        self.screens = []
        self.save_screens()
        print("[pomo-ai] Đã xóa toàn bộ màn hình. Đặt lại về mặc định.")
        return True

    def _extract_landmarks(self, frame):
        """Trích xuất 468/478 điểm mốc khuôn mặt từ khung hình OpenCV."""
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.use_legacy_mesh:
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0].landmark
            return None

        elif self.use_tasks_mesh:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.face_landmarker.detect(mp_image)
            if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
                return detection_result.face_landmarks[0]
            return None

        return None

    def check_focus(self, frame):
        """
        Hàm chính nhận diện trạng thái tập trung từ 1 khung hình camera:
        Trả về dictionary kết quả:
        {
            "status": "FOCUSED" | "DISTRACTED" | "USING_PHONE" | "SLEEPING" | "AFK",
            "is_focused": True/False,
            "active_screen": 1 (hoặc None),
            "total_screens": len(self.screens),
            "look_x": float,
            "look_y": float
        }
        """
        raw_status = "AFK"
        active_screen = None
        look_x = None
        look_y = None

        # 1. Trích xuất landmark khuôn mặt bằng MediaPipe Face Mesh (Chính xác cao khi nhìn thẳng)
        landmarks = self._extract_landmarks(frame)
        points = None
        mesh_points = []
        face_box = None
        h, w = frame.shape[:2]

        if landmarks is not None:
            # MediaPipe bắt được mặt thành công
            self.has_face = True

            # Toàn bộ các điểm mốc khuôn mặt (dùng vẽ lưới FaceMesh như ảnh 2)
            mesh_points = [(int(p.x * w), int(p.y * h)) for p in landmarks]

            # Tính khung bao khuôn mặt (face bounding box)
            xs = [pt[0] for pt in mesh_points]
            ys = [pt[1] for pt in mesh_points]
            pad_x = int((max(xs) - min(xs)) * 0.1)
            pad_y = int((max(ys) - min(ys)) * 0.15)
            face_box = [
                max(0, min(xs) - pad_x),
                max(0, min(ys) - pad_y),
                min(w, max(xs) + pad_x),
                min(h, max(ys) + int(pad_y * 0.6))
            ]

            # Lấy các điểm mốc quan trọng:
            nose = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            upper_lid = landmarks[159]
            lower_lid = landmarks[145]

            # Lưu tọa độ pixel để vẽ HUD trực quan lên camera
            points = {
                "nose": (int(nose.x * w), int(nose.y * h)),
                "left_eye": (int(left_eye.x * w), int(left_eye.y * h)),
                "right_eye": (int(right_eye.x * w), int(right_eye.y * h))
            }

            # --- KIỂM TRA NHẮM MẮT (NGỦ GẬT) ---
            eyelid_dist = abs(upper_lid.y - lower_lid.y)
            current_time = time.time()

            if eyelid_dist < self.eye_closed_threshold:
                if self.eye_closed_start_time is None:
                    self.eye_closed_start_time = current_time
                elif (current_time - self.eye_closed_start_time) >= 2.5:
                    raw_status = "SLEEPING"
            else:
                self.eye_closed_start_time = None

            # --- TÍNH TOÁN HƯỚNG NHÌN (LOOK_X, LOOK_Y) TỪ MEDIAPIPE ---
            eye_center_x = (left_eye.x + right_eye.x) / 2.0
            eye_center_y = (left_eye.y + right_eye.y) / 2.0
            look_x = round(nose.x - eye_center_x, 3)
            look_y = round(nose.y - eye_center_y, 3)

        else:
            # 2. Cơ chế dự phòng (Fallback): Khi người dùng quay đầu sang màn hình phụ (góc nghiêng > 45 độ),
            # MediaPipe thường mất dấu, ta dùng YOLOv8-Pose để bắt hướng quay đầu của cơ thể/khuôn mặt
            pose_found = False
            if self.pose is not None:
                try:
                    pose_res = self.pose(frame, verbose=False, conf=0.25)
                    if len(pose_res) > 0 and pose_res[0].keypoints is not None:
                        kpts = pose_res[0].keypoints.data
                        if len(kpts) > 0:
                            k = kpts[0]  # Keypoints của người đầu tiên [17, 3]
                            nose_k = k[0]
                            l_eye_k = k[1]
                            r_eye_k = k[2]
                            l_ear_k = k[3]
                            r_ear_k = k[4]
                            l_sh_k = k[5]
                            r_sh_k = k[6]

                            # Kiểm tra nếu nhận diện được mũi hoặc tai/vai
                            if nose_k[2] > 0.25 or (l_sh_k[2] > 0.3 and r_sh_k[2] > 0.3):
                                pose_found = True
                                self.has_face = True
                                self.eye_closed_start_time = None

                                # Tính hướng nhìn dựa trên tương quan mũi với vai/tai
                                if l_sh_k[2] > 0.3 and r_sh_k[2] > 0.3:
                                    sh_center_x = (l_sh_k[0] + r_sh_k[0]) / 2.0
                                    sh_width = max(float(abs(r_sh_k[0] - l_sh_k[0])), 60.0)
                                    # Mũi lệch so với trung tâm vai
                                    look_x = round(float((nose_k[0] - sh_center_x) / sh_width), 3)
                                    look_y = 0.09
                                elif l_ear_k[2] > 0.25 and r_ear_k[2] > 0.25:
                                    ear_center_x = (l_ear_k[0] + r_ear_k[0]) / 2.0
                                    ear_dist = max(float(abs(r_ear_k[0] - l_ear_k[0])), 30.0)
                                    look_x = round(float((nose_k[0] - ear_center_x) / (ear_dist * 2.0)), 3)
                                    look_y = 0.09
                                elif l_ear_k[2] > 0.25:  # Quay mặt sang trái (hướng màn hình 2)
                                    look_x = -0.35
                                    look_y = 0.09
                                elif r_ear_k[2] > 0.25:  # Quay mặt sang phải (hướng màn hình 3)
                                    look_x = 0.35
                                    look_y = 0.09
                                else:
                                    look_x = -0.25 if nose_k[0] < (w / 2) else 0.25
                                    look_y = 0.09

                                points = {
                                    "nose": (int(nose_k[0]), int(nose_k[1])),
                                    "left_eye": (int(l_eye_k[0]), int(l_eye_k[1])) if l_eye_k[2] > 0.2 else (int(l_ear_k[0]), int(l_ear_k[1])),
                                    "right_eye": (int(r_eye_k[0]), int(r_eye_k[1])) if r_eye_k[2] > 0.2 else (int(r_ear_k[0]), int(r_ear_k[1]))
                                }

                                # Tạo face_box từ các keypoint vùng đầu nếu chưa có từ MediaPipe
                                if face_box is None:
                                    head_pts = [nose_k, l_eye_k, r_eye_k, l_ear_k, r_ear_k]
                                    v_pts = [p for p in head_pts if p[2] > 0.2]
                                    if v_pts:
                                        kx = [int(p[0]) for p in v_pts]
                                        ky = [int(p[1]) for p in v_pts]
                                        face_box = [
                                            max(0, min(kx) - 25),
                                            max(0, min(ky) - 35),
                                            min(w, max(kx) + 25),
                                            min(h, max(ky) + 35)
                                        ]
                except Exception as e:
                    print(f"[pomo-ai] Cảnh báo YOLO Pose: {e}")

            if not pose_found:
                # Không thấy cả mặt lẫn tư thế người -> AFK (Rời vị trí)
                self.has_face = False
                self.eye_closed_start_time = None
                raw_status = "AFK"

        # 3. Quét nhận diện điện thoại bằng YOLOv8n với bộ lọc chống nhận diện nhầm ghế / kính
        phone_boxes = []
        has_phone = False
        if self.yolo is not None:
            try:
                # Tăng ngưỡng conf lên 0.55 để tránh vật thể lạ (ghế, kính) bị nhận diện nhầm
                yolo_res = self.yolo(frame, verbose=False, conf=0.55, classes=[67])
                for r in yolo_res:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        coords = [int(v) for v in box.xyxy[0].tolist()]
                        bx1, by1, bx2, by2 = coords
                        bw = bx2 - bx1
                        bh = by2 - by1

                        # Kích thước tối thiểu cho điện thoại cầm tay
                        if bw < 40 or bh < 40 or conf < 0.58:
                            continue

                        # Kiểm tra vị trí để loại trừ ghế tựa đầu & kính mắt
                        if face_box:
                            fx1, fy1, fx2, fy2 = face_box
                            fcx = (fx1 + fx2) / 2.0
                            bcx = (bx1 + bx2) / 2.0

                            # Loại trừ ghế tựa đầu (Headrest): nằm phía sau hướng quay đầu và ở ngang tầm đầu
                            if look_x is not None:
                                # Quay mặt sang trái -> gáy/ghế ở bên phải
                                if look_x < -0.15 and bcx > fcx and by1 < fy2:
                                    continue
                                # Quay mặt sang phải -> gáy/ghế ở bên trái
                                if look_x > 0.15 and bcx < fcx and by1 < fy2:
                                    continue

                            # Loại trừ kính mắt: nếu khung nằm lọt thỏm ngay vùng mắt/mũi
                            if bx1 >= fx1 and bx2 <= fx2 and by1 >= fy1 and by2 <= (fy1 + (fy2 - fy1) * 0.65):
                                continue

                        phone_boxes.append({"box": coords, "conf": round(conf, 2)})
                        has_phone = True
            except Exception as e:
                print(f"[pomo-ai] Cảnh báo YOLO phone: {e}")

        # 4. Phân loại trạng thái nếu người dùng đang có mặt
        detail = ""
        if self.has_face:
            self.last_look_x = look_x
            self.last_look_y = look_y

            # Ưu tiên 1: Đang ngủ gật
            if raw_status == "SLEEPING":
                raw_status = "DISTRACTED"
                detail = "Nhắm mắt ngủ gật (>2.5s)"
                active_screen = None

            # Ưu tiên 2: Cầm hoặc sử dụng điện thoại thật sự
            elif has_phone:
                raw_status = "DISTRACTED"
                detail = "Phát hiện dùng điện thoại"
                active_screen = None

            # Ưu tiên 3: Kiểm tra hướng nhìn vào các màn hình đã lưu
            else:
                screens_to_check = self.screens if len(self.screens) > 0 else [{"id": 1, "x": 0.0, "y": 0.0}]
                matched_screen_id = None
                min_distance = float("inf")

                for s in screens_to_check:
                    dx = abs(look_x - s["x"])
                    dy = abs(look_y - s["y"])

                    if dx <= self.tolerance_x and dy <= self.tolerance_y:
                        dist = math.hypot(dx, dy)
                        if dist < min_distance:
                            min_distance = dist
                            matched_screen_id = s["id"]

                if matched_screen_id is not None:
                    raw_status = "FOCUSED"
                    detail = f"Màn hình {matched_screen_id}"
                    active_screen = matched_screen_id
                else:
                    raw_status = "DISTRACTED"
                    detail = "Nhìn ra ngoài vùng màn hình"
                    active_screen = None
        else:
            raw_status = "AFK"
            detail = "Không thấy người trước camera"

        # --- LÀM MƯỢT TRẠNG THÁI VỚI DEQUE (SMOOTHING) ---
        self.history.append(raw_status)
        smoothed_status = Counter(self.history).most_common(1)[0][0]

        is_focused = (smoothed_status == "FOCUSED")
        if is_focused:
            detail = f"Màn hình {active_screen}" if active_screen else "Góc nhìn chuẩn xác"
        elif not detail:
            detail = "Mất tập trung"

        return {
            "status": smoothed_status,
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
