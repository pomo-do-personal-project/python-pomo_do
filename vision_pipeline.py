"""
vision_pipeline.py - Khởi tạo mô hình và pipeline xử lý hình ảnh cho pomo-ai
"""

import os
import cv2
import mediapipe as mp

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print(f"[pomo-ai] Chú ý: Không thể import YOLO ({e}).")


class VisionPipeline:
    """Đóng gói việc nạp và suy luận của MediaPipe và YOLO."""

    def __init__(self, base_dir):
        self.use_legacy_mesh = False
        self.use_tasks_mesh = False
        self._init_mediapipe(base_dir)
        self.yolo, self.pose = None, None
        self.last_wrists, self.last_hands = [], []
        if YOLO is not None:
            try:
                # Nạp duy nhất YOLOv8n nhận diện điện thoại (tiết kiệm ~120MB RAM cho Render 512MB)
                self.yolo = YOLO("yolov8n.pt")
                self.pose = None
                print("[pomo-ai] Nạp YOLOv8n thành công (Tối ưu RAM Render < 350MB)!")
            except Exception as e:
                print(f"[pomo-ai] Cảnh báo nạp YOLO: {e}")

    def _init_mediapipe(self, current_dir):
        """Khởi tạo MediaPipe Face Mesh hỗ trợ mọi phiên bản."""
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
            self.use_legacy_mesh = True
            return

        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import urllib.request

            model_file = os.path.join(current_dir, "face_landmarker.task")
            if not os.path.exists(model_file):
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                urllib.request.urlretrieve(url, model_file)

            opts = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_file),
                running_mode=vision.RunningMode.IMAGE, num_faces=1,
                min_face_detection_confidence=0.25, min_face_presence_confidence=0.25, min_tracking_confidence=0.25
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(opts)
            self.use_tasks_mesh = True
        except Exception as err:
            raise RuntimeError(f"Không thể khởi tạo MediaPipe: {err}")

    def extract_landmarks(self, frame):
        """Trích xuất 468/478 điểm mốc khuôn mặt."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.use_legacy_mesh:
            res = self.face_mesh.process(rgb_frame)
            return res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None
        elif self.use_tasks_mesh:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            res = self.face_landmarker.detect(mp_img)
            return res.face_landmarks[0] if (res.face_landmarks and len(res.face_landmarks) > 0) else None
        return None

    def _extract_hands(self, k, w, h):
        """Trích xuất vị trí bàn tay (trái & phải) bằng phép chiếu cánh tay."""
        hands = []
        if len(k) < 11:
            return hands
        for side, e_idx, w_idx in [("Trai", 7, 9), ("Phai", 8, 10)]:
            elbow, wrist = k[e_idx], k[w_idx]
            if float(wrist[2]) > 0.22:
                wx, wy = float(wrist[0]), float(wrist[1])
                if float(elbow[2]) > 0.22:
                    ex, ey = float(elbow[0]), float(elbow[1])
                    vx, vy = wx - ex, wy - ey
                    hx, hy = int(wx + 0.35 * vx), int(wy + 0.35 * vy)
                    r = max(35, min(int((vx**2 + vy**2)**0.5 * 0.45), 85))
                else:
                    hx, hy, r = int(wx), int(wy), 45
                hands.append({
                    "id": side,
                    "center": [max(0, min(w, hx)), max(0, min(h, hy))],
                    "wrist": [int(wx), int(wy)],
                    "radius": r,
                    "holding_phone": False
                })
        return hands

    def estimate_pose_look(self, frame, w, h):
        """Fallback tư thế đầu và tay bằng YOLOv8-pose khi quay mặt góc lớn."""
        if self.pose is None:
            return False, None, None, None, None

        try:
            res = self.pose(frame, verbose=False, conf=0.25)
            if len(res) == 0 or res[0].keypoints is None or len(res[0].keypoints.data) == 0:
                return False, None, None, None, None

            k = res[0].keypoints.data[0]
            self.last_hands = self._extract_hands(k, w, h)
            nose, l_eye, r_eye, l_ear, r_ear, l_sh, r_sh = k[0], k[1], k[2], k[3], k[4], k[5], k[6]
            if nose[2] <= 0.25 and (l_sh[2] <= 0.3 or r_sh[2] <= 0.3):
                return False, None, None, None, None

            if l_sh[2] > 0.3 and r_sh[2] > 0.3:
                sh_cx = (l_sh[0] + r_sh[0]) / 2.0
                look_x = round(float((nose[0] - sh_cx) / max(float(abs(r_sh[0] - l_sh[0])), 60.0)), 3)
            elif l_ear[2] > 0.25 and r_ear[2] > 0.25:
                ear_cx = (l_ear[0] + r_ear[0]) / 2.0
                look_x = round(float((nose[0] - ear_cx) / (max(float(abs(r_ear[0] - l_ear[0])), 30.0) * 2.0)), 3)
            else:
                look_x = -0.35 if l_ear[2] > 0.25 else 0.35 if r_ear[2] > 0.25 else (-0.25 if nose[0] < (w / 2) else 0.25)

            points = {
                "nose": (int(nose[0]), int(nose[1])),
                "left_eye": (int(l_eye[0]), int(l_eye[1])) if l_eye[2] > 0.2 else (int(l_ear[0]), int(l_ear[1])),
                "right_eye": (int(r_eye[0]), int(r_eye[1])) if r_eye[2] > 0.2 else (int(r_ear[0]), int(r_ear[1]))
            }
            head_pts = [p for p in [nose, l_eye, r_eye, l_ear, r_ear] if p[2] > 0.2]
            face_box = None
            if head_pts:
                kx, ky = [int(p[0]) for p in head_pts], [int(p[1]) for p in head_pts]
                face_box = [max(0, min(kx) - 25), max(0, min(ky) - 35), min(w, max(kx) + 25), min(h, max(ky) + 35)]

            return True, look_x, 0.09, points, face_box
        except Exception as e:
            print(f"[pomo-ai] Cảnh báo YOLO Pose: {e}")
            return False, None, None, None, None

    def check_suspicious_hand_near_ear(self, frame, face_box):
        """Kiểm tra tay gần tai và cập nhật danh sách bàn tay (hands)."""
        if self.pose is None:
            return False, None

        h, w = frame.shape[:2]
        try:
            res = self.pose(frame, verbose=False, conf=0.25, imgsz=320)
            if len(res) == 0 or res[0].keypoints is None or len(res[0].keypoints.data) == 0:
                self.last_wrists, self.last_hands = [], []
                return False, None

            k = res[0].keypoints.data[0]
            self.last_hands = self._extract_hands(k, w, h)
            if face_box is None:
                return False, None

            fx1, fy1, fx2, fy2 = face_box
            fw, fh = fx2 - fx1, fy2 - fy1
            fcx = (fx1 + fx2) / 2.0
            l_ear, r_ear = k[3], k[4]
            self.last_wrists = [k[9], k[10]] if len(k) > 10 else []

            for wrist in self.last_wrists:
                wx, wy, w_conf = float(wrist[0]), float(wrist[1]), float(wrist[2])
                if w_conf < 0.25:
                    continue
                is_height = (fy1 - fh * 0.25) <= wy <= (fy2 + fh * 0.35)
                dist_l = ((wx - float(l_ear[0]))**2 + (wy - float(l_ear[1]))**2)**0.5 if l_ear[2] > 0.25 else 9999
                dist_r = ((wx - float(r_ear[0]))**2 + (wy - float(r_ear[1]))**2)**0.5 if r_ear[2] > 0.25 else 9999
                if (is_height and abs(wx - fcx) <= (fw * 1.3)) or min(dist_l, dist_r) < (fw * 1.1):
                    pad = int(fw * 0.8)
                    return True, [max(0, int(wx - pad)), max(0, int(wy - pad)), min(w, int(wx + pad)), min(h, int(wy + pad))]
        except Exception as e:
            print(f"[pomo-ai] Cảnh báo kiểm tra tay gần tai: {e}")

        return False, None

    def _is_hand_attached(self, box):
        """Nguyên lý 2: Kiểm tra vật thể có bàn tay/cổ tay tiếp xúc."""
        if not self.last_hands:
            return True
        bx1, by1, bx2, by2 = box
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        b_max = max(bx2 - bx1, by2 - by1)
        for h in self.last_hands:
            hx, hy = h["center"]
            if ((hx - bcx)**2 + (hy - bcy)**2)**0.5 <= (h["radius"] + b_max * 0.75):
                h["holding_phone"] = True
                return True
        return False

    def _is_eyeglasses(self, b, face_box):
        """Loại bỏ hoàn toàn false positive do gọng/mắt kính bị nhầm thành điện thoại."""
        if not face_box:
            return False
        bx1, by1, bx2, by2 = b
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        bw, bh = bx2 - bx1, by2 - by1
        fx1, fy1, fx2, fy2 = face_box
        fw, fh = fx2 - fx1, fy2 - fy1
        in_eye_level = (fy1 + fh * 0.05) <= bcy <= (fy1 + fh * 0.60)
        is_centered = abs(bcx - (fx1 + fx2) / 2.0) < (fw * 0.38)
        return in_eye_level and is_centered and (bh < fh * 0.48) and (bw <= fw * 1.15)

    def detect_phone(self, frame, face_box, look_x, suspicious_roi=None):
        """Nhận diện điện thoại 2 tầng kết hợp Hand-Object Interaction (HOI)."""
        if self.yolo is None:
            return False, []

        for h in self.last_hands:
            h["holding_phone"] = False

        phone_boxes = []
        try:
            res = self.yolo(frame, verbose=False, conf=0.35, classes=[67], imgsz=320)
            for r in res:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    b = [int(v) for v in box.xyxy[0].tolist()]
                    if (b[2] - b[0]) < 35 or (b[3] - b[1]) < 35 or conf < 0.38:
                        continue
                    if self._is_eyeglasses(b, face_box) or not self._is_hand_attached(b):
                        continue
                    phone_boxes.append({"box": b, "conf": round(conf, 2)})

            if suspicious_roi is not None:
                rx1, ry1, rx2, ry2 = suspicious_roi
                if (rx2 - rx1) >= 50 and (ry2 - ry1) >= 50:
                    roi_res = self.yolo(frame[ry1:ry2, rx1:rx2], verbose=False, conf=0.24, classes=[67], imgsz=224)
                    for r in roi_res:
                        for box in r.boxes:
                            conf = float(box.conf[0])
                            cb = [int(v) for v in box.xyxy[0].tolist()]
                            gbox = [rx1 + cb[0], ry1 + cb[1], rx1 + cb[2], ry1 + cb[3]]
                            if not self._is_eyeglasses(gbox, face_box) and self._is_hand_attached(gbox):
                                if not any(abs(gbox[0] - pb["box"][0]) < 30 and abs(gbox[1] - pb["box"][1]) < 30 for pb in phone_boxes):
                                    phone_boxes.append({"box": gbox, "conf": round(conf, 2)})
        except Exception as e:
            print(f"[pomo-ai] Cảnh báo YOLO phone: {e}")

        return len(phone_boxes) > 0, phone_boxes
