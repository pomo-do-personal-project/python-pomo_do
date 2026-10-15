"""
vision_pipeline.py - Khởi tạo mô hình và pipeline xử lý hình ảnh cho pomo-ai
"""

import os
import cv2
import mediapipe as mp

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

        self.yolo = None
        self.pose = None
        if YOLO is not None:
            try:
                self.yolo = YOLO("yolov8n.pt")
                self.pose = YOLO("yolov8n-pose.pt")
                print("[pomo-ai] Nạp YOLOv8n và YOLOv8n-pose thành công!")
            except Exception as e:
                print(f"[pomo-ai] Cảnh báo nạp YOLO: {e}")

    def _init_mediapipe(self, current_dir):
        """Khởi tạo MediaPipe Face Mesh hỗ trợ mọi phiên bản."""
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

        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import urllib.request

            model_file = os.path.join(current_dir, "face_landmarker.task")
            if not os.path.exists(model_file):
                print("[pomo-ai] Đang tải mô hình face_landmarker.task...")
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                urllib.request.urlretrieve(url, model_file)

            base_opts = python.BaseOptions(model_asset_path=model_file)
            options = vision.FaceLandmarkerOptions(
                base_options=base_opts,
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

    def estimate_pose_look(self, frame, w, h):
        """Fallback tư thế đầu bằng YOLOv8-pose khi quay mặt góc lớn."""
        if self.pose is None:
            return False, None, None, None, None

        try:
            res = self.pose(frame, verbose=False, conf=0.25)
            if len(res) == 0 or res[0].keypoints is None or len(res[0].keypoints.data) == 0:
                return False, None, None, None, None

            k = res[0].keypoints.data[0]
            nose, l_eye, r_eye, l_ear, r_ear, l_sh, r_sh = k[0], k[1], k[2], k[3], k[4], k[5], k[6]
            if nose[2] <= 0.25 and (l_sh[2] <= 0.3 or r_sh[2] <= 0.3):
                return False, None, None, None, None

            if l_sh[2] > 0.3 and r_sh[2] > 0.3:
                sh_cx = (l_sh[0] + r_sh[0]) / 2.0
                sh_w = max(float(abs(r_sh[0] - l_sh[0])), 60.0)
                look_x = round(float((nose[0] - sh_cx) / sh_w), 3)
            elif l_ear[2] > 0.25 and r_ear[2] > 0.25:
                ear_cx = (l_ear[0] + r_ear[0]) / 2.0
                ear_d = max(float(abs(r_ear[0] - l_ear[0])), 30.0)
                look_x = round(float((nose[0] - ear_cx) / (ear_d * 2.0)), 3)
            elif l_ear[2] > 0.25:
                look_x = -0.35
            elif r_ear[2] > 0.25:
                look_x = 0.35
            else:
                look_x = -0.25 if nose[0] < (w / 2) else 0.25

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

    def detect_phone(self, frame, face_box, look_x):
        """Nhận diện điện thoại bằng YOLOv8n với bộ lọc chống nhiễu."""
        if self.yolo is None:
            return False, []

        phone_boxes = []
        try:
            res = self.yolo(frame, verbose=False, conf=0.55, classes=[67])
            for r in res:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    b = [int(v) for v in box.xyxy[0].tolist()]
                    bx1, by1, bx2, by2 = b
                    if (bx2 - bx1) < 40 or (by2 - by1) < 40 or conf < 0.58:
                        continue
                    if face_box:
                        fx1, fy1, fx2, fy2 = face_box
                        fcx, bcx = (fx1 + fx2) / 2.0, (bx1 + bx2) / 2.0
                        if look_x is not None:
                            if look_x < -0.15 and bcx > fcx and by1 < fy2:
                                continue
                            if look_x > 0.15 and bcx < fcx and by1 < fy2:
                                continue
                        if bx1 >= fx1 and bx2 <= fx2 and by1 >= fy1 and by2 <= (fy1 + (fy2 - fy1) * 0.65):
                            continue
                    phone_boxes.append({"box": b, "conf": round(conf, 2)})
        except Exception as e:
            print(f"[pomo-ai] Cảnh báo YOLO phone: {e}")

        return len(phone_boxes) > 0, phone_boxes
