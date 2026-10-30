"""
vision_pipeline.py - Khởi tạo mô hình và pipeline xử lý hình ảnh cho pomo-ai
"""

import os
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import mediapipe as mp
from phone_detector import PhoneDetector


class VisionPipeline:
    """Đóng gói việc nạp và suy luận của MediaPipe và YOLO (Tối ưu RAM siêu nhẹ bằng ONNX / OpenCV DNN)."""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.use_legacy_mesh = False
        self.use_tasks_mesh = False
        self._init_mediapipe(base_dir)
        self.yolo = None
        self.net = None
        self.input_shape = (320, 416)
        self.yolo_input_shape = self.input_shape
        self._pool = ThreadPoolExecutor(max_workers=1)

        # 1. Ưu tiên nạp yolov8s_fast.onnx (320x416 native 4:3, nhanh gấp 3.4 lần ~20ms/frame và độ chính xác cao hơn)
        yolo_pref = os.getenv("YOLO_MODEL", "yolov8s_fast.onnx").strip()
        onnx_model_path = os.path.join(base_dir, yolo_pref)
        if not os.path.exists(onnx_model_path):
            onnx_model_path = os.path.join(base_dir, "yolov8s.onnx")
            self.input_shape = (640, 640)
        elif "fast" not in os.path.basename(onnx_model_path):
            self.input_shape = (640, 640)
        if not os.path.exists(onnx_model_path):
            onnx_model_path = os.path.join(base_dir, "yolov8n.onnx")
            self.input_shape = (640, 640)

        if os.path.exists(onnx_model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(onnx_model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                model_name = os.path.basename(onnx_model_path)
                print(f"[pomo-ai] Nap {model_name} (OpenCV DNN {self.input_shape}) thanh cong -> Sieu toc & chinh xac!")
            except Exception as e:
                print(f"[pomo-ai] Canh bao nap ONNX: {e}")

        # 2. Fallback sang PyTorch Ultralytics nếu chưa có file ONNX
        if self.net is None:
            try:
                import torch
                torch.set_num_threads(1)
                from ultralytics import YOLO
                pt_path = os.path.join(base_dir, "yolov8s.pt")
                if not os.path.exists(pt_path):
                    pt_path = os.path.join(base_dir, "yolov8n.pt")
                self.yolo = YOLO(pt_path)
            except Exception as e:
                print(f"[pomo-ai] Cảnh báo nạp YOLO PyTorch: {e}")

        # Khởi tạo bộ nhận diện điện thoại
        self.yolo_input_shape = self.input_shape
        self.phone_detector = PhoneDetector(self.net, self.yolo, input_shape=self.input_shape)

    def analyze_frame_parallel(self, frame, run_yolo=True):
        """Chạy song song YOLO OpenCV DNN trên luồng nền trong khi MediaPipe Face + Hand chạy trên luồng chính (~30ms toàn trình)."""
        f_yolo = None
        if run_yolo and self.net is not None:
            f_yolo = self._pool.submit(self.phone_detector._get_forward_preds, frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = self.extract_landmarks(frame, rgb_frame=rgb_frame)
        hands = self.extract_hand_landmarks(frame, rgb_frame=rgb_frame)
        if f_yolo is not None:
            f_yolo.result()
        return landmarks, hands

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

            # Nạp HandLandmarker nhận diện 21 điểm mốc bàn tay (chấm bi ngón tay và khớp)
            hand_model_file = os.path.join(current_dir, "hand_landmarker.task")
            if not os.path.exists(hand_model_file):
                url_hand = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(url_hand, hand_model_file)

            hand_opts = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_model_file),
                running_mode=vision.RunningMode.IMAGE, num_hands=2,
                min_hand_detection_confidence=0.25, min_hand_presence_confidence=0.25, min_tracking_confidence=0.25
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)

            # Nạp SelfieSegmenter để làm mờ nền (Background Blur) và loại bỏ false positive vật thể nền
            self.segmenter = None
            segmenter_path = os.path.join(current_dir, "selfie_segmenter.tflite")
            if not os.path.exists(segmenter_path):
                try:
                    url_seg = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
                    urllib.request.urlretrieve(url_seg, segmenter_path)
                except Exception as e:
                    print(f"[pomo-ai] Không thể tải selfie_segmenter: {e}")

            if os.path.exists(segmenter_path):
                try:
                    seg_opts = vision.ImageSegmenterOptions(
                        base_options=python.BaseOptions(model_asset_path=segmenter_path),
                        running_mode=vision.RunningMode.IMAGE,
                        output_confidence_masks=True,
                    )
                    self.segmenter = vision.ImageSegmenter.create_from_options(seg_opts)
                except Exception as e:
                    print(f"[pomo-ai] Cảnh báo nạp ImageSegmenter: {e}")

            self.use_tasks_mesh = True
        except Exception as err:
            raise RuntimeError(f"Không thể khởi tạo MediaPipe: {err}")

    def extract_landmarks(self, frame, rgb_frame=None):
        """Trích xuất 468/478 điểm mốc khuôn mặt."""
        if rgb_frame is None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.use_legacy_mesh:
            res = self.face_mesh.process(rgb_frame)
            return res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None
        elif self.use_tasks_mesh:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            res = self.face_landmarker.detect(mp_img)
            return res.face_landmarks[0] if (res.face_landmarks and len(res.face_landmarks) > 0) else None
        return None

    def extract_hand_landmarks(self, frame, rgb_frame=None):
        """Trích xuất 21 điểm mốc (chấm bi) cho mỗi bàn tay và vùng bao quanh."""
        if not hasattr(self, "hand_landmarker") or self.hand_landmarker is None:
            return []
        try:
            if rgb_frame is None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            res = self.hand_landmarker.detect(mp_img)
            h, w = frame.shape[:2]
            hands_data = []
            if res.hand_landmarks:
                for hand_lms in res.hand_landmarks:
                    pts = [(int(p.x * w), int(p.y * h)) for p in hand_lms]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    cx = int(sum(xs) / len(xs))
                    cy = int(sum(ys) / len(ys))
                    radius = int(max(max(xs) - min(xs), max(ys) - min(ys)) / 2.0)
                    hands_data.append({
                        "points": pts,
                        "box": [max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys))],
                        "center": (cx, cy),
                        "radius": radius,
                        "holding_phone": False
                    })
            return hands_data
        except Exception as e:
            print(f"[pomo-ai] Cảnh báo nhận diện bàn tay: {e}")
            return []


    def detect_person_and_head(self, frame, conf_threshold=0.25, imgsz=640):
        """Phát hiện người và ước lượng vị trí đầu khi MediaPipe bị che khuất tạm thời (chạm tay, quay đầu)."""
        if self.net is None and self.yolo is None:
            return False, None

        h, w = frame.shape[:2]
        try:
            if self.net is not None:
                preds, (r, dw, dh) = self.phone_detector._get_forward_preds(frame, imgsz=imgsz)
                if preds is None:
                    return False, None
                person_scores = preds[:, 4]  # Class 0: person
                best_idx = int(np.argmax(person_scores))
                if person_scores[best_idx] > conf_threshold:
                    p = preds[best_idx]
                    cx = (p[0] - dw) / r
                    cy = (p[1] - dh) / r
                    bw = p[2] / r
                    bh = p[3] / r
                    py1 = max(0, int(cy - bh / 2.0))
                    head_w = int(bw * 0.40)
                    hx1 = max(0, int(cx - head_w / 2.0))
                    hx2 = min(w, int(cx + head_w / 2.0))
                    head_bottom = min(h, int(py1 + bh * 0.65))
                    return True, [hx1, py1, hx2, head_bottom]
            elif self.yolo is not None:
                res = self.yolo(frame, verbose=False, conf=conf_threshold, classes=[0], imgsz=640)
                for r in res:
                    for box in r.boxes:
                        b = [int(v) for v in box.xyxy[0].tolist()]
                        bw = b[2] - b[0]
                        bh = b[3] - b[1]
                        cx = (b[0] + b[2]) / 2.0
                        head_w = int(bw * 0.40)
                        hx1 = max(0, int(cx - head_w / 2.0))
                        hx2 = min(w, int(cx + head_w / 2.0))
                        return True, [hx1, b[1], hx2, min(h, int(b[1] + bh * 0.65))]
        except Exception as e:
            print(f"[pomo-ai] Cảnh báo phát hiện người: {e}")

        return False, None

    def get_person_mask(self, frame, rgb_frame=None):
        """Trích xuất mặt nạ người (Foreground mask: 0.0 đến 1.0) qua MediaPipe ImageSegmenter."""
        if getattr(self, "segmenter", None) is None:
            return None
        try:
            if rgb_frame is None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            res = self.segmenter.segment(mp_img)
            if res and res.confidence_masks:
                mask = res.confidence_masks[0].numpy_view()
                if len(mask.shape) == 3 and mask.shape[2] == 1:
                    mask = mask[:, :, 0]
                return mask.astype(np.float32)
        except Exception:
            pass
        return None

    def apply_background_blur(self, frame, ksize=35, rgb_frame=None):
        """Làm mờ toàn bộ hậu cảnh (Background Blur) và giữ sắc nét người dùng."""
        mask = self.get_person_mask(frame, rgb_frame=rgb_frame)
        if mask is None:
            return frame, None

        if ksize % 2 == 0:
            ksize += 1

        blurred_bg = cv2.GaussianBlur(frame, (ksize, ksize), 0)
        # Làm mịn viền alpha để tạo hiệu ứng bokeh tự nhiên, không bị viền nham nhở
        mask_soft = cv2.GaussianBlur(mask, (7, 7), 0)
        mask_3d = mask_soft[:, :, np.newaxis]

        composed = (frame * mask_3d + blurred_bg * (1.0 - mask_3d)).astype(np.uint8)
        return composed, mask

    def detect_phone(self, frame, face_box, look_x=None, hands=None, person_mask=None, look_y=None):
        """Nhận diện điện thoại qua PhoneDetector kết hợp bàn tay và mặt nạ người."""
        return self.phone_detector.detect_phone(frame, face_box, look_x, hands, person_mask=person_mask, look_y=look_y)

