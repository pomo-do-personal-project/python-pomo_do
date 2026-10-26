"""
phone_detector.py - Module nhận diện điện thoại và khử nhiễu cho pomo-ai
"""

import cv2
import numpy as np


class PhoneDetector:
    """Xử lý nhận diện điện thoại 2 tầng kết hợp Hand-Object Interaction (HOI)."""

    def __init__(self, net=None, yolo=None):
        self.net = net
        self.yolo = yolo

    def is_hand_attached(self, box, hands):
        """Kiểm tra vật thể có bàn tay/cổ tay tiếp xúc hay không."""
        if not hands:
            return True
        bx1, by1, bx2, by2 = box
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        b_max = max(bx2 - bx1, by2 - by1)
        for h in hands:
            hx, hy = h["center"]
            if ((hx - bcx) ** 2 + (hy - bcy) ** 2) ** 0.5 <= (h["radius"] + b_max * 0.75):
                h["holding_phone"] = True
                return True
        return False

    def is_eyeglasses(self, b, face_box):
        """Loại bỏ false positive do gọng/mắt kính bị nhầm thành điện thoại."""
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

    def detect_phone_onnx(self, img, conf_threshold=0.52, imgsz=256):
        """Chạy inference YOLOv8 qua OpenCV DNN cực kỳ nhẹ RAM và CPU."""
        if self.net is None:
            return []
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (imgsz, imgsz), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()[0].transpose(1, 0)

        phone_scores = preds[:, 71]   # class 67 (cell phone)
        mouse_scores = preds[:, 68]   # class 64 (mouse)
        remote_scores = preds[:, 69]  # class 65 (remote)

        mask = (phone_scores >= conf_threshold) & (phone_scores > mouse_scores * 1.25) & (phone_scores > remote_scores * 1.15)
        if not np.any(mask):
            return []

        valid_preds = preds[mask]
        boxes, scores = [], []
        scale_x, scale_y = w / float(imgsz), h / float(imgsz)

        for p in valid_preds:
            cx, cy, bw, bh = p[0] * scale_x, p[1] * scale_y, p[2] * scale_x, p[3] * scale_y
            w_box, h_box = int(bw), int(bh)
            aspect = max(w_box, h_box) / max(1, min(w_box, h_box))
            if aspect < 1.30 or aspect > 2.85:
                continue
            boxes.append([int(cx - bw / 2.0), int(cy - bh / 2.0), w_box, h_box])
            scores.append(float(p[71]))

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.45)
        results = []
        if len(indices) > 0:
            for idx in indices:
                i = idx if isinstance(idx, (int, np.integer)) else idx[0]
                x, y, bw, bh = boxes[i]
                results.append({
                    "box": [max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)],
                    "conf": round(scores[i], 2)
                })
        return results

    def detect_phone(self, frame, face_box, look_x, suspicious_roi=None, last_hands=None):
        """Nhận diện điện thoại 2 tầng kết hợp Hand-Object Interaction (HOI)."""
        hands = last_hands or []
        for h in hands:
            h["holding_phone"] = False

        phone_boxes = []
        try:
            if self.net is not None:
                candidates = self.detect_phone_onnx(frame, conf_threshold=0.55, imgsz=256)
                for cand in candidates:
                    b, conf = cand["box"], cand["conf"]
                    if (b[2] - b[0]) < 38 or (b[3] - b[1]) < 42 or conf < 0.55:
                        continue
                    if self.is_eyeglasses(b, face_box) or not self.is_hand_attached(b, hands):
                        continue
                    phone_boxes.append(cand)

                if suspicious_roi is not None:
                    rx1, ry1, rx2, ry2 = suspicious_roi
                    if (rx2 - rx1) >= 50 and (ry2 - ry1) >= 50:
                        roi_crop = frame[ry1:ry2, rx1:rx2]
                        roi_cands = self.detect_phone_onnx(roi_crop, conf_threshold=0.38, imgsz=224)
                        for cand in roi_cands:
                            cb, conf = cand["box"], cand["conf"]
                            gbox = [rx1 + cb[0], ry1 + cb[1], rx1 + cb[2], ry1 + cb[3]]
                            if not self.is_eyeglasses(gbox, face_box) and self.is_hand_attached(gbox, hands):
                                if not any(abs(gbox[0] - pb["box"][0]) < 30 and abs(gbox[1] - pb["box"][1]) < 30 for pb in phone_boxes):
                                    phone_boxes.append({"box": gbox, "conf": round(conf, 2)})
            elif self.yolo is not None:
                res = self.yolo(frame, verbose=False, conf=0.55, classes=[67], imgsz=256)
                for r in res:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        b = [int(v) for v in box.xyxy[0].tolist()]
                        bw, bh = b[2] - b[0], b[3] - b[1]
                        if bw < 38 or bh < 42 or conf < 0.55:
                            continue
                        aspect = max(bw, bh) / max(1, min(bw, bh))
                        if aspect < 1.30 or aspect > 2.85 or self.is_eyeglasses(b, face_box) or not self.is_hand_attached(b, hands):
                            continue
                        phone_boxes.append({"box": b, "conf": round(conf, 2)})

                if suspicious_roi is not None:
                    rx1, ry1, rx2, ry2 = suspicious_roi
                    if (rx2 - rx1) >= 50 and (ry2 - ry1) >= 50:
                        roi_res = self.yolo(frame[ry1:ry2, rx1:rx2], verbose=False, conf=0.38, classes=[67], imgsz=224)
                        for r in roi_res:
                            for box in r.boxes:
                                conf = float(box.conf[0])
                                cb = [int(v) for v in box.xyxy[0].tolist()]
                                gbox = [rx1 + cb[0], ry1 + cb[1], rx1 + cb[2], ry1 + cb[3]]
                                if not self.is_eyeglasses(gbox, face_box) and self.is_hand_attached(gbox, hands):
                                    if not any(abs(gbox[0] - pb["box"][0]) < 30 and abs(gbox[1] - pb["box"][1]) < 30 for pb in phone_boxes):
                                        phone_boxes.append({"box": gbox, "conf": round(conf, 2)})
        except Exception as e:
            print(f"[pomo-ai] Cảnh báo phát hiện điện thoại: {e}")

        return len(phone_boxes) > 0, phone_boxes
