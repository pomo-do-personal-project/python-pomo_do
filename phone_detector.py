"""
phone_detector.py - Module nhận diện điện thoại và khử nhiễu cho pomo-ai
Tham khảo các best practices từ:
- cell-phone-detection-using-yolov3 (trandinhson3086)
- Student-Online-Exam-AntiCheat-Tool (Zorrat)
- judgy_reachy_no_phone (yaseminozkut)
- Action_Recognition_Surveillance (Delicacies)
- distraction_detection (johannesharmse)

Nguyên tắc cốt lõi:
1. Phát hiện điện thoại thật (Ground Truth) bằng YOLOv8s ONNX siêu nhẹ và chính xác.
2. TUYỆT ĐỐI KHÔNG TỰ SINH BOUNDING BOX ẢO từ màu da / bóng đổ / góc ngón tay
   (khử triệt để lỗi tay không chạm mặt, chống cằm bị báo PHONE CALL 95%).
3. Xác minh bàn tay có cầm điện thoại hay không (Hand-Object Association):
   - Nếu điện thoại nằm ở vùng tai/má -> Tư thế nghe điện thoại (CALL).
   - Nếu điện thoại gắn với bàn tay ở trước ngực/bàn -> Tư thế cầm/lướt điện thoại (HOLDING).
   - Nếu vật thể ở xa không có tay cầm và không gần người -> Bỏ qua.
4. Caching 1 forward pass duy nhất cho cả Person và Phone -> Giảm thiểu CPU và khử lag hoàn toàn.
"""

import cv2
import numpy as np


class PhoneDetector:
    """Xử lý nhận diện điện thoại thời gian thực bằng YOLO kết hợp quan hệ không gian Bàn tay - Khuôn mặt."""

    def __init__(self, net=None, yolo=None, input_shape=(320, 416)):
        self.net = net
        self.yolo = yolo
        self.input_shape = input_shape
        self._cached_img_id = None
        self._cached_preds = None
        self._cached_meta = None

    @staticmethod
    def _finger_extensions(pts):
        """Tính tỷ lệ duỗi thẳng (st / seg) của 4 ngón tay chính: trỏ, giữa, áp út, út."""
        if not pts or len(pts) != 21:
            return []
        fingers = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
        extensions = []
        for m, p, d, t in fingers:
            mcp = np.array(pts[m], dtype=np.float32)
            pip = np.array(pts[p], dtype=np.float32)
            dip = np.array(pts[d], dtype=np.float32)
            tip = np.array(pts[t], dtype=np.float32)
            seg = np.linalg.norm(pip - mcp) + np.linalg.norm(dip - pip) + np.linalg.norm(tip - dip)
            st = np.linalg.norm(tip - mcp)
            if seg > 1e-5:
                extensions.append(float(st / seg))
        return extensions

    @staticmethod
    def _hand_grip_metrics(pts):
        """Phân tích hình học khớp bàn tay để phân biệt C-grip và các tư thế cầm nắm."""
        if not pts or len(pts) != 21:
            return {
                "avg_mcp_angle": 0.0,
                "avg_pip_curl": 0.0,
                "curled_pip_count": 0,
                "upward_fingers": 4,
                "d_thumb_idx": 0.0,
                "is_c_grip": False,
                "is_angled_ear_grip": False,
                "palm_h": 20.0,
            }
        P = np.array(pts, dtype=np.float32)
        mcp_angles = []
        for m, t in [(5, 8), (9, 12), (13, 16), (17, 20)]:
            v1 = P[m] - P[0]
            v2 = P[t] - P[m]
            c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            mcp_angles.append(float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
        avg_mcp_angle = sum(mcp_angles) / 4.0

        pip_curls = []
        for m, p, t in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:
            v1 = P[p] - P[m]
            v2 = P[t] - P[p]
            c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            pip_curls.append(float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
        avg_pip_curl = sum(pip_curls) / 4.0
        curled_pip_count = sum(1 for a in pip_curls if a >= 15.0)

        finger_rises = [float(P[p][1] - P[t][1]) for p, t in [(6, 8), (10, 12), (14, 16), (18, 20)]]
        upward_fingers = sum(1 for r in finger_rises if r > 15.0)

        tips_x = [float(P[8][0]), float(P[12][0]), float(P[16][0]), float(P[20][0])]
        fingers_x_span = max(tips_x) - min(tips_x)
        palm_h = max(20.0, float(max(P[:, 1]) - min(P[:, 1])))
        thumb_gap_x = abs(float(P[4][0]) - (sum(tips_x) / 4.0))
        d_thumb_idx = min(float(np.linalg.norm(P[4] - P[i])) for i in (5, 6, 7, 8))

        is_c_grip = bool(
            (upward_fingers >= 3)
            and (curled_pip_count >= 2)
            and (avg_pip_curl >= 14.0)
            and (d_thumb_idx >= max(24.0, palm_h * 0.18))
            and (fingers_x_span < palm_h * 0.22)
            and (thumb_gap_x >= 28.0)
        )
        is_angled_ear_grip = bool(
            (avg_mcp_angle >= 25.0)
            and (upward_fingers <= 1)
            and (curled_pip_count >= 2)
            and (avg_pip_curl >= 16.0)
        )
        return {
            "avg_mcp_angle": avg_mcp_angle,
            "avg_pip_curl": avg_pip_curl,
            "curled_pip_count": curled_pip_count,
            "upward_fingers": upward_fingers,
            "fingers_x_span": fingers_x_span,
            "thumb_gap_x": thumb_gap_x,
            "d_thumb_idx": d_thumb_idx,
            "palm_h": palm_h,
            "is_c_grip": is_c_grip,
            "is_angled_ear_grip": is_angled_ear_grip,
        }

    @staticmethod
    def is_flat_open_hand(hand):
        """Kiểm tra bàn tay có xòe phẳng / duỗi thẳng ngón tay (empty hand) hay không."""
        pts = hand.get("points", []) if isinstance(hand, dict) else []
        if not pts or len(pts) != 21:
            return False
        grip = PhoneDetector._hand_grip_metrics(pts)
        if grip["is_c_grip"] or grip["is_angled_ear_grip"]:
            return False
        extensions = PhoneDetector._finger_extensions(pts)
        if not extensions:
            return False
        if grip["curled_pip_count"] <= 1 and grip["avg_pip_curl"] < 15.0:
            if sum(1 for e in extensions if e > 0.75) >= 2:
                return True
        palm_w = float(np.linalg.norm(np.array(pts[5]) - np.array(pts[17])))
        if palm_w > 8.0:
            finger_span = float(np.linalg.norm(np.array(pts[8]) - np.array(pts[20])))
            if finger_span >= palm_w * 1.25 and (sum(extensions) / len(extensions)) > 0.65:
                return True
        return False

    @staticmethod
    def is_calling_pose(hand, face_box, look_x=0.0, head_tilt_deg=0.0, allow_flat=False):
        """Kiểm tra tư thế đưa tay lên áp sát tai."""
        if not hand or not face_box:
            return False
        if not allow_flat and PhoneDetector.is_flat_open_hand(hand):
            return False
        hx1, hy1, hx2, hy2 = hand.get("box", hand) if isinstance(hand, dict) else hand
        fx1, fy1, fx2, fy2 = face_box
        fw, fh = max(1, fx2 - fx1), max(1, fy2 - fy1)
        hw, hh = max(1, hx2 - hx1), max(1, hy2 - hy1)
        if hh < fh * 0.20 or hw > (hh * 0.95):
            return False
        reaches_ear = (hy1 <= fy2 - fh * 0.10) and (hy1 >= fy1 - fh * 0.45)
        wrist_down = (hy2 >= fy1 + fh * 0.35)
        return reaches_ear and wrist_down

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
        """Biến đổi ảnh sang kích thước vuông với padding giữ nguyên tỉ lệ không bị méo."""
        shape = im.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2.0
        dh /= 2.0
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

    def _get_forward_preds(self, img, imgsz=None):
        """Chạy forward YOLO duy nhất 1 lần cho 1 khung hình, tái sử dụng chung cho cả Person và Phone."""
        if getattr(self, "_cached_img_id", None) == id(img) and getattr(self, "_cached_preds", None) is not None:
            return self._cached_preds, self._cached_meta

        if self.net is None:
            return None, (1.0, 0.0, 0.0)

        target_shape = getattr(self, "input_shape", (320, 416))
        if isinstance(target_shape, int):
            target_shape = (target_shape, target_shape)

        lb_img, r, (dw, dh) = self.letterbox(img, target_shape)
        blob = cv2.dnn.blobFromImage(lb_img, 1.0 / 255.0, (target_shape[1], target_shape[0]), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()[0].transpose(1, 0)
        self._cached_img_id = id(img)
        self._cached_preds = preds
        self._cached_meta = (r, dw, dh)
        return preds, (r, dw, dh)

    def detect_phone_onnx(self, img, conf_threshold=0.25, imgsz=640):
        """Chạy inference YOLOv8 qua OpenCV DNN nhẹ RAM và CPU."""
        if self.net is None:
            return []
        h, w = img.shape[:2]
        preds, (r, dw, dh) = self._get_forward_preds(img, imgsz=imgsz)
        if preds is None:
            return []

        phone_scores = preds[:, 71]   # class 67 (cell phone)
        mouse_scores = preds[:, 68]   # class 64 (mouse)
        remote_scores = preds[:, 69]  # class 65 (remote)

        mask = (phone_scores >= conf_threshold) & (phone_scores > mouse_scores) & (phone_scores > remote_scores)
        if not np.any(mask):
            return []

        valid_preds = preds[mask]
        valid_scores = phone_scores[mask]
        boxes, scores = [], []

        for p, sc in zip(valid_preds, valid_scores):
            cx = (p[0] - dw) / r
            cy = (p[1] - dh) / r
            bw = p[2] / r
            bh = p[3] / r
            w_box, h_box = int(bw), int(bh)
            aspect = max(w_box, h_box) / max(1, min(w_box, h_box))
            if aspect < 1.10 or aspect > 3.50:
                continue
            boxes.append([int(cx - bw / 2.0), int(cy - bh / 2.0), w_box, h_box])
            scores.append(float(sc))

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
                    "conf": round(scores[i], 2),
                    "is_call": False,
                    "is_near_user": False,
                })
        return results

    def is_face_or_eyeglasses(self, b, face_box):
        """Loại bỏ false positive khi YOLO nhận nhầm cả khuôn mặt hoặc gọng kính ngang là điện thoại."""
        if not face_box:
            return False
        bx1, by1, bx2, by2 = b
        fx1, fy1, fx2, fy2 = face_box
        bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
        fw, fh = max(1, fx2 - fx1), max(1, fy2 - fy1)
        b_area = bw * bh
        f_area = fw * fh

        # 1. Khử trường hợp YOLO nhầm cả khuôn mặt là điện thoại (IoU > 0.65)
        inter_x1 = max(bx1, fx1)
        inter_y1 = max(by1, fy1)
        inter_x2 = min(bx2, fx2)
        inter_y2 = min(by2, fy2)
        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            iou = inter_area / max(1, (b_area + f_area - inter_area))
            if iou > 0.65 or (inter_area / f_area > 0.80 and inter_area / b_area > 0.80):
                return True

        # 2. Khử gọng kính cận: Thanh ngang hẹp (bh <= fh * 0.30) nằm ngang giữa hai mắt
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        in_eye_y = (fy1 + fh * 0.15) <= bcy <= (fy1 + fh * 0.55)
        in_face_x = (fx1 + fw * 0.10) <= bcx <= (fx2 - fw * 0.10)
        if in_eye_y and in_face_x and (bh <= fh * 0.30) and (bw <= fw * 0.85):
            return True

        # 3. Các mảnh nhỏ rác do bóng đổ / gọng kính trên má/mặt
        if max(bw, bh) < (fh * 0.40) and b_area < (f_area * 0.20):
            return True

        return False

    def is_eyeglasses(self, b, face_box):
        """Alias tương thích ngược."""
        return self.is_face_or_eyeglasses(b, face_box)

    @staticmethod
    def is_phone_near_face(phone_box, face_box):
        """Kiểm tra điện thoại có áp sát trực tiếp vào mặt/tai hay không (Tư thế gọi điện)."""
        if not phone_box or not face_box:
            return False
        px1, py1, px2, py2 = phone_box
        fx1, fy1, fx2, fy2 = face_box
        fw, fh = max(1, fx2 - fx1), max(1, fy2 - fy1)

        dx = max(0, max(fx1 - px2, px1 - fx2))
        dy = max(0, max(fy1 - py2, py1 - fy2))
        dist = (dx**2 + dy**2) ** 0.5
        pcy = (py1 + py2) / 2.0
        return dist <= (fw * 0.50) and pcy <= (fy2 + fh * 0.40)

    @staticmethod
    def is_near_face_vicinity(b, face_box):
        """Kiểm tra vật thể có nằm trong vùng lân cận với khuôn mặt/người hay không."""
        if not b or not face_box:
            return False
        bx1, by1, bx2, by2 = b
        fx1, fy1, fx2, fy2 = face_box
        fw, fh = max(1, fx2 - fx1), max(1, fy2 - fy1)
        dx = max(0, max(fx1 - bx2, bx1 - fx2))
        dy = max(0, max(fy1 - by2, by1 - fy2))
        dist = (dx**2 + dy**2) ** 0.5
        return dist <= (fw * 0.85) and (by1 <= fy2 + fh * 0.65)

    def is_hand_attached(self, box, hands, face_box=None, allow_resting=False):
        """Kiểm tra xem khung điện thoại có liên kết vật lý với bàn tay hay không."""
        if not hands:
            return False
        bx1, by1, bx2, by2 = box
        bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
        b_area = bw * bh
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0

        for h in hands:
            # Nếu bàn tay xòe phẳng, cử chỉ chỉ ngón, hoặc đang đặt nghỉ trên má/mặt, nó không thể cầm vật thể
            if not allow_resting:
                if self.is_flat_open_hand(h) or self.is_hand_resting_on_face(h, face_box) or self.is_non_holding_gesture(h):
                    continue

            hx1, hy1, hx2, hy2 = h.get("box", h) if isinstance(h, dict) else h
            hw, hh = max(1, hx2 - hx1), max(1, hy2 - hy1)
            h_area = hw * hh
            hcx = (hx1 + hx2) / 2.0
            hcy = (hy1 + hy2) / 2.0

            # Giao cắt trực tiếp
            inter_w = max(0, min(bx2, hx2) - max(bx1, hx1))
            inter_h = max(0, min(by2, hy2) - max(by1, hy1))
            inter_area = inter_w * inter_h

            if inter_area > 0:
                overlap_ratio = inter_area / max(1, min(b_area, h_area))
                if overlap_ratio >= 0.15:
                    if isinstance(h, dict):
                        h["holding_phone"] = True
                    return True

            dist_centers = ((bcx - hcx)**2 + (bcy - hcy)**2) ** 0.5
            if dist_centers <= max(hw, hh) * 0.95:
                if isinstance(h, dict):
                    h["holding_phone"] = True
                return True

            pts = h.get("points", []) if isinstance(h, dict) else []
            if pts and len(pts) == 21:
                pts_inside = sum(1 for pt in pts if (bx1 - 10) <= pt[0] <= (bx2 + 10) and (by1 - 10) <= pt[1] <= (by2 + 10))
                if pts_inside >= 2:
                    if isinstance(h, dict):
                        h["holding_phone"] = True
                    return True

        return False

    @staticmethod
    def is_hand_resting_on_face(hand, face_box):
        """Kiểm tra bàn tay có đang tựa/chống cằm hoặc chạm vào mặt."""
        if not hand or not face_box:
            return False
        pts = hand.get("points", []) if isinstance(hand, dict) else []
        if pts and len(pts) == 21:
            grip = PhoneDetector._hand_grip_metrics(pts)
            if grip["is_c_grip"] or grip["is_angled_ear_grip"]:
                return False
        hx1, hy1, hx2, hy2 = hand.get("box", hand) if isinstance(hand, dict) else hand
        fx1, fy1, fx2, fy2 = face_box
        fw, fh = max(1, fx2 - fx1), max(1, fy2 - fy1)
        overlaps_face = (hx2 >= fx1 - fw * 0.10) and (hx1 <= fx2 + fw * 0.10) and (hy2 >= fy1) and (hy1 <= fy2 + fh * 0.20)
        return overlaps_face

    @staticmethod
    def is_non_holding_gesture(hand):
        """Kiểm tra cử chỉ không cầm đồ vật (chữ V, giơ 1 ngón)."""
        pts = hand.get("points", []) if isinstance(hand, dict) else []
        if not pts or len(pts) != 21:
            return False
        palm_h = max(20.0, float(max(p[1] for p in pts) - min(p[1] for p in pts)))
        other_tips_min_y = min(pts[12][1], pts[16][1], pts[20][1])
        return (other_tips_min_y - pts[8][1]) >= (palm_h * 0.25)

    @staticmethod
    def detect_phone_chassis_features(crop):
        """Kiểm tra đặc điểm thân máy smartphone."""
        if crop is None or crop.shape[0] < 10 or crop.shape[1] < 10:
            return False, 0.0, 0.0
        ch, cw = crop.shape[:2]
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        ycc = ycrcb.astype(np.int16)
        warm_skin = (ycc[..., 1] >= 131) & ((ycc[..., 1] - ycc[..., 2]) >= 8)
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127])) | (warm_skin.astype(np.uint8) * 255)
        non_skin_ratio = 1.0 - (float(np.count_nonzero(skin_mask)) / max(1, skin_mask.size))
        top_h = max(1, int(ch * 0.32))
        top_part = ycc[:top_h, :]
        bot_part = ycc[top_h:, :]
        top_bright_wall = (top_part[..., 0] >= 145) & (np.abs(top_part[..., 1] - 128) <= 8) & (np.abs(top_part[..., 2] - 128) <= 8)
        bot_dark_cloth = bot_part[..., 0] < 65
        if float(np.mean(top_bright_wall)) >= 0.20 and float(np.mean(bot_dark_cloth)) >= 0.28:
            return False, 0.0, 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 45, 135)
        edge_ratio = float(np.count_nonzero(edges)) / max(1, edges.size)
        has_chassis = (non_skin_ratio >= 0.38 and edge_ratio >= 0.05) or (non_skin_ratio >= 0.50 and edge_ratio >= 0.03)
        return has_chassis, non_skin_ratio, edge_ratio

    @staticmethod
    def is_grasping_pose(hand):
        """Kiểm tra bàn tay có đang ở tư thế cầm nắm hay không."""
        pts = hand.get("points", []) if isinstance(hand, dict) else []
        if not pts or len(pts) != 21:
            return False
        thumb_tip = np.array(pts[4])
        index_tip = np.array(pts[8])
        gap = float(np.linalg.norm(thumb_tip - index_tip))
        return gap >= 18.0

    def score_ear_call_evidence(self, frame, hand, face_box, look_x=0.0, head_tilt_deg=0.0):
        """Hàm tương thích ngược: Đảm bảo không tạo box ảo khi không có điện thoại thật."""
        return {"score": 0.0, "level": "NONE", "box": None, "is_call": False}

    def detect_grip_phone_slab(self, frame, hand, face_box=None, look_x=0.0, head_tilt_deg=0.0):
        """Phát hiện điện thoại gắn với bàn tay khi cầm máy (C-grip / Hand holding)."""
        if not hand or frame is None:
            return None
        cands = self.detect_phone_onnx(frame, conf_threshold=0.20)
        for c in cands:
            if self.is_hand_attached(c["box"], [hand], face_box, allow_resting=True):
                return c
        return None

    def has_held_object(self, frame, hand, face_box=None, detected_phone_boxes=None, was_holding=False):
        """Kiểm tra bàn tay có gắn với điện thoại thực tế không."""
        if not hand or not detected_phone_boxes:
            return False
        for b in detected_phone_boxes:
            if self.is_hand_attached(b["box"], [hand], face_box, allow_resting=True):
                return True
        return False

    def detect_phone(self, frame, face_box, look_x=None, hands=None, person_mask=None, look_y=None):
        """Nhận diện điện thoại chuẩn xác, thời gian thực, không giật lag.
        Tuân thủ mô hình của AntiCheat / Action Surveillance:
        1. YOLOv8s là nguồn chân lý (Ground Truth) phát hiện vật thể cell phone.
        2. Liên kết không gian với Bàn tay và Khuôn mặt để xác định hành động (Calling / Browsing).
        3. TUYỆT ĐỐI KHÔNG TỰ SINH BOX ẢO khi tay không chạm mặt.
        4. LOẠI BỎ TRIỆT ĐỂ VẬT THỂ Ở BACKGROUND (balo, cặp xách, ghế tựa, hộp phía sau).
        """
        if hands:
            for h in hands:
                if isinstance(h, dict):
                    h["holding_phone"] = False

        has_hand = bool(hands and len(hands) > 0)
        any_resting_hand = any(self.is_hand_resting_on_face(h, face_box) for h in (hands or []))
        conf_thresh = 0.22 if (has_hand or face_box) else 0.32

        if self.net is not None:
            candidates = self.detect_phone_onnx(frame, conf_threshold=conf_thresh, imgsz=640)
        elif self.yolo is not None:
            candidates = []
            res = self.yolo(frame, verbose=False, conf=conf_thresh, classes=[67], imgsz=640)
            for r in res:
                for box in r.boxes:
                    b = [int(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    candidates.append({"box": b, "conf": round(conf, 2), "is_call": False, "is_near_user": False})
        else:
            candidates = []

        if not candidates:
            return False, []

        h_f, w_f = frame.shape[:2]
        valid_phone_boxes = []

        fw, fh = (max(1, face_box[2] - face_box[0]), max(1, face_box[3] - face_box[1])) if face_box else (1, 1)
        fcx = (face_box[0] + face_box[2]) / 2.0 if face_box else (w_f / 2.0)

        for cand in candidates:
            b = cand["box"]
            conf = cand["conf"]
            bw, bh = b[2] - b[0], b[3] - b[1]

            if min(bw, bh) < 16 or max(bw, bh) < 32:
                continue

            # Loại trừ nếu là khuôn mặt hoặc gọng kính ngang
            if self.is_face_or_eyeglasses(b, face_box):
                continue

            # Loại trừ vùng da trần (như bàn tay chống cằm / áp má bị nhầm)
            crop_b = frame[max(0, b[1]):min(h_f, b[3]), max(0, b[0]):min(w_f, b[2])]
            if crop_b.shape[0] >= 16 and crop_b.shape[1] >= 16:
                has_chassis, non_skin_ratio, _ = self.detect_phone_chassis_features(crop_b)
                if not has_chassis and non_skin_ratio < 0.25:
                    continue

            near_face = self.is_phone_near_face(b, face_box) if face_box else False
            attached_to_hand = self.is_hand_attached(b, hands, face_box) if hands else False
            near_torso = self.is_near_face_vicinity(b, face_box) if face_box else False

            # --- BỘ LỌC KHỬ VẬT THỂ BACKGROUND (CẶP, BALO, GIỎ ĐỒ, GHẾ SAU LƯNG) ---
            # 1. Nếu có person_mask từ ImageSegmenter: kiểm tra tỷ lệ thuộc Foreground
            # Áp dụng cho mọi vật thể không được tay cầm chặt (kể cả khi ở gần đầu/thân trên tọa độ 2D)
            if person_mask is not None and not attached_to_hand:
                bx1_c = max(0, min(w_f - 1, b[0]))
                by1_c = max(0, min(h_f - 1, b[1]))
                bx2_c = max(bx1_c + 1, min(w_f, b[2]))
                by2_c = max(by1_c + 1, min(h_f, b[3]))
                sub_mask = person_mask[by1_c:by2_c, bx1_c:bx2_c]
                fg_ratio = float(np.mean(sub_mask > 0.40)) if sub_mask.size > 0 else 0.0
                if fg_ratio < 0.40:
                    # Hơn 60% diện tích nằm ở Background (ngoài cơ thể người) -> Bỏ qua 100%
                    continue

            # 2. Kiểm tra lệch trục thân người & kích cỡ vật thể (Background Object Rejection)
            bcx = (b[0] + b[2]) / 2.0
            if face_box and not attached_to_hand:
                # Kích thước vật thể: smartphone cầm tay không thể to hơn khuôn mặt người
                if bw > (fw * 0.90) and bh > (fh * 1.10):
                    continue
                if (bw * bh) > ((fw * fh) * 0.70):
                    continue

            # 3. NGUYÊN TẮC CỐT LÕI LOẠI TRỪ 100% ĐỒ VẬT HẬU CẢNH (BALO, GIỎ ĐỒ, GIÁ TREO ĐỒ):
            # Người dùng chỉ có thể sử dụng điện thoại khi:
            # - Hoặc ĐANG CẦM TRÊN TAY (attached_to_hand == True)
            # - Hoặc ĐANG NGHE MÁY SÁT TAI (near_face == True VÀ có bàn tay ở tư thế nghe gọi)
            # Nếu KHÔNG CÓ BÀN TAY NÀO CẦM và KHÔNG CÓ BÀN TAY ĐƯA LÊN TAI:
            # Bất kỳ vật thể nào trơ trọi (giỏ đồ, balo, gối, áo sau lưng) TUYỆT ĐỐI BỎ QUA!
            any_calling_hand = any(self.is_calling_pose(h, face_box) for h in (hands or []))
            hand_near_ear = any(
                self.is_phone_near_face(h.get("box", h) if isinstance(h, dict) else h, face_box)
                for h in (hands or [])
            )

            if not attached_to_hand and not (near_face and (any_calling_hand or hand_near_ear)):
                continue

            # Phân loại hành động:
            if near_face and (attached_to_hand or any_calling_hand or hand_near_ear) and not any_resting_hand:
                # Điện thoại áp sát mặt / tai kèm tay đưa lên nghe máy -> Cuộc gọi điện thoại (CALL)
                cand["is_call"] = True
                cand["is_near_user"] = True
                valid_phone_boxes.append(cand)
            elif attached_to_hand:
                # Bàn tay đang thực sự cầm nắm điện thoại trước ngực / bàn làm việc
                cand["is_call"] = False
                cand["is_near_user"] = True
                valid_phone_boxes.append(cand)

        has_phone = len(valid_phone_boxes) > 0
        return has_phone, valid_phone_boxes
