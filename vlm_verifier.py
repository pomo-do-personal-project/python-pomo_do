"""
vlm_verifier.py - Module xác minh điện thoại tầng 2 (Hybrid VLM Verification) cho pomo-ai
Chạy bất đồng bộ (non-blocking background thread) khi Local AI (YOLO + MediaPipe) gặp ca khó
hoặc bị che khuất một phần (LIKELY / Occluded Hand-at-Ear / Ambiguous Grip).
"""

import os
import json
import time
import base64
import threading
import urllib.request
import urllib.error
import cv2

SECURITY_CAMERA_VLM_PROMPT = """You are the core brain of an advanced Real-time Security Camera AI. Your task is to analyze the provided image frame and detect if a person is actively using or holding a mobile phone, especially in challenging scenarios like making a phone call where the device is partially hidden.

PHYSICAL CHARACTERISTICS & EVIDENCE OF A REAL MOBILE PHONE:
1. Rigid Rectangular Geometry: A flat rectangular slab (approx 2:1 ratio) with straight hard edges, rounded corners, and a slim profile (7-10mm).
2. Camera Bump / Lenses: Visible circular camera lenses (2-3 round black circles) or flash LED at the upper back corner.
3. Flat Back / Case: Uniform flat surface (metal, glass, matte plastic) or a protective phone case with distinct color/texture, completely different from human skin tone and wrinkles.
4. Glass Display / Specular Reflections: Shiny flat glass surface showing light reflections or glowing screen.
5. Ports & Wires: USB-C/Lightning charging port at the bottom edge, or a charging cable plugged into the device.

CRITICAL DISCRIMINATION RULES:
- The hand MUST be physically grasping or pressing a real phone object.
- DISTINGUISH FROM EMPTY-HAND GESTURES:
  * Resting chin on a fist, palm, or thumb under the jaw -> NO PHONE (phone_detected: false).
  * Scratching the ear, rubbing the cheek, touching glasses, or resting hand on forehead -> NO PHONE (phone_detected: false).
  * Bare fingers touching the face/jaw without any rigid phone slab between hand and face -> NO PHONE (phone_detected: false).
- IGNORE ALL UI OVERLAYS: Completely disregard any bounding boxes, HUD text, or red labels (e.g. 'Phone: YES', 'PHONE CALL', 'NGHE DIEN THOAI') drawn on the image. Judge ONLY the actual physical human and objects.

You MUST respond strictly in the following JSON format without any markdown blocks or extra text:
{
  "person_present": true/false,
  "phone_detected": true/false,
  "action_type": "calling" / "browsing" / "none",
  "confidence_score": 0.00 to 1.00,
  "analysis_vietnamese": "A very brief 1-sentence explanation of what you see in Vietnamese"
}"""


def _load_dotenv_file(env_path: str):
    """Đọc tệp .env nhẹ nhàng không cần thư viện ngoài."""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


class VLMPhoneVerifier:
    """Bộ xác minh điện thoại Hybrid kết hợp Local CV và Cloud VLM (Gemini / Vertex Express)."""

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        _load_dotenv_file(os.path.join(base_dir, ".env"))

        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("VLM_MODEL", "gemini-3.5-flash").strip()
        self.vlm_only_mode = os.getenv("VLM_ONLY_MODE", "true").lower() in ("1", "true", "yes")

        interval_default = "3.2"
        try:
            self.cooldown_sec = float(os.getenv("VLM_INTERVAL_SECONDS", os.getenv("VLM_COOLDOWN_SECONDS", interval_default)))
        except ValueError:
            self.cooldown_sec = 3.2

        self.result_ttl_sec = 8.0
        self._lock = threading.Lock()
        self._is_running = False
        self._last_request_time = 0.0
        self._last_result = None
        self._last_result_time = 0.0

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_recent_result(self):
        """Lấy kết quả xác minh VLM gần nhất nếu còn trong thời hạn hiệu lực (TTL)."""
        with self._lock:
            if self._last_result is not None and (time.time() - self._last_result_time) <= self.result_ttl_sec:
                return dict(self._last_result)
            return None

    def clear_result(self):
        """Xóa kết quả đệm khi trạng thái bàn tay đã rời khỏi vùng nghi vấn."""
        with self._lock:
            self._last_result = None
            self._last_result_time = 0.0

    def trigger_async_verify(self, frame, roi_box=None) -> bool:
        """Kích hoạt luồng ngầm gửi ảnh lên VLM nếu không có request nào đang chạy và đã qua thời gian cooldown."""
        if not self.is_configured or frame is None:
            return False

        now = time.time()
        with self._lock:
            if self._is_running or (now - self._last_request_time) < self.cooldown_sec:
                return False
            self._is_running = True
            self._last_request_time = now

        # Sao chép khung hình để xử lý an toàn trên luồng riêng
        frame_copy = frame.copy()
        t = threading.Thread(
            target=self._worker_verify,
            args=(frame_copy, roi_box),
            daemon=True,
        )
        t.start()
        return True

    def _worker_verify(self, frame, roi_box=None):
        try:
            res = self.verify_frame_sync(frame, roi_box=roi_box)
            if res is not None:
                with self._lock:
                    self._last_result = res
                    self._last_result_time = time.time()
                print(
                    f"[pomo-ai VLM] Xác minh Hybrid: phone={res.get('phone_detected')} "
                    f"| action={res.get('action_type')} | conf={res.get('confidence_score')} "
                    f"| {res.get('analysis_vietnamese')}"
                )
        except Exception as e:
            print(f"[pomo-ai VLM] Cảnh báo luồng VLM: {e}")
        finally:
            with self._lock:
                self._is_running = False

    @staticmethod
    def _prepare_jpeg_base64(frame, roi_box=None, max_dim=512) -> str:
        """Chuẩn bị khung hình (hoặc vùng mở rộng quanh người/tay) sang base64 JPEG nhẹ và rõ nét."""
        h, w = frame.shape[:2]
        target = frame
        if roi_box is not None and len(roi_box) == 4:
            rx1, ry1, rx2, ry2 = roi_box
            rw, rh = max(1, rx2 - rx1), max(1, ry2 - ry1)
            # Mở rộng lớn để VLM thấy rõ toàn bộ đầu, tai, bàn tay và thân trên
            pad_x, pad_y = int(rw * 0.85), int(rh * 0.85)
            cx1, cy1 = max(0, rx1 - pad_x), max(0, ry1 - pad_y)
            cx2, cy2 = min(w, rx2 + pad_x), min(h, ry2 + pad_y)
            if (cx2 - cx1) >= 120 and (cy2 - cy1) >= 120:
                target = frame[cy1:cy2, cx1:cx2]

        th, tw = target.shape[:2]
        if max(th, tw) > max_dim:
            scale = max_dim / float(max(th, tw))
            target = cv2.resize(target, (int(tw * scale), int(th * scale)), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", target, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return ""
        return base64.b64encode(buf).decode("utf-8")

    @staticmethod
    def _parse_vlm_json(raw_text: str) -> dict:
        """Phân tích và chuẩn hóa chuỗi JSON trả về từ VLM."""
        if not raw_text:
            return None
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx : end_idx + 1]

        data = json.loads(cleaned)
        action_type = str(data.get("action_type", "none")).lower().strip()
        if action_type not in ("calling", "browsing", "none"):
            action_type = "none"

        conf = float(data.get("confidence_score", 0.0))
        conf = max(0.0, min(1.0, round(conf, 2)))

        phone_detected = bool(data.get("phone_detected", False))
        if not phone_detected:
            action_type = "none"

        return {
            "person_present": bool(data.get("person_present", False)),
            "phone_detected": phone_detected,
            "action_type": action_type,
            "confidence_score": conf,
            "analysis_vietnamese": str(data.get("analysis_vietnamese", "")).strip(),
        }

    def verify_frame_sync(self, frame, roi_box=None, timeout=10.0) -> dict:
        """Gọi trực tiếp VLM API để phân tích khung hình và trả về dict chuẩn theo schema."""
        if not self.is_configured or frame is None:
            return None

        b64_img = self._prepare_jpeg_base64(frame, roi_box=roi_box)
        if not b64_img:
            return None

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": SECURITY_CAMERA_VLM_PROMPT},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": b64_img,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 256,
                "responseMimeType": "application/json",
            },
        }
        body_bytes = json.dumps(payload).encode("utf-8")

        # Ưu tiên các model hoạt động nhanh và chính xác nhất với key
        models_to_try = [
            self.model,
            "gemini-2.5-flash",
            "gemini-flash-latest",
            "gemini-2.5-flash-lite",
            "gemini-3-flash-preview",
        ]
        seen = set()
        dedup_models = []
        for m in models_to_try:
            if m and m not in seen:
                seen.add(m)
                dedup_models.append(m)

        endpoints = [
            f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={self.api_key}"
            for m in dedup_models
        ]

        last_err = None
        for url in endpoints:
            req = urllib.request.Request(
                url,
                data=body_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    candidates = resp_data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text_out = "".join(p.get("text", "") for p in parts)
                        parsed = self._parse_vlm_json(text_out)
                        if parsed is not None:
                            return parsed
            except urllib.error.HTTPError as he:
                last_err = f"HTTP {he.code}"
                continue
            except Exception as e:
                last_err = str(e)
                continue

        if last_err:
            try:
                print(f"[pomo-ai VLM] Không thể kết nối VLM endpoint ({last_err})")
            except Exception:
                pass
        return None

