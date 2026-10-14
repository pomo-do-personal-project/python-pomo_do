"""
server.py - FastAPI WebSocket Server cho pomo-ai
Dành cho dự án: pomo-ai (Pomo-do)
Mục tiêu: Cung cấp WebSocket endpoint để giao diện web pomo-do kết nối,
nhận dữ liệu sự tập trung thời gian thực và điều khiển cấu hình màn hình.

Endpoints:
  - GET  /         : Kiểm tra trạng thái server (Health check).
  - GET  /screens  : Lấy danh sách các màn hình hiện tại.
  - WS   /ws       : WebSocket kết nối trực tiếp với web app pomo-do.
"""

import asyncio
import json
import base64
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

from detector import FocusDetector

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Pomo-AI Focus Tracking Service",
    description="WebSocket & API Service nhận diện tập trung và hiệu chuẩn nhiều màn hình cho Pomo-do",
    version="1.0.0"
)

# Cho phép Cross-Origin để Frontend (Next.js / React) kết nối thoải mái
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo bộ nhận diện FocusDetector dùng chung
detector = FocusDetector()


@app.get("/")
def health_check():
    """Kiểm tra hoạt động của service."""
    return {
        "status": "online",
        "service": "pomo-ai",
        "screens_count": len(detector.screens)
    }


@app.get("/screens")
def get_screens():
    """Lấy danh sách các màn hình đã lưu trong config.json."""
    return {
        "total_screens": len(detector.screens),
        "screens": detector.screens
    }


class CameraManager:
    """Quản lý Webcam dùng chung để nhiều tab trình duyệt kết nối đồng thời không bị lỗi xung đột camera."""
    def __init__(self):
        self.cap = None
        self.clients = set()
        self.is_running = False
        self.lock = asyncio.Lock()
        self.latest_result = {
            "status": "AFK",
            "is_focused": False,
            "active_screen": None,
            "total_screens": 0,
            "look_x": None,
            "look_y": None,
            "has_face": False,
            "phone_boxes": []
        }

    async def add_client(self, websocket: WebSocket):
        async with self.lock:
            self.clients.add(websocket)
            if not self.is_running:
                self.is_running = True
                asyncio.create_task(self._capture_loop())

    async def remove_client(self, websocket: WebSocket):
        async with self.lock:
            self.clients.discard(websocket)
            if len(self.clients) == 0 and self.is_running:
                self.is_running = False
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                print("[pomo-ai server] Tat ca client da ngat, da giai phong webcam.")

    async def _capture_loop(self):
        print("[pomo-ai server] Dang khoi dong Webcam cho cac client...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[pomo-ai server] Khong the mo webcam!")
            self.is_running = False
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.is_running and len(self.clients) > 0:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.latest_result = detector.check_focus(frame)

                # Vẽ HUD và FaceMesh theo phong cách ảnh 2 của người dùng
                preview = frame.copy()
                h_f, w_f = preview.shape[:2]

                has_face = self.latest_result.get("has_face", False)
                has_phone = self.latest_result.get("has_phone", False)
                is_focused = self.latest_result.get("is_focused", False)
                active_screen = self.latest_result.get("active_screen")
                face_box = self.latest_result.get("face_box")
                mesh_points = self.latest_result.get("mesh_points", [])
                phone_boxes = self.latest_result.get("phone_boxes", [])

                # 1. Vẽ các điểm mốc FaceMesh chấm trắng li ti trên khuôn mặt (chuẩn ảnh 2)
                for pt in mesh_points:
                    cv2.circle(preview, pt, 1, (245, 245, 245), -1)

                # 2. Vẽ khung bao khuôn mặt (Face Bounding Box: Xanh khi tập trung, Đỏ khi mất tập trung)
                if face_box:
                    fx1, fy1, fx2, fy2 = face_box
                    box_color = (0, 255, 0) if is_focused else (0, 0, 255)
                    cv2.rectangle(preview, (fx1, fy1), (fx2, fy2), box_color, 2)

                    # Text góc trên khung: "Status: Focused" hoặc "Status: Distracted"
                    status_text = "Status: Focused" if is_focused else f"Status: {self.latest_result.get('detail', 'Distracted')}"
                    cv2.putText(preview, status_text, (fx1, max(fy1 - 8, 22)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

                    # Text góc dưới khung: "Gaze: forward" hoặc "Gaze: Screen X"
                    if is_focused:
                        gaze_text = f"Gaze: Screen {active_screen}" if active_screen else "Gaze: forward"
                    else:
                        gaze_text = "Gaze: away"
                    cv2.putText(preview, gaze_text, (fx1, min(fy2 + 20, h_f - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # 3. Vẽ chữ HUD góc trên bên trái (chuẩn ảnh 2):
                # Dòng 1: "No phone" (xanh lá) hoặc "Phone: YES" (đỏ)
                phone_str = "Phone: YES" if has_phone else "No phone"
                phone_col = (0, 0, 255) if has_phone else (0, 255, 0)
                cv2.putText(preview, phone_str, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, phone_col, 2, cv2.LINE_AA)

                # Dòng 2: "Face: YES" (xanh lá) hoặc "Face: NO" (đỏ)
                face_str = "Face: YES" if has_face else "Face: NO"
                face_col = (0, 255, 0) if has_face else (0, 0, 255)
                cv2.putText(preview, face_str, (15, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, face_col, 2, cv2.LINE_AA)

                # 4. Chỉ vẽ khung đỏ khi phát hiện điện thoại thật sự (đã qua lọc ghế và kính)
                for p in phone_boxes:
                    x1, y1, x2, y2 = p["box"]
                    conf = p["conf"]
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(preview, f"PHONE {int(conf*100)}%", (x1 + 4, max(y1 - 6, 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

                # Nén ảnh JPEG sắc nét (480x360) dạng Base64 gửi cho màn hình hiển thị trên web
                thumb = cv2.resize(preview, (480, 360))
                _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
                image_base64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

                # Broadcast kết quả đến tất cả client đang kết nối
                disconnected = []
                for ws in list(self.clients):
                    try:
                        await ws.send_json({
                            "event": "focus_update",
                            "image": image_base64,
                            **self.latest_result
                        })
                    except Exception:
                        disconnected.append(ws)

                for ws in disconnected:
                    self.clients.discard(ws)

            await asyncio.sleep(0.3)

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False


camera_manager = CameraManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint /ws phục vụ giao tiếp 2 chiều với web pomo-do:
    - Server gửi: kết quả nhận diện (status, is_focused, look_x, look_y, active_screen...) mỗi 0.3 giây.
    - Web gửi lên: lệnh điều khiển (action: 'add_screen', 'reset', 'get_screens').
    """
    await websocket.accept()
    print("[pomo-ai server] Client web pomo-do da ket noi vao WebSocket /ws")
    await camera_manager.add_client(websocket)

    try:
        while True:
            message_text = await websocket.receive_text()
            try:
                payload = json.loads(message_text)
            except Exception:
                payload = {"action": message_text}

            action = payload.get("action")

            if action == "add_screen":
                new_screen = detector.add_screen()
                await websocket.send_json({
                    "event": "screen_added",
                    "success": new_screen is not None,
                    "screen": new_screen,
                    "total_screens": len(detector.screens)
                })

            elif action == "reset":
                detector.reset_screens()
                await websocket.send_json({
                    "event": "screens_reset",
                    "total_screens": 0
                })

            elif action == "get_screens":
                await websocket.send_json({
                    "event": "screens_list",
                    "screens": detector.screens,
                    "total_screens": len(detector.screens)
                })

            elif action == "get_status":
                await websocket.send_json({
                    "event": "focus_update",
                    **camera_manager.latest_result
                })

            elif action == "stop_camera":
                async with camera_manager.lock:
                    camera_manager.is_running = False
                    if camera_manager.cap is not None:
                        camera_manager.cap.release()
                        camera_manager.cap = None
                print("[pomo-ai server] Da tat phan cung webcam theo yeu cau tu web.")
                await websocket.send_json({"event": "camera_stopped"})

            elif action == "start_camera":
                async with camera_manager.lock:
                    if not camera_manager.is_running:
                        camera_manager.is_running = True
                        asyncio.create_task(camera_manager._capture_loop())
                print("[pomo-ai server] Da bat lai phan cung webcam theo yeu cau tu web.")
                await websocket.send_json({"event": "camera_started"})

    except WebSocketDisconnect:
        print("[pomo-ai server] Client web pomo-do da ngat ket noi.")
    except Exception as e:
        print(f"[pomo-ai server] Loi ket noi client: {e}")
    finally:
        await camera_manager.remove_client(websocket)


if __name__ == "__main__":
    print("=" * 60)
    print("  POMO-AI FASTAPI WEBSOCKET SERVER DANG KHOI CHAY")
    print("  Dia chi WebSocket: ws://localhost:8001/ws")
    print("  Dia chi API doc:    http://localhost:8001/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8001)
