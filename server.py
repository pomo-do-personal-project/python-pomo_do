"""
server.py - FastAPI WebSocket Server cho pomo-ai (Pomo-do)
Quản lý luồng webcam, nhận diện sự tập trung và điều khiển hiệu chuẩn màn hình.
"""

import asyncio
import json
import base64
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from detector import FocusDetector
from hud_drawer import draw_hud

app = FastAPI(title="Pomo-AI Focus Service", version="1.0.0")

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
    return {"status": "online", "service": "pomo-ai", "screens_count": len(detector.screens)}


@app.get("/screens")
def get_screens():
    return {"total_screens": len(detector.screens), "screens": detector.screens}


@app.post("/analyze_session")
async def analyze_session(data: dict):
    """Phân tích dữ liệu quan sát của phiên học và đưa ra kết luận thông minh."""
    total = max(1, int(data.get("total_seconds", 0)))
    focused = int(data.get("focused_seconds", 0))
    phone_cnt = int(data.get("phone_count", 0))
    sleep_cnt = int(data.get("sleep_count", 0))
    look_cnt = int(data.get("look_away_count", 0))
    task_name = data.get("task_name") or "Tự do"

    score = min(100, max(0, round((focused / total) * 100)))
    grade = "S" if score >= 95 else "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 50 else "D"

    insights = []
    if score >= 85:
        insights.append(f"Xuất sắc! Bạn duy trì độ tập trung {score}% trong phiên '{task_name}'.")
    elif score >= 65:
        insights.append(f"Khá tốt ({score}%), bạn đã hoàn thành phần lớn thời gian học.")
    else:
        insights.append(f"Phiên này bị xao nhãng nhiều ({score}% tập trung).")

    if phone_cnt > 0:
        insights.append(f"Có {phone_cnt} lần dùng điện thoại.")
    if sleep_cnt > 0:
        insights.append(f"Có {sleep_cnt} lần ngủ gật.")
    if look_cnt > 0:
        insights.append(f"Có {look_cnt} lần nhìn ra ngoài.")

    conclusion = " ".join(insights)
    advice = "Để điện thoại xa tầm tay ở phiên kế tiếp." if phone_cnt > 0 else "Nên rửa mặt hoặc vươn vai nếu thấy buồn ngủ." if sleep_cnt > 0 else "Tiếp tục duy trì phong độ tập trung này!"

    return {
        "score": score,
        "grade": grade,
        "conclusion": conclusion,
        "advice": advice,
        "phone_count": phone_cnt,
        "sleep_count": sleep_cnt,
        "look_away_count": look_cnt,
        "focused_seconds": focused,
        "total_seconds": total
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
            "phone_boxes": [],
            "hands": []
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
                self.latest_result = await asyncio.to_thread(detector.check_focus, frame)

                # 1. Vẽ HUD và FaceMesh theo module hud_drawer
                preview = draw_hud(frame.copy(), self.latest_result)

                # 2. Nén ảnh JPEG sắc nét tối ưu (480x360, chất lượng 60) dạng Base64
                thumb = cv2.resize(preview, (480, 360))
                _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 60])
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

            # Nghỉ 15ms để đạt tốc độ mượt mà 25-30 FPS, không nghẽn CPU
            await asyncio.sleep(0.015)

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
                await websocket.send_json({"event": "screen_added", "success": new_screen is not None, "total_screens": len(detector.screens)})
            elif action == "reset":
                detector.reset_screens()
                await websocket.send_json({"event": "screens_reset", "total_screens": 0})
            elif action == "get_screens":
                await websocket.send_json({"event": "screens_list", "screens": detector.screens, "total_screens": len(detector.screens)})
            elif action == "get_status":
                await websocket.send_json({"event": "focus_update", **camera_manager.latest_result})
            elif action == "set_sensitivity":
                level = payload.get("level", "relaxed")
                res = detector.set_sensitivity(level)
                await websocket.send_json({"event": "sensitivity_updated", **res})
            elif action == "stop_camera":
                async with camera_manager.lock:
                    camera_manager.is_running = False
                    if camera_manager.cap is not None:
                        camera_manager.cap.release()
                        camera_manager.cap = None
                await websocket.send_json({"event": "camera_stopped"})
            elif action == "start_camera":
                async with camera_manager.lock:
                    if not camera_manager.is_running:
                        camera_manager.is_running = True
                        asyncio.create_task(camera_manager._capture_loop())
                await websocket.send_json({"event": "camera_started"})
    except (WebSocketDisconnect, Exception) as e:
        print(f"[pomo-ai server] Client ngat ket noi: {e}")
    finally:
        await camera_manager.remove_client(websocket)

if __name__ == "__main__":
    print("POMO-AI Server running on ws://localhost:8001/ws")
    uvicorn.run(app, host="0.0.0.0", port=8001)
