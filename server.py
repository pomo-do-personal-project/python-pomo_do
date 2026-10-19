"""
server.py - FastAPI WebSocket Server cho pomo-ai (Pomo-do)
Quản lý luồng webcam, nhận diện sự tập trung và điều khiển hiệu chuẩn màn hình.
"""

import os
import sys
import asyncio
import json

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import base64
import cv2
import gc
import numpy as np
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


def get_current_ram_mb() -> float:
    """Lấy số MB RAM thực tế (RSS) đang sử dụng (hoạt động chuẩn trên cả Windows CMD và Linux/Render)."""
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    except Exception:
        try:
            if os.path.exists("/proc/self/status"):
                with open("/proc/self/status", "r") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            return round(int(line.split()[1]) / 1024, 2)
        except Exception:
            pass
    return 0.0


@app.get("/screens")
def get_screens():
    return {"total_screens": len(detector.screens), "screens": detector.screens}


@app.get("/memory")
def get_memory_stats():
    """Đo lường RAM thực tế tiến trình Python đang sử dụng (MB)."""
    ram_mb = get_current_ram_mb()
    percent = round((ram_mb / 512.0) * 100, 1)
    return {
        "current_ram_mb": ram_mb,
        "render_free_limit_mb": 512.0,
        "usage_percentage": f"{percent}%",
        "status": "HEALTHY" if percent <= 85.0 else "WARNING",
        "message": "RAM an toàn trong hạn mức 512MB" if percent <= 85.0 else "RAM cao"
    }


async def ram_monitor_loop():
    """Vòng lặp chạy ngầm in RAM sử dụng trực tiếp ra màn hình CMD mỗi 10 giây."""
    while True:
        await asyncio.sleep(10)
        ram = get_current_ram_mb()
        percent = round((ram / 512.0) * 100, 1)
        status = "AN TOÀN" if percent <= 80 else "CẢNH BÁO CAO"
        print(f"📊 [pomo-ai RAM] Đang dùng: {ram:.1f} MB / 512 MB ({percent}%) | Đã nhận: {frame_counter} frames | Trạng thái: {status}")


@app.on_event("startup")
async def startup_event():
    ram = get_current_ram_mb()
    percent = round((ram / 512.0) * 100, 1)
    print("\n" + "=" * 60)
    print(f"🚀 POMO-AI SERVER ĐÃ SẴN SÀNG!")
    print(f"📊 RAM lúc nạp Model: {ram:.1f} MB / 512 MB ({percent}%)")
    print(f"🌐 WebSocket: ws://localhost:8001/ws")
    print(f"🔗 Xem API RAM: http://localhost:8001/memory")
    print("=" * 60 + "\n")
    asyncio.create_task(ram_monitor_loop())


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


frame_counter = 0

async def process_client_frame(websocket: WebSocket, image_data: str):
    """Xử lý từng khung hình do trình duyệt của người dùng gửi lên qua WebSocket."""
    global frame_counter
    try:
        frame_counter += 1
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # Lật ảnh ngang (mirror) để người dùng nhìn tự nhiên như gương
        frame = cv2.flip(frame, 1)

        # Chạy suy luận nhận diện tập trung qua detector
        result = await asyncio.to_thread(detector.check_focus, frame)

        # Vẽ HUD và FaceMesh trực quan (Tối ưu 440px và chất lượng 65 để độ trễ siêu thấp)
        preview = draw_hud(frame.copy(), result)
        h_f, w_f = preview.shape[:2]
        if w_f > 440:
            preview = cv2.resize(preview, (440, int(h_f * 440 / w_f)))
        _, buf = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 65])
        image_base64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

        # Dọn dẹp biến để giải phóng RAM ngay lập tức
        del img_bytes, np_arr, frame, preview, buf

        # Thu hồi rác định kỳ mỗi 15 frame để giữ RAM siêu nhẹ
        if frame_counter % 15 == 0:
            gc.collect()
            try:
                import ctypes
                if os.name == "nt":
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
                else:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
            ram_mb = get_current_ram_mb()
            percent = round((ram_mb / 512.0) * 100, 1)
            print(f"📸 [pomo-ai Frame #{frame_counter}] RAM: {ram_mb:.1f} MB ({percent}%) | Thu hồi rác GC xong")

        # Trả kết quả kèm ảnh preview đã vẽ khung cho chính client này
        await websocket.send_json({
            "event": "focus_update",
            "image": image_base64,
            **result
        })
    except Exception as e:
        print(f"[pomo-ai server] Lỗi xử lý frame từ client: {e}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint /ws phục vụ giao tiếp 2 chiều với web pomo-do:
    - Client gửi: khung hình webcam (action: 'frame', image: 'data:image/jpeg;base64,...')
                  và các lệnh điều khiển ('add_screen', 'reset', 'get_screens', 'set_sensitivity').
    - Server gửi: kết quả nhận diện (status, is_focused, look_x, look_y, active_screen, image...).
    """
    await websocket.accept()
    print("[pomo-ai server] Client web pomo-do đã kết nối vào WebSocket /ws")

    try:
        while True:
            message_text = await websocket.receive_text()
            try:
                payload = json.loads(message_text)
            except Exception:
                payload = {"action": message_text}

            action = payload.get("action")

            if action == "frame":
                img = payload.get("image")
                if img:
                    await process_client_frame(websocket, img)
            elif action == "add_screen":
                new_screen = detector.add_screen()
                await websocket.send_json({"event": "screen_added", "success": new_screen is not None, "total_screens": len(detector.screens)})
            elif action == "reset":
                detector.reset_screens()
                await websocket.send_json({"event": "screens_reset", "total_screens": 0})
            elif action == "get_screens":
                await websocket.send_json({"event": "screens_list", "screens": detector.screens, "total_screens": len(detector.screens)})
            elif action == "set_sensitivity":
                level = payload.get("level", "relaxed")
                res = detector.set_sensitivity(level)
                await websocket.send_json({"event": "sensitivity_updated", **res})
    except (WebSocketDisconnect, Exception) as e:
        print(f"[pomo-ai server] Client ngắt kết nối: {e}")

if __name__ == "__main__":
    init_ram = get_current_ram_mb()
    percent = round((init_ram / 512.0) * 100, 1)
    print("=" * 60)
    print("🚀 POMO-AI Server running on ws://localhost:8001/ws")
    print(f"📊 RAM lúc nạp Model: {init_ram:.1f} MB / 512 MB ({percent}%)")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8001)
