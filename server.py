"""
server.py - FastAPI WebSocket Server cho pomo-ai (Pomo-do)
Quản lý luồng webcam, nhận diện sự tập trung và điều khiển hiệu chuẩn màn hình.
"""

import os
import sys
import asyncio
import json
import base64
import cv2
import gc
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from contextlib import asynccontextmanager
from detector import FocusDetector
from vision_pipeline import VisionPipeline
from hud_drawer import draw_hud
from routes_theme import router as theme_router
from routes_session import router as session_router, get_current_ram_mb, analyze_session

global_frame_counter = 0


async def ram_monitor_loop():
    """Vòng lặp chạy ngầm in RAM sử dụng trực tiếp ra màn hình CMD mỗi 10 giây."""
    while True:
        await asyncio.sleep(10)
        ram = get_current_ram_mb()
        percent = round((ram / 512.0) * 100, 1)
        status = "AN TOÀN" if percent <= 80 else "CẢNH BÁO CAO"
        print(f"📊 [pomo-ai RAM] Đang dùng: {ram:.1f} MB / 512 MB ({percent}%) | Đã nhận: {global_frame_counter} frames | Trạng thái: {status}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ram = get_current_ram_mb()
    percent = round((ram / 512.0) * 100, 1)
    print("\n" + "=" * 60)
    print("🚀 POMO-AI SERVER ĐÃ SẴN SÀNG!")
    print(f"📊 RAM lúc nạp Model: {ram:.1f} MB / 512 MB ({percent}%)")
    print("🌐 WebSocket: ws://localhost:8001/ws")
    print("🔗 Xem API RAM: http://localhost:8001/memory")
    print("=" * 60 + "\n")
    monitor_task = asyncio.create_task(ram_monitor_loop())
    try:
        yield
    finally:
        monitor_task.cancel()


app = FastAPI(title="Pomo-AI Focus Service", version="1.0.0", lifespan=lifespan)

# Siết CORS: Chỉ cho phép nguồn frontend được chỉ định (tránh wildcard * khi có credentials)
cors_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,https://localhost:3000")
allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Gắn các route chức năng độc lập
app.include_router(theme_router)
app.include_router(session_router)

# Nạp model AI 1 lần duy nhất để tối ưu RAM (Shared Vision Pipeline)
current_dir = os.path.dirname(os.path.abspath(__file__))
shared_vision = VisionPipeline(current_dir)
_module_mtimes = {}
_last_reload_check = 0.0


def _hot_reload_if_changed(client_detector: FocusDetector):
    """Tự động nạp lại logic khi tệp thay đổi (giới hạn kiểm tra mỗi 3.0s để không gây lag frame)."""
    global _last_reload_check
    import time
    now = time.time()
    if (now - _last_reload_check) < 3.0:
        return
    _last_reload_check = now

    import importlib
    import vision_pipeline as vp_mod
    import phone_detector as pd_mod
    import detector as det_mod
    import vlm_verifier as vlm_mod

    vp_path = os.path.join(current_dir, "vision_pipeline.py")
    pd_path = os.path.join(current_dir, "phone_detector.py")
    det_path = os.path.join(current_dir, "detector.py")
    vlm_path = os.path.join(current_dir, "vlm_verifier.py")
    try:
        vp_m = os.path.getmtime(vp_path) if os.path.exists(vp_path) else 0
        pd_m = os.path.getmtime(pd_path)
        det_m = os.path.getmtime(det_path)
        vlm_m = os.path.getmtime(vlm_path) if os.path.exists(vlm_path) else 0
        if (
            _module_mtimes.get("vp") != vp_m
            or _module_mtimes.get("pd") != pd_m
            or _module_mtimes.get("det") != det_m
            or _module_mtimes.get("vlm") != vlm_m
        ):
            if _module_mtimes:
                importlib.reload(vlm_mod)
                importlib.reload(pd_mod)
                if _module_mtimes.get("vp") != vp_m:
                    importlib.reload(vp_mod)
                    new_vp = vp_mod.VisionPipeline(current_dir)
                    shared_vision.__dict__.update(new_vp.__dict__)
                    shared_vision.__class__ = vp_mod.VisionPipeline
                importlib.reload(det_mod)
                shared_vision.phone_detector = pd_mod.PhoneDetector(
                    shared_vision.net,
                    shared_vision.yolo,
                    input_shape=getattr(shared_vision, "yolo_input_shape", (320, 416)),
                )
                client_detector.__class__ = det_mod.FocusDetector
                client_detector.vlm_verifier = vlm_mod.VLMPhoneVerifier(current_dir)
                print("[pomo-ai server] Đã tự động cập nhật (hot-reload) pipeline & detector!")
            _module_mtimes["vp"] = vp_m
            _module_mtimes["pd"] = pd_m
            _module_mtimes["det"] = det_m
            _module_mtimes["vlm"] = vlm_m
        if getattr(client_detector, "vlm_verifier", None) is None:
            client_detector.vlm_verifier = vlm_mod.VLMPhoneVerifier(current_dir)
    except Exception:
        pass


@app.get("/")
@app.get("/ai")
@app.get("/ai/")
def health_check():
    return {"status": "online", "service": "pomo-ai"}


@app.get("/screens")
@app.get("/ai/screens")
def get_screens():
    return {"message": "Mỗi kết nối WebSocket quản lý cấu hình màn hình độc lập theo phiên client."}


async def process_client_frame(websocket: WebSocket, image_data: str, client_detector: FocusDetector, client_frame_idx: int):
    """Xử lý từng khung hình do trình duyệt của người dùng gửi lên qua WebSocket với detector độc lập của phiên đó."""
    global global_frame_counter
    try:
        global_frame_counter += 1
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # Lật ảnh ngang (mirror) để người dùng nhìn tự nhiên như gương
        frame = cv2.flip(frame, 1)
        _hot_reload_if_changed(client_detector)

        # Chạy suy luận nhận diện tập trung qua detector của client này (không bị ô nhiễm state từ client khác)
        result = await asyncio.to_thread(client_detector.check_focus, frame)

        # Không vẽ HUD / không mã hoá lại JPEG nữa: trình duyệt tự hiển thị webcam trực tiếp
        # và vẽ overlay từ toạ độ trả về -> giảm độ trễ & băng thông.
        result.pop("mesh_points", None)
        result.pop("points", None)
        for h in result.get("hands") or []:
            h.pop("held_phone_box", None)

        del img_bytes, np_arr, frame

        # Thu hồi rác định kỳ mỗi 200 frame để không gây khựng video
        if global_frame_counter % 200 == 0:
            gc.collect()

        await websocket.send_json({
            "event": "focus_update",
            **result
        })
    except Exception as e:
        print(f"[pomo-ai server] Lỗi xử lý frame từ client: {e}")


@app.websocket("/ws")
@app.websocket("/ai/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint /ws phục vụ giao tiếp 2 chiều với web pomo-do:
    - Mỗi phiên kết nối sở hữu một FocusDetector riêng biệt với shared VisionPipeline
    - Xác thực token bí mật nếu môi trường có đặt AI_SERVICE_SECRET
    """
    ai_secret = os.getenv("AI_SERVICE_SECRET")
    if ai_secret:
        token = websocket.query_params.get("token")
        if token != ai_secret:
            await websocket.close(code=1008, reason="Xác thực AI service thất bại")
            return

    await websocket.accept()
    print("[pomo-ai server] Client web pomo-do đã kết nối vào WebSocket /ws (Cô lập trạng thái)")

    # Khởi tạo detector độc lập cho client này (dùng chung VisionPipeline đã nạp sẵn model)
    client_detector = FocusDetector(config_filename=None, vision=shared_vision)
    client_frame_idx = 0

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
                    client_frame_idx += 1
                    await process_client_frame(websocket, img, client_detector, client_frame_idx)
            elif action == "add_screen":
                new_screen = client_detector.add_screen()
                await websocket.send_json({"event": "screen_added", "success": new_screen is not None, "total_screens": len(client_detector.screens)})
            elif action == "reset":
                client_detector.reset_screens()
                await websocket.send_json({"event": "screens_reset", "total_screens": 0})
            elif action == "get_screens":
                await websocket.send_json({"event": "screens_list", "screens": client_detector.screens, "total_screens": len(client_detector.screens)})
            elif action == "set_sensitivity":
                level = payload.get("level", "relaxed")
                res = client_detector.set_sensitivity(level)
                await websocket.send_json({"event": "sensitivity_updated", **res})
            elif action == "set_desk_reading_mode":
                enabled = payload.get("enabled", False)
                res = client_detector.set_desk_reading_mode(enabled)
                await websocket.send_json({"event": "reading_mode_updated", **res})
            elif action == "set_blur_background":
                enabled = payload.get("enabled", False)
                res = client_detector.set_blur_background(enabled)
                await websocket.send_json({"event": "blur_background_updated", **res})
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
