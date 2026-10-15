"""
screen_manager.py - Quản lý cấu hình màn hình hiệu chuẩn cho pomo-ai
"""

import os
import json
import math


class ScreenManager:
    """Quản lý lưu trữ và đối chiếu các màn hình hiệu chuẩn."""

    def __init__(self, config_filename="config.json"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(current_dir, config_filename)
        self.screens = []
        self.load_screens()

    def load_screens(self):
        """Đọc danh sách màn hình từ config.json."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.screens = json.load(f)
                    print(f"[pomo-ai] Đã tải {len(self.screens)} màn hình từ {self.config_path}")
            except Exception as e:
                print(f"[pomo-ai] Lỗi đọc config.json ({e}), khởi tạo danh sách trống.")
                self.screens = []
        else:
            self.screens = []

    def save_screens(self):
        """Lưu danh sách màn hình vào config.json."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.screens, f, indent=2, ensure_ascii=False)
            print(f"[pomo-ai] Đã lưu {len(self.screens)} màn hình vào {self.config_path}")
            return True
        except Exception as e:
            print(f"[pomo-ai] Lỗi lưu file config.json: {e}")
            return False

    def add_screen(self, x=None, y=None, has_face=True, default_x=0.0, default_y=0.0):
        """Thêm màn hình mới vào danh sách."""
        if x is None and y is None and not has_face:
            print("[pomo-ai] Cảnh báo: Camera không thấy mặt! Vui lòng nhìn rõ vào camera/màn hình.")
            return None

        target_x = x if x is not None else default_x
        target_y = y if y is not None else default_y

        new_screen = {
            "id": len(self.screens) + 1,
            "x": round(float(target_x), 3),
            "y": round(float(target_y), 3)
        }
        self.screens.append(new_screen)
        self.save_screens()
        print(f"[pomo-ai] Đã thêm Màn hình {new_screen['id']} (x={new_screen['x']}, y={new_screen['y']})")
        return new_screen

    def reset_screens(self):
        """Xóa toàn bộ màn hình đã lưu."""
        self.screens = []
        self.save_screens()
        print("[pomo-ai] Đã đặt lại danh sách màn hình.")
        return True

    def match_screen(self, look_x, look_y, tol_x, tol_y):
        """Tìm màn hình khớp với hướng nhìn (look_x, look_y)."""
        screens_to_check = self.screens if len(self.screens) > 0 else [{"id": 1, "x": 0.0, "y": 0.0}]
        matched_id = None
        min_dist = float("inf")

        for s in screens_to_check:
            dx = abs(look_x - s["x"])
            dy = abs(look_y - s["y"])
            if dx <= tol_x and dy <= tol_y:
                dist = math.hypot(dx, dy)
                if dist < min_dist:
                    min_dist = dist
                    matched_id = s["id"]

        return matched_id
