"""
test_cam.py - File kiểm tra webcam trực tiếp bằng giao diện OpenCV
Dành cho dự án: pomo-ai (Pomo-do)
Mục tiêu: Đơn giản, trực quan, hỗ trợ hiệu chuẩn nhiều màn hình bằng phím bấm.

Phím tắt:
  [c] Calibrate: Thêm hướng nhìn hiện tại làm 1 màn hình mới (bấm lần 1 cho màn 1, lần 2 cho màn 2...).
  [r] Reset: Xóa toàn bộ màn hình đã lưu để làm lại từ đầu.
  [q] Quit: Thoát khỏi chương trình.
"""

import time
import cv2
from detector import FocusDetector


def draw_hud(frame, result, toast_message, toast_time, toast_is_error=False):
    """Vẽ giao diện hiển thị thông số (HUD) lên khung hình OpenCV."""
    h, w = frame.shape[:2]

    status = result.get("status", "AFK")
    is_focused = result.get("is_focused", False)
    active_screen = result.get("active_screen")
    total_screens = result.get("total_screens", 0)
    look_x = result.get("look_x")
    look_y = result.get("look_y")
    points = result.get("points")
    has_face = result.get("has_face", False)

    # 1. Vẽ các điểm mốc trên mặt (mắt và đỉnh mũi) để người dùng dễ kiểm tra
    if points:
        if "nose" in points and points["nose"]:
            cv2.circle(frame, points["nose"], 6, (0, 0, 255), -1)       # Mũi màu đỏ
        if "left_eye" in points and points["left_eye"]:
            cv2.circle(frame, points["left_eye"], 5, (0, 255, 255), -1) # Mắt/tai trái
        if "right_eye" in points and points["right_eye"]:
            cv2.circle(frame, points["right_eye"], 5, (0, 255, 255), -1)# Mắt/tai phải

    # 2. Vẽ khung viền đỏ nổi bật quanh điện thoại nếu phát hiện
    phone_boxes = result.get("phone_boxes", [])
    for p in phone_boxes:
        x1, y1, x2, y2 = p["box"]
        conf = p["conf"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        tag = f"DIEN THOAI: {int(conf * 100)}%"
        cv2.putText(frame, tag, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    detail = result.get("detail", "")

    # 3. Bảng màu sắc tương ứng với 3 trạng thái lõi
    color_map = {
        "FOCUSED": (46, 204, 113),      # Xanh lá
        "DISTRACTED": (30, 30, 230),   # Đỏ
        "AFK": (140, 140, 140)          # Xám
    }
    status_color = color_map.get(status, (200, 200, 200))

    # 3. Hộp thông tin trạng thái ở góc trên bên trái (độ mờ trong suốt cao)
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (440, 125), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # Viền màu theo trạng thái
    cv2.rectangle(frame, (15, 15), (440, 125), status_color, 2)

    # Dòng 1: Trạng thái hiện tại
    if status == "FOCUSED":
        status_label = f"TRANG THAI: FOCUSED (Man {active_screen})"
    elif status == "AFK":
        status_label = "TRANG THAI: AFK (Khong thay nguoi)"
    else:
        status_label = "TRANG THAI: DISTRACTED"
    cv2.putText(frame, status_label, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.68, status_color, 2)

    # Dòng 2: Chi tiết giải thích trạng thái
    detail_info = detail if detail else ("Dang tap trung" if is_focused else "Mat tap trung")
    cv2.putText(frame, detail_info, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

    # Dòng 3: Tọa độ hướng nhìn (look_x, look_y)
    if has_face and look_x is not None and look_y is not None:
        coord_text = f"Goc nhin: X={look_x:+.3f}, Y={look_y:+.3f} (Da bat mat)"
        coord_color = (180, 240, 180)
    else:
        coord_text = "Goc nhin: Khong xac dinh (Mất dấu mặt)"
        coord_color = (160, 160, 255)
    cv2.putText(frame, coord_text, (25, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.48, coord_color, 1)

    # 4. Thanh thông báo (Toast Message khi bấm phím C hoặc R)
    if toast_message and (time.time() - toast_time < 3.0):
        msg_overlay = frame.copy()
        bg_color = (0, 0, 160) if toast_is_error else (0, 120, 0)
        border_color = (0, 0, 255) if toast_is_error else (0, 255, 0)

        cv2.rectangle(msg_overlay, (w // 2 - 270, 15), (w // 2 + 270, 58), bg_color, -1)
        cv2.addWeighted(msg_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (w // 2 - 270, 15), (w // 2 + 270, 58), border_color, 2)
        cv2.putText(frame, toast_message, (w // 2 - 255, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # 5. Hướng dẫn phím bấm ở đáy màn hình
    footer_overlay = frame.copy()
    cv2.rectangle(footer_overlay, (0, h - 35), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(footer_overlay, 0.75, frame, 0.25, 0, frame)

    guide_text = "[c]: Luu goc nhin man hinh nay  |  [r]: Reset tat ca  |  [q]: Thoat"
    cv2.putText(frame, guide_text, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def main():
    print("=" * 60)
    print("      POMO-AI: HE THONG THEO DOI TAP TRUNG QUA WEBCAM      ")
    print("=" * 60)
    print("Huong dan:")
    print("  1. Nhin vao Man hinh 1 -> Bam 'c' de luu Man hinh 1.")
    print("  2. Nhin sang Man hinh 2 -> Bam 'c' de luu Man hinh 2.")
    print("  3. Bam 'r' de xoa toan bo cac man hinh va cai dat lai.")
    print("  4. Bam 'q' de thoat.")
    print("=" * 60)

    # Khởi tạo bộ nhận diện FocusDetector
    detector = FocusDetector()

    # Mở camera mặc định (webcam máy tính)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Loi] Khong the ket noi den Webcam. Vui long kiem tra lai thiet bi!")
        return

    # Tối ưu kích thước khung hình hiển thị
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    toast_message = ""
    toast_time = 0
    toast_is_error = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Loi] Khong doc duoc du lieu tu Webcam!")
                break

            # Lật ảnh ngang như gương để người dùng dễ nhìn theo tự nhiên
            frame = cv2.flip(frame, 1)

            # Gọi class FocusDetector để phân tích sự tập trung
            result = detector.check_focus(frame)

            # Bắt sự kiện bàn phím
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[pomo-ai] Nguoi dung yeu cau thoat chuong trinh.")
                break

            elif key == ord('c'):
                # Kiểm tra xem camera có đang bắt được khuôn mặt không
                if not result.get("has_face"):
                    toast_message = "LOI: Khong thay mat! Hay quay mat ve camera roi bam 'c'!"
                    toast_is_error = True
                    toast_time = time.time()
                    print(f"[pomo-ai] {toast_message}")
                else:
                    # Calibrate: Lưu hướng nhìn hiện tại thành màn hình mới
                    new_scr = detector.add_screen()
                    if new_scr:
                        toast_message = f"Da luu Man hinh {new_scr['id']} (X={new_scr['x']}, Y={new_scr['y']})!"
                        toast_is_error = False
                    else:
                        toast_message = "LOI: Khong the luu man hinh!"
                        toast_is_error = True
                    toast_time = time.time()
                    print(f"[pomo-ai] {toast_message}")

            elif key == ord('r'):
                # Reset: Xóa toàn bộ màn hình đã lưu
                detector.reset_screens()
                toast_message = "Da xoa toan bo man hinh da luu!"
                toast_is_error = False
                toast_time = time.time()
                print(f"[pomo-ai] {toast_message}")

            # Vẽ giao diện thông số HUD lên khung hình
            draw_hud(frame, result, toast_message, toast_time, toast_is_error)

            # Hiển thị cửa sổ OpenCV
            cv2.imshow("Pomo-AI Focus Tracking Camera (Bam 'q' de thoat)", frame)

    finally:
        # Giải phóng tài nguyên camera và đóng toàn bộ cửa sổ OpenCV
        cap.release()
        cv2.destroyAllWindows()
        print("[pomo-ai] Da giai phong Webcam thanh cong!")


if __name__ == "__main__":
    main()
