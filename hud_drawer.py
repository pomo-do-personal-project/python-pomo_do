"""
hud_drawer.py - Vẽ giao diện HUD, FaceMesh và nhận diện bàn tay lên camera
"""

import cv2


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),        # Ngón trỏ
    (5, 9), (9, 10), (10, 11), (11, 12),   # Ngón giữa
    (9, 13), (13, 14), (14, 15), (15, 16), # Ngón áp út
    (13, 17), (17, 18), (18, 19), (19, 20),# Ngón út
    (0, 17)                                # Cạnh bàn tay
]


def draw_hud(preview, latest_result):
    """Vẽ FaceMesh, bounding box, thông tin trạng thái, bàn tay và điện thoại."""
    h_f, w_f = preview.shape[:2]

    has_face = latest_result.get("has_face", False)
    has_phone = latest_result.get("has_phone", False)
    raw_has_phone = latest_result.get("raw_has_phone", False)
    is_focused = latest_result.get("is_focused", False)
    active_screen = latest_result.get("active_screen")
    face_box = latest_result.get("face_box")
    mesh_points = latest_result.get("mesh_points", [])
    phone_boxes = latest_result.get("phone_boxes", [])
    hands = latest_result.get("hands", [])

    # 1. Vẽ khung bao khuôn mặt (Xanh khi tập trung, Đỏ khi mất tập trung)
    if face_box:
        fx1, fy1, fx2, fy2 = face_box
        box_color = (0, 255, 0) if is_focused else (0, 0, 255)
        cv2.rectangle(preview, (fx1, fy1), (fx2, fy2), box_color, 2)

        status_text = "Status: Focused" if is_focused else f"Status: {latest_result.get('detail', 'Distracted')}"
        cv2.putText(preview, status_text, (fx1, max(fy1 - 8, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

        gaze_text = f"Gaze: Screen {active_screen}" if (is_focused and active_screen) else ("Gaze: forward" if is_focused else "Gaze: away")
        cv2.putText(preview, gaze_text, (fx1, min(fy2 + 20, h_f - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Vẽ chữ HUD góc trên bên trái
    is_calling = latest_result.get("is_calling", False)
    has_call = latest_result.get("has_call", False)

    if has_phone:
        phone_str = "Phone: YES (CALL)" if has_call else "Phone: YES"
        phone_col = (0, 0, 255)
    elif is_calling and has_call:
        phone_str = "Phone: CHECKING CALL..."
        phone_col = (0, 215, 255)
    elif raw_has_phone or len(phone_boxes) > 0:
        phone_str = "Phone: CHECKING..."
        phone_col = (0, 215, 255)
    else:
        phone_str = "Phone: NO"
        phone_col = (0, 255, 0)
    cv2.putText(preview, phone_str, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, phone_col, 2, cv2.LINE_AA)

    face_str = "Face: YES" if has_face else "Face: NO"
    face_col = (0, 255, 0) if has_face else (0, 0, 255)
    cv2.putText(preview, face_str, (15, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, face_col, 2, cv2.LINE_AA)

    # 4. Vẽ nhận diện bàn tay & 21 chấm bi khớp ngón tay (Hand Landmarks)
    for hand in hands:
        is_holding = hand.get("holding_phone", False)
        hand_is_calling = hand.get("is_calling", False)
        pts = hand.get("points", [])

        # Phân biệt màu sắc: Đỏ khi đã xác nhận nghe điện thoại/cầm máy, Trắng/vàng khi tay tự do
        if hand_is_calling and has_call and has_phone:
            line_col = (0, 0, 255)
            dot_col = (0, 0, 255)
        elif is_holding and has_phone:
            line_col = (0, 0, 255)
            dot_col = (0, 0, 255)
        else:
            line_col = (0, 215, 255)
            dot_col = (255, 255, 255)

        # Vẽ xương nối giữa các khớp ngón tay và chấm bi
        if pts and len(pts) == 21:
            for (start, end) in HAND_CONNECTIONS:
                cv2.line(preview, pts[start], pts[end], line_col, 1, cv2.LINE_AA)
            for i, pt in enumerate(pts):
                r_dot = 3 if i in (4, 8, 12, 16, 20) else 2
                cv2.circle(preview, pt, r_dot, dot_col, -1)

        # Hiển thị nhãn trạng thái bàn tay CHỈ KHI có điện thoại thực tế
        hx, hy = hand["center"]
        r = hand.get("radius", 35)
        if hand_is_calling and has_call and has_phone:
            cv2.putText(preview, "NGHE DIEN THOAI", (max(10, hx - 65), max(hy - int(r * 0.8) - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2, cv2.LINE_AA)
        elif is_holding and has_phone:
            cv2.putText(preview, "HOLDING PHONE", (max(10, hx - 45), max(hy - int(r * 0.8) - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)
            # Vẽ tia nối trực tiếp từ tay tới điện thoại
            for p in phone_boxes:
                px1, py1, px2, py2 = p["box"]
                cv2.line(preview, (hx, hy), (int((px1 + px2) / 2), int((py1 + py2) / 2)), (0, 0, 255), 2, cv2.LINE_AA)

    # 5. Vẽ khung bao quanh điện thoại (Vàng khi đang kiểm tra buffer, Đỏ khi đã xác nhận)
    for p in phone_boxes:
        conf = p["conf"]
        x1, y1, x2, y2 = p["box"]
        is_call = p.get("is_call", False)
        if has_phone:
            b_color = (0, 0, 255)
            b_label = f"PHONE CALL {int(conf*100)}%" if is_call else f"PHONE {int(conf*100)}%"
        else:
            b_color = (0, 215, 255)
            b_label = f"CHECKING CALL {int(conf*100)}%" if is_call else f"CHECKING {int(conf*100)}%"
        cv2.rectangle(preview, (x1, y1), (x2, y2), b_color, 2)
        cv2.putText(preview, b_label, (x1 + 4, max(y1 - 6, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, b_color, 1, cv2.LINE_AA)

    return preview
