"""
hud_drawer.py - Vẽ giao diện HUD, FaceMesh và nhận diện bàn tay lên camera
"""

import cv2


def draw_hud(preview, latest_result):
    """Vẽ FaceMesh, bounding box, thông tin trạng thái, bàn tay và điện thoại."""
    h_f, w_f = preview.shape[:2]

    has_face = latest_result.get("has_face", False)
    has_phone = latest_result.get("has_phone", False)
    is_focused = latest_result.get("is_focused", False)
    active_screen = latest_result.get("active_screen")
    face_box = latest_result.get("face_box")
    mesh_points = latest_result.get("mesh_points", [])
    phone_boxes = latest_result.get("phone_boxes", [])
    hands = latest_result.get("hands", [])

    # 1. Vẽ các điểm mốc FaceMesh chấm trắng li ti trên khuôn mặt
    for pt in mesh_points:
        cv2.circle(preview, pt, 1, (245, 245, 245), -1)

    # 2. Vẽ khung bao khuôn mặt (Xanh khi tập trung, Đỏ khi mất tập trung)
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
    phone_str = "Phone: YES" if has_phone else "No phone"
    phone_col = (0, 0, 255) if has_phone else (0, 255, 0)
    cv2.putText(preview, phone_str, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, phone_col, 2, cv2.LINE_AA)

    face_str = "Face: YES" if has_face else "Face: NO"
    face_col = (0, 255, 0) if has_face else (0, 0, 255)
    cv2.putText(preview, face_str, (15, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, face_col, 2, cv2.LINE_AA)

    # 4. Vẽ nhận diện bàn tay & tương tác Hand-Object Interaction
    for hand in hands:
        hx, hy = hand["center"]
        is_holding = hand.get("holding_phone", False)
        hand_id = hand.get("id", "Tay")
        r = hand.get("radius", 35)

        if is_holding:
            # Vòng tròn đỏ báo động khi bàn tay đang cầm điện thoại
            cv2.circle(preview, (hx, hy), int(r * 0.8), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(preview, (hx, hy), 4, (0, 0, 255), -1)
            cv2.putText(preview, f"{hand_id}: CAM DT", (max(10, hx - 45), max(hy - int(r * 0.8) - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)
            # Vẽ tia nối trực tiếp từ tay tới điện thoại
            for p in phone_boxes:
                px1, py1, px2, py2 = p["box"]
                cv2.line(preview, (hx, hy), (int((px1 + px2) / 2), int((py1 + py2) / 2)), (0, 0, 255), 2, cv2.LINE_AA)
        else:
            # Vòng tròn xanh ngọc công nghệ theo dõi vị trí bàn tay
            cv2.circle(preview, (hx, hy), int(r * 0.65), (255, 230, 0), 1, cv2.LINE_AA)
            cv2.circle(preview, (hx, hy), 3, (255, 230, 0), -1)
            cv2.putText(preview, f"TAY ({hand_id[0]})", (max(10, hx - 26), max(hy - int(r * 0.65) - 6, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 230, 0), 1, cv2.LINE_AA)

    # 5. Vẽ khung bao đỏ quanh điện thoại
    for p in phone_boxes:
        x1, y1, x2, y2 = p["box"]
        conf = p["conf"]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(preview, f"PHONE {int(conf*100)}%", (x1 + 4, max(y1 - 6, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    return preview
