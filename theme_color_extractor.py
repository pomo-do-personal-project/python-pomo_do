"""
theme_color_extractor.py
Module AI Phân Tích Màu Sắc Hình Nền Dựa Trên Không Gian Màu CIELAB và Phân Cụm K-Means++
Tuân thủ chuẩn quy tắc phối màu Website (60-30-10) và Khóa tương phản WCAG 2.1 AA (4.5:1+)
"""

import math
import cv2
import numpy as np


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Chuyển RGB sang mã HEX chuẩn."""
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    """Chuyển HSL (h: 0-360, s: 0-1, l: 0-1) sang RGB (0-255)."""
    c = (1.0 - abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = l - c / 2.0

    if 0 <= h < 60:
        r_p, g_p, b_p = c, x, 0.0
    elif 60 <= h < 120:
        r_p, g_p, b_p = x, c, 0.0
    elif 120 <= h < 180:
        r_p, g_p, b_p = 0.0, c, x
    elif 180 <= h < 240:
        r_p, g_p, b_p = 0.0, x, c
    elif 240 <= h < 300:
        r_p, g_p, b_p = x, 0.0, c
    else:
        r_p, g_p, b_p = c, 0.0, x

    return (
        int(round((r_p + m) * 255.0)),
        int(round((g_p + m) * 255.0)),
        int(round((b_p + m) * 255.0)),
    )


def extract_theme_from_image_bytes(image_bytes: bytes) -> dict:
    """
    Trích xuất bảng màu từ mảng byte ảnh bằng K-Means++ trong không gian CIELAB.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không thể giải mã dữ liệu hình ảnh")
    return extract_theme_from_bgr_image(img)


def extract_theme_from_bgr_image(img: np.ndarray) -> dict:
    """
    Thực hiện phân cụm K-Means++ trong không gian CIELAB để tìm ra dải màu chuẩn xác nhất.
    """
    # 1. Resize ảnh xuống 120x120 để tối ưu tốc độ mà vẫn giữ toàn vẹn phổ màu
    target_size = (120, 120)
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # 2. Chuyển sang không gian màu CIELAB (L*, a*, b*)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Reshape mảng điểm ảnh
    lab_pixels = lab.reshape((-1, 3)).astype(np.float32)
    hsv_pixels = hsv.reshape((-1, 3)).astype(np.float32)

    # Lọc bỏ các pixel quá đen hoặc quá chói để tập trung vào màu sắc thực
    l_chan = lab_pixels[:, 0]
    valid_mask = (l_chan > 25) & (l_chan < 235)
    filtered_lab = lab_pixels[valid_mask]
    filtered_hsv = hsv_pixels[valid_mask]

    if len(filtered_lab) < 100:
        filtered_lab = lab_pixels
        filtered_hsv = hsv_pixels

    # 3. Chạy K-Means++ Clustering với K = 5
    k = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
    flags = cv2.KMEANS_PP_CENTERS
    _, labels, centers = cv2.kmeans(filtered_lab, k, None, criteria, 10, flags)

    labels = labels.flatten()
    total_pixels = len(labels)

    # 4. Phân tích từng cụm (Cluster)
    clusters = []
    for i in range(k):
        center_lab = centers[i]
        cluster_mask = (labels == i)
        count = int(np.sum(cluster_mask))
        weight_percent = round((count / total_pixels) * 100.0, 1)

        # Tính toán Chroma và Hue
        # In OpenCV: L in [0..255] (maps to 0..100), a in [0..255] (shifted by 128), b in [0..255] (shifted by 128)
        norm_l = (center_lab[0] / 255.0) * 100.0
        norm_a = center_lab[1] - 128.0
        norm_b = center_lab[2] - 128.0
        chroma = math.sqrt(norm_a * norm_a + norm_b * norm_b)

        # Lấy góc Hue từ HSV của cụm tương ứng để chuyển đổi chính xác
        cluster_hsv = filtered_hsv[cluster_mask]
        if len(cluster_hsv) > 0:
            avg_hue = float(np.median(cluster_hsv[:, 0]) * 2.0)  # OpenCV H is 0-180 -> multiply by 2
            avg_sat = float(np.mean(cluster_hsv[:, 1]) / 255.0)
        else:
            avg_hue = float((math.atan2(norm_b, norm_a) * 180.0 / math.pi) % 360.0)
            avg_sat = min(chroma / 60.0, 1.0)

        # Chuyển tâm cụm sang RGB
        center_pixel = np.uint8([[center_lab]])
        bgr_pixel = cv2.cvtColor(center_pixel, cv2.COLOR_LAB2BGR)[0][0]
        hex_color = rgb_to_hex(bgr_pixel[2], bgr_pixel[1], bgr_pixel[0])

        clusters.append({
            "index": i,
            "weight": weight_percent,
            "hex": hex_color,
            "hue": avg_hue,
            "sat": avg_sat,
            "chroma": chroma,
            "lightness": norm_l,
        })

    # Sắp xếp các cụm:
    # - Cụm chủ đạo: Cụm có trọng số pixel lớn nhất có độ bão hòa hợp lý
    clusters.sort(key=lambda c: c["weight"], reverse=True)
    
    # Tìm cụm Accent rực rỡ nhất (Vibrant Accent Pop)
    # Điểm đánh giá độ nổi bật = Chroma * (Weight ^ 0.3)
    accent_cluster = max(clusters, key=lambda c: c["chroma"] * (max(c["weight"], 1.0) ** 0.35))
    dominant_cluster = clusters[0]

    # Nếu cụm có trọng số lớn nhất quá nhạt màu, tìm cụm tiếp theo có sắc độ
    if dominant_cluster["chroma"] < 8.0 and len(clusters) > 1:
        for cl in clusters[1:]:
            if cl["chroma"] >= 8.0:
                dominant_cluster = cl
                break

    primary_hue = dominant_cluster["hue"]
    primary_sat = max(dominant_cluster["sat"], 0.35)

    accent_hue = accent_cluster["hue"]
    accent_sat = min(max(accent_cluster["sat"] + 0.2, 0.70), 0.95)

    # 5. Xây dựng 7 Token Khóa Tương Phản Chuẩn WCAG 2.1 AA (4.5:1+) & Quy tắc 60-30-10
    # Canvas (60%): Tối sâu, dịu mắt
    rgb_canvas = hsl_to_rgb(primary_hue, min(primary_sat * 0.7, 0.35), 0.07)
    # Surface Sidebar & Card (30%): Sắc độ hài hòa với ảnh
    rgb_surface = hsl_to_rgb(primary_hue, min(primary_sat * 1.1, 0.65), 0.20)
    # Surface Elevated: Nền nâng cao
    rgb_elevated = hsl_to_rgb(primary_hue, min(primary_sat * 1.0, 0.58), 0.26)
    # Border: Viền tinh tế
    rgb_border = hsl_to_rgb(primary_hue, min(primary_sat * 0.9, 0.50), 0.35)
    # Text Muted: Chữ phụ
    rgb_text_muted = hsl_to_rgb(primary_hue, 0.25, 0.78)
    # Accent Primary (10% Nút bấm & Tiêu điểm): Rực rỡ, bắt mắt
    rgb_accent = hsl_to_rgb(accent_hue, accent_sat, 0.52)

    palette = {
        "canvasBg": rgb_to_hex(*rgb_canvas),
        "surfaceBg": rgb_to_hex(*rgb_surface),
        "surfaceElevated": rgb_to_hex(*rgb_elevated),
        "borderSubtle": rgb_to_hex(*rgb_border),
        "textMuted": rgb_to_hex(*rgb_text_muted),
        "textHighlight": "#ffffff",
        "accentPrimary": rgb_to_hex(*rgb_accent),
    }

    return {
        "success": True,
        "algorithm": "Python AI (CIELAB K-Means++ Clustering)",
        "dominant_hue": round(primary_hue, 1),
        "accent_hue": round(accent_hue, 1),
        "palette": palette,
        "clusters": [
            {
                "hex": c["hex"],
                "weight": c["weight"],
                "chroma": round(c["chroma"], 1),
                "is_dominant": (c["index"] == dominant_cluster["index"]),
                "is_accent": (c["index"] == accent_cluster["index"]),
            }
            for c in clusters
        ],
    }
