"""
routes_theme.py - API router trích xuất theme màu bằng AI
"""

import os
import base64
import ipaddress
import socket
import urllib.request
from urllib.parse import urlparse
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from theme_color_extractor import extract_theme_from_image_bytes

router = APIRouter(tags=["Theme"])


def is_safe_external_url(url: str) -> bool:
    """Kiểm tra ngăn chặn tấn công SSRF: chặn truy cập vào private IP, loopback và metadata cloud."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        if hostname.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return False

        addr_infos = socket.getaddrinfo(hostname, None)
        for addr in addr_infos:
            ip_str = addr[4][0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        return True
    except Exception:
        return False


class AnalyzeThemeRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None


@router.post("/api/analyze-theme")
async def analyze_theme_endpoint(request: Request):
    """
    Endpoint phân tích màu sắc giao diện bằng AI (CIELAB K-Means++).
    Nhận file tải lên trực tiếp (multipart/form-data), base64 hoặc URL ảnh (JSON).
    """
    content_type = request.headers.get("content-type", "")

    # 1. Tải file trực tiếp qua multipart/form-data
    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
            uploaded_file = form.get("file")
            if uploaded_file and hasattr(uploaded_file, "read"):
                contents = await uploaded_file.read()
                return extract_theme_from_image_bytes(contents)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Lỗi đọc file tải lên: {str(e)}")

    # 2. Nhận JSON payload (base64 hoặc URL)
    try:
        body = await request.json()
    except Exception:
        body = {}

    image_base64 = body.get("image_base64")
    if image_base64:
        try:
            b64_data = str(image_base64)
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            contents = base64.b64decode(b64_data)
            return extract_theme_from_image_bytes(contents)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Lỗi giải mã base64: {str(e)}")

    image_url = body.get("image_url")
    if image_url:
        try:
            url = str(image_url).strip()
            if url.startswith("/wallpapers/") or url.startswith("wallpapers/"):
                clean_rel = url.lstrip("/\\").replace("/", os.sep)
                frontend_public = os.getenv(
                    "FRONTEND_PUBLIC_DIR",
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pomo-do-frontend", "public"))
                )
                local_path = os.path.abspath(os.path.join(frontend_public, clean_rel))
                # Ngăn chặn Directory Traversal ra ngoài thư mục static hợp lệ
                if local_path.startswith(os.path.abspath(frontend_public)) and os.path.exists(local_path):
                    with open(local_path, "rb") as f:
                        return extract_theme_from_image_bytes(f.read())

            # Chặn triệt để tấn công SSRF vào mạng nội bộ hoặc IP Private
            if not is_safe_external_url(url):
                raise HTTPException(
                    status_code=400,
                    detail="URL ảnh không hợp lệ hoặc trỏ đến dải địa chỉ nội bộ bị cấm (SSRF Protection)."
                )

            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                # Giới hạn kích thước ảnh tải về tối đa 15MB
                contents = response.read(15 * 1024 * 1024 + 1)
                if len(contents) > 15 * 1024 * 1024:
                    raise HTTPException(status_code=400, detail="Kích thước ảnh vượt quá giới hạn 15MB cho phép.")
                return extract_theme_from_image_bytes(contents)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Không thể tải ảnh từ URL: {str(e)}")

    raise HTTPException(status_code=400, detail="Vui lòng cung cấp file ảnh, image_base64 hoặc image_url")
