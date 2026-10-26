"""
routes_theme.py - API router trích xuất theme màu bằng AI
"""

import os
import base64
import urllib.request
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from theme_color_extractor import extract_theme_from_image_bytes

router = APIRouter(tags=["Theme"])


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
                local_path = os.path.join(r"c:\Users\DELL\OneDrive\Documents\pomo-do\pomo-do-frontend\public", clean_rel)
                if os.path.exists(local_path):
                    with open(local_path, "rb") as f:
                        return extract_theme_from_image_bytes(f.read())

            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                contents = response.read()
                return extract_theme_from_image_bytes(contents)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Không thể tải ảnh từ URL: {str(e)}")

    raise HTTPException(status_code=400, detail="Vui lòng cung cấp file ảnh, image_base64 hoặc image_url")
