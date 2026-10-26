"""
routes_session.py - Router phân tích phiên học và giám sát tài nguyên bộ nhớ
"""

import os
from fastapi import APIRouter

router = APIRouter(tags=["Session & Memory"])


def get_current_ram_mb() -> float:
    """Lấy số MB RAM thực tế (RSS) đang sử dụng."""
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


@router.get("/memory")
@router.get("/ai/memory")
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


@router.post("/analyze_session")
@router.post("/ai/analyze_session")
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
    advice = (
        "Để điện thoại xa tầm tay ở phiên kế tiếp."
        if phone_cnt > 0
        else "Nên rửa mặt hoặc vươn vai nếu thấy buồn ngủ."
        if sleep_cnt > 0
        else "Tiếp tục duy trì phong độ tập trung này!"
    )

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
