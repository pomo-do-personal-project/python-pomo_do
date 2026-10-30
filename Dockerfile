FROM python:3.11-slim

WORKDIR /app

# Tối ưu RAM cho môi trường đám mây giới hạn (Render Free 512MB)
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Cài đặt các thư viện hệ thống cần thiết cho MediaPipe và OpenCV trên Linux
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgles2 \
    libegl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt dependencies Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn và models (.task, .onnx, .pt)
COPY . .

# Tạo người dùng non-root và phân quyền thư mục làm việc để tăng cường an ninh container
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Render tự động cấp biến $PORT, uvicorn sẽ lắng nghe cổng này
ENV PORT=8001
EXPOSE 8001

# Healthcheck định kỳ kiểm tra trạng thái phản hồi của service AI
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request, os; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8001') + '/').read()" || exit 1

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8001}"]
