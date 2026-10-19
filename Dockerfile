FROM python:3.11-slim

WORKDIR /app

# Tối ưu RAM cho môi trường đám mây giới hạn (Render Free 512MB)
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Cài đặt các thư viện hệ thống cần thiết cho MediaPipe và OpenCV trên Linux
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

# Copy toàn bộ mã nguồn và models (.task, .pt)
COPY . .

# Render tự động cấp biến $PORT, uvicorn sẽ lắng nghe cổng này
ENV PORT=8001
EXPOSE 8001

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8001}"]
