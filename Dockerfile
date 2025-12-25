# 使用 Google 官方優化過的 TF 映像檔
# FROM asia-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12.py310:latest
FROM python:3.12-slim

WORKDIR /app

# 安裝 gcc，因為有些套件編譯需要
RUN apt-get update && apt-get install -y --no-install-recommends gcc libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train/train.py .

# 這裡不寫 CMD，因為我們會透過 Vertex Pipeline 傳入參數來呼叫它
ENTRYPOINT ["python", "train.py"]