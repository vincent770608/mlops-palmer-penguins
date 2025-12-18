# 使用 Google 官方優化過的 TF 映像檔
FROM asia-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

# 這裡不寫 CMD，因為我們會透過 Vertex Pipeline 傳入參數來呼叫它
ENTRYPOINT ["python", "train.py"]