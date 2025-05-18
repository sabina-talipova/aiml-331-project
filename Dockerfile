FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip bash nano && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install Flask

CMD ["python3", "app.py"]
