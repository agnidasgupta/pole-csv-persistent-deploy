FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /app/requirements.txt

COPY pole_csv_persistent_server.py /app/pole_csv_persistent_server.py
COPY best_model.pt /app/best_model.pt

EXPOSE 8000

CMD ["python3", "/app/pole_csv_persistent_server.py", \
     "--model-path", "/app/best_model.pt", \
     "--device", "cuda", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--positive-only-output", "1"]
