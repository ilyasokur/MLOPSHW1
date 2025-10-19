FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY run_pipeline.sh .
RUN chmod +x run_pipeline.sh

RUN mkdir -p /mnt/input /mnt/output /mnt/work

ENTRYPOINT ["/bin/bash", "run_pipeline.sh"]