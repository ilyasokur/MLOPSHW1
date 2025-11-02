FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg PYTHONPATH=/workspace
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python","app/kafka_worker.py"]