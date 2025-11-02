FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 8501
CMD ["streamlit","run","app/main.py","--server.address=0.0.0.0","--server.port","8501"]