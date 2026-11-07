FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY run_dashboard.sh run_alerts.sh ./

# Candle history is written here; mount a volume to keep it across restarts.
RUN mkdir -p data out
ENV CANDLE_DB=data/candles.db

# PaaS providers inject $PORT; default 12000 for local docker run.
ENV PORT=12000
EXPOSE 12000

CMD ["python", "dashboard.py"]
