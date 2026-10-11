FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY frontend/ frontend/
COPY configs/ configs/
COPY data/ data/
COPY models/ models/
COPY deployment/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860

CMD ["./entrypoint.sh"]
