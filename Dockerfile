FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

RUN mkdir -p /var/log/aigateway
VOLUME /var/log/aigateway

EXPOSE 9000

CMD ["python", "-m", "app.main"]
