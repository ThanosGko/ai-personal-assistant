FROM python:3.9-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OPENROUTER_API_KEY=""

EXPOSE 8888

CMD ["python", "main.py"]