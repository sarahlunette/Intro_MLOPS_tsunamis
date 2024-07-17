FROM python:3.10-slim

WORKDIR /apps

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main_functional:api", "--host", "127.0.0.1", "--port", "8000"]
