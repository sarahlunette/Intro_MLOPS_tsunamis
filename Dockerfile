FROM python:3.10-slim

WORKDIR /apps

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "apps.main_functional_2:api", "--host", "127.0.0.1", "--port", "8000"]
