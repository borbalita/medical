FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install . --no-cache-dir

COPY . .

EXPOSE 5000

CMD [ "python", "app.py" ]