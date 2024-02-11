FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install . --no-cache-dir

EXPOSE 5000

CMD [ "streamlit", "run", "app.py" ]

