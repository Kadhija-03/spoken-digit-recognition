FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir flask tensorflow librosa numpy soundfile

EXPOSE 5000

CMD ["python", "app.py"]