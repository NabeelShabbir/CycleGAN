FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY app/ /app/
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["python", "main.py"]
