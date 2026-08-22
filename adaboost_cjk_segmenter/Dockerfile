FROM python:3.12.11-slim-bookworm

WORKDIR /

COPY . /

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "-u", "train.py"]
