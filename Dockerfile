FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-16.py310:latest

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        pkg-config libicu-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY . /

RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

ENTRYPOINT ["python3", "train.py"]
