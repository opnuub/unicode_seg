FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-distutils \
        build-essential g++ make \
        libopenblas-dev liblapack-dev \
        pkg-config libicu-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY . /

RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

ENTRYPOINT ["python3", "train_cnn.py"]
