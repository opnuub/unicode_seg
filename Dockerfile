FROM ubuntu:22.04

FROM python:3.9-bookworm

RUN apt-get -y update && apt-get -y upgrade
RUN apt-get -y install pkg-config 
RUN apt-get -y install libicu-dev

WORKDIR /root

COPY . /root

RUN pip install -r /root/requirements.txt

ENTRYPOINT ["python", "train_thai.py"]
