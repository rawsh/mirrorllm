FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ADD local_reward.py .