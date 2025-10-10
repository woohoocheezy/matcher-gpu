# Dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda config --set report_errors false && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "faiss_env", "/bin/bash", "-c"]

COPY build_index.py .
COPY api_server /app/api_server
COPY faiss_server /app/faiss_server
