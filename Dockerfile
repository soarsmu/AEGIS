FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get -y install --no-install-recommends \
    software-properties-common \
    git \
    vim \
    htop \
    tmux \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.7
RUN apt-get -y update && \
    apt-get -y install python3.7 python3-pip python3.7-distutils python3.7-dev && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

WORKDIR /root/AEGIS

# Upgrade pip and setuptools
RUN pip3 install --no-cache-dir --upgrade pip setuptools

# Install Python packages
RUN pip3 install --no-cache-dir \
    tensorflow==1.13.1 \
    tflearn==0.5.0 \
    protobuf==3.20.3 \
    tqdm \
    pandas \
    scikit-learn \
    scikit-optimize==0.9.0 \
    matplotlib

# Install Z3 solver
RUN wget https://github.com/Z3Prover/z3/releases/download/z3-4.14.1/z3_solver-4.14.1.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl && \
    pip3 install --no-cache-dir z3_solver-4.14.1.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl && \
    rm z3_solver-4.14.1.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl