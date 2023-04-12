FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Define build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG DEBCONF_NOWARNINGS="yes"
ARG PIP_ROOT_USER_ACTION=ignore

# install packages
RUN apt-get update && apt-get install -y \
    nano \
    git \
    g++ \
    gcc \
    libgl1 \
    libglib2.0-0

# update pip package
RUN python3 -m pip install --upgrade pip

# install pytorch geometric for GPU
RUN python3 -m pip install --no-cache-dir -f https://data.pyg.org/whl/torch-1.11.0+cu113.html\
    torch-scatter==2.0.9 \
    torch-sparse==0.6.15 \
    torch-cluster==1.6.0 \
    torch-spline-conv==1.2.1 \
    torch-geometric==2.1.0.post1

# install detectron2 (only used for for nms_rotated)
RUN python3 -m pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# install additional packages
COPY ./requirements.txt /app/
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# copy src code and package installation configurations 
COPY ./src /app/src
COPY ./pyproject.toml ./setup.cfg ./setup.py /app/

# copy test code 
COPY ./test /app/test

# set working directory and entry point 
WORKDIR /app
ENTRYPOINT [ "/bin/bash"]


