# setting up x86_64
FROM --platform=linux/amd64 ubuntu:22.04

# prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

ENV USER=root

# install dependencies and Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    graphviz \
    libgraphviz-dev \
    locales \
    libgl1 \
    libglib2.0-0 \
    && locale-gen en_US.UTF-8

# Hailo's CLI assumes a UTF-8 locale; unset locale on a bare Ubuntu image
# is a common cause of crashes on first run
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# create a working directory
WORKDIR /app

# copy the hailo dataflow compiler wheel into the image
COPY hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl .
COPY hailo_model_zoo-2.19.0-py3-none-any.whl .

RUN python3 -m venv /opt/hailo_venv
ENV PATH="/opt/hailo_venv/bin:$PATH"

# instal the hailo dataflow compiler
RUN pip install hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl
RUN pip install hailo_model_zoo-2.19.0-py3-none-any.whl

# set up standard environment variables
ENV PYTHONPATH=$PYTHONPATH:/app

# default command to enter a bash shell
CMD ["/bin/bash"]