# starting point of every dockerfile, base image
# pinned to linux/amd64: the Hailo Dataflow Compiler wheel is x86_64-only,
# so this must be forced on arm64 hosts (e.g. Apple Silicon)
FROM --platform=linux/amd64 nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# install dependencies and Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    graphviz \
    libgraphviz-dev \
    locales \
    && locale-gen en_US.UTF-8

# Hailo's CLI assumes a UTF-8 locale; unset locale on a bare Ubuntu image
# is a common cause of crashes on first run
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Expose all GPUs automatically when --gpus all is passed at docker run time.
# These vars are read by nvidia-container-toolkit and have zero effect on hosts
# without it (macOS, CPU-only Linux).
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=12.1"

# create a working directory
WORKDIR /app

# copy the hailo dataflow compiler wheel into the image.
# requirements.txt is intentionally copied later (just before its install) so
# that edits to it don't invalidate the expensive torch/hailo layers below.
COPY hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl .

# Hailo's SDK locates its bundled native tools (hailo_tools/build/compiler)
# by checking whether it was installed into a directory literally named
# "site-packages". Ubuntu/Debian's system Python renames this to
# "dist-packages", which breaks that detection and makes the compiler step
# fail with a confusing "expected str, bytes or os.PathLike, not NoneType"
# error. A venv always uses standard "site-packages" naming, so install there.
RUN python3 -m venv /opt/hailo_venv
ENV PATH="/opt/hailo_venv/bin:$PATH"

# Install the Hailo DFC (does not depend on torch).
RUN pip install hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl

# Install a CUDA (cu121) build of torch BEFORE the rest of the requirements.
# cu121 wheels are only published up to 2.5.1, and this build runs fine on the
# CUDA 12.1 base image. Installing it first means ultralytics (in requirements)
# sees torch already satisfied and won't pull the CPU-only wheel over it.
RUN pip install \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121

# Copied here (after torch/hailo) so requirements edits rebuild only this layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

# set up standard environment variables
ENV PYTHONPATH=$PYTHONPATH:/app

# default command to enter a bash shell
CMD ["/bin/bash"]