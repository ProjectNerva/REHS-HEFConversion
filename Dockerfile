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

# copy the hailo dataflow compiler wheel into the image
COPY hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl .
COPY requirements.txt .

# Hailo's SDK locates its bundled native tools (hailo_tools/build/compiler)
# by checking whether it was installed into a directory literally named
# "site-packages". Ubuntu/Debian's system Python renames this to
# "dist-packages", which breaks that detection and makes the compiler step
# fail with a confusing "expected str, bytes or os.PathLike, not NoneType"
# error. A venv always uses standard "site-packages" naming, so install there.
RUN python3 -m venv /opt/hailo_venv
ENV PATH="/opt/hailo_venv/bin:$PATH"

# Install the Hailo DFC — this pulls in CPU-only torch as a transitive dependency.
RUN pip install hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl

# Detect the exact torch version DFC installed, then swap in the matching CUDA
# build from PyTorch's cu121 index. --no-deps replaces only the torch wheel
# itself without re-resolving any of DFC's other pinned dependencies.
RUN TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])") && \
    echo "Replacing CPU torch ${TORCH_VER} with CUDA build (cu121)..." && \
    pip install --no-deps \
        "torch==${TORCH_VER}+cu121" \
        --index-url https://download.pytorch.org/whl/cu121

RUN pip install -r requirements.txt

# set up standard environment variables
ENV PYTHONPATH=$PYTHONPATH:/app

# default command to enter a bash shell
CMD ["/bin/bash"]