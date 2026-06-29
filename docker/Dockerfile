FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    sudo \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python-is-python3 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel

# Install torch before COPY so this layer is cached independently of source
# changes. Any edit to local files invalidates COPY and everything after it,
# so keeping the large (~3 GB) torch download above COPY means it only
# re-runs when the torch version changes, not on every file edit.
# cu128 wheel is required: the default PyPI torch ships cu13, which mismatches
# this 12.8 base image and breaks CUDA extension compilation.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace

# Copy the entire repository into the Docker image
COPY . .

RUN pip install -e .

# Preload DEVA propagation checkpoint expected by deva/inference/eval_args.py.
RUN mkdir -p submodules/ckpts/deva \
    && curl -L --fail https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth \
       -o submodules/ckpts/deva/DEVA-propagation.pth

# Preload GroundingDINO defaults expected by deva/ext/ext_eval_args.py.
RUN cp submodules/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        submodules/ckpts/deva/GroundingDINO_SwinT_OGC.py \
     && curl -L --fail https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
         -o submodules/ckpts/deva/groundingdino_swint_ogc.pth

# Install non-CUDA reconstruction dependencies.
# ninja, hatchling, and the three CUDA extensions (vid2sim-rasterizer,
# groundingdino, simple-knn) are intentionally NOT built here: docker build
# has no GPU access so torch's import intermittently segfaults (SIGSEGV/-11)
# even before nvcc runs. Run docker/setup_extensions.sh inside the container
# (via ./docker/run.sh) to compile them once against the live GPU — the
# compiled .so files persist in the mounted /workspace.
RUN pip install ninja hatchling editables
RUN pip install --no-build-isolation -e submodules/vid2sim-deva-segmentation
RUN pip install -r submodules/Grounded-SAM-2/grounding_dino/requirements.txt
RUN pip install segment-anything
RUN pip install segment-anything-hq

# Install RL dependencies
RUN pip install -r src/vid2sim_rl/requirements.txt
RUN pip install imageio imageio-ffmpeg
RUN pip install -e submodules/ml-agents/ml-agents-envs
RUN pip install -e submodules/ml-agents/ml-agents

RUN pip install -e submodules/r3m

CMD ["/bin/bash"]
