FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
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
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel

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

# Install reconstruction dependencies.
# Build CUDA extensions without PEP517 isolation so torch env vars are honored.
RUN TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0" pip install --no-build-isolation -e submodules/vid2sim-rasterizer
RUN pip install -e submodules/vid2sim-deva-segmentation
RUN pip install -r submodules/Grounded-SAM-2/grounding_dino/requirements.txt
RUN pip install segment-anything
RUN pip install segment-anything-hq
RUN TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0" pip install --no-build-isolation -e submodules/Grounded-SAM-2/grounding_dino
RUN TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0" pip install --no-build-isolation -e submodules/simple-knn

# Install RL dependencies
RUN pip install -r src/vid2sim_rl/requirements.txt
RUN pip install -e submodules/ml-agents/ml-agents-envs
RUN pip install -e submodules/ml-agents/ml-agents

RUN pip install -e submodules/r3m

CMD ["/bin/bash"]
