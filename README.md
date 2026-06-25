# 🎬 Vid2Sim 🤖: Realistic and Interactive Simulation from Video for Urban Navigation
> [Ziyang Xie](https://ziyangxie.site/), [Zhizheng Liu](https://scholar.google.com/citations?user=Asc7j9oAAAAJ&hl=en), [Zhenghao Peng](https://pengzhenghao.github.io/), [Wayne Wu](https://wywu.github.io/), [Bolei Zhou](https://boleizhou.github.io/)
>
> [![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2501.06693)
> [![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://metadriverse.github.io/vid2sim/)

Vid2Sim is a novel framework that converts monocular videos into photorealistic and physically interactive simulation environments for training embodied agents with minimal sim-to-real gap.

<p align="center">
  <img src="./assets/teaser.png" width="100%">
</p>


## 🚧 Dockerized Quickstart

```bash
# Clone the repository
git clone https://github.com/serve-robotics/Vid2Sim.git --recursive
cd Vid2Sim

# 1) Build the Docker image
cd docker
./build.sh

# 2) Start an interactive container shell
./run.sh

# 3) Inside the container: extract frames from videos
# Input:  /workspace/data/videos (contains .mp4/.mov/... files, recursively)
# Output: /workspace/output/images/<relative_path>/<video_name>/%04d.jpg
bash scripts/extract_frames_recursive.sh /workspace/data/videos /workspace/output/images

# 4) Inside the container: generate masks from extracted frames
# This supports nested image folders under /workspace/output/images.
./src/vid2sim_recon/generate_mask.sh /workspace/output
```

To use lower-memory mask settings:

```bash
VID2SIM_MASK_CHUNK_SIZE=1 VID2SIM_MASK_SIZE=720 VID2SIM_DISABLE_LONG_TERM=1 \
  ./src/vid2sim_recon/generate_mask.sh /workspace/output
```

## 🎥 Reconstruct the simulation envs from videos
Vid2Sim transforms monocular videos into simulation environments by reconstructing the scene geometry and appearance. The generated environments preserve real-world diversity and visual fidelity, providing minimal sim-to-real gap for agent training.


👉 To get started, follow the reconstruction guide in [vid2sim_recon](src/vid2sim_recon/README.md) to reconstruct the simulation environment from video.

## 🤖 Train the Agent in Real-to-Sim Environments

After the environment is reconstructed, Vid2Sim translates the real-to-sim environments into a interactive environment with both realistic visual appearance and physical collision to train the agent in diverse situations.

👉 To set up the environment and launch RL training, refer to [vid2sim_rl](src/vid2sim_rl/README.md). 


## 📦 Repository Structure
```
Vid2Sim/
├── data/ # Source data
├── src/
│   ├── vid2sim_recon/ # Reconstruct the simulation environment from video
│   ├── vid2sim_rl/ # Train the agent in real-to-sim environments
├── tools/ # Tools scripts
├── README.md # This file
```


## 📚 Vid2Sim Dataset

The Vid2Sim dataset includes 30 high-quality real-to-sim simulation environments reconstructed from video clips sourced from 9 web videos. Each clip includes 15 seconds of forward-facing video recorded at 30 fps, providing 450 frames per scene for environment reconstruction and simulation.  

We provide the source [video data](https://drive.google.com/drive/folders/1jGmKxZL6hKvjCg6qhM9wmW1_HjMwCUGa?usp=sharing), and [interactive Unity environments](https://drive.google.com/drive/folders/1LCruqb6M3mCgsjaqI1ON6WVoZ-9CmQDY?usp=sharing) for agent training.

<p align="center">
  <img src="./assets/dataset.png" width="100%">
</p>

## Citation 📝

If you find this work useful in your research, please consider citing:

```bibtex
@article{xie2024vid2sim,
  title={Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation},
  author={Xie, Ziyang and Liu, Zhizheng and Peng, Zhenghao and Wu, Wayne and Zhou, Bolei},
  journal={CVPR},
  year={2025}
}
```
