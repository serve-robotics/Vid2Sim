# Vid2Sim Demo — Pre-built Unity env + random-policy inference

Renders a photorealistic Gaussian Splatting urban scene in Unity, drives a robot
with a forward-biased random policy, and saves RGB videos. No trained model needed.

## What this demo is

[Vid2Sim](https://metadriverse.github.io/vid2sim/) (CVPR 2025) turns a monocular
walking video of a real street into an interactive, photorealistic simulator. Each
scene is reconstructed as a hybrid **3D Gaussian Splatting** (for appearance) +
**mesh** (for collision/geometry) representation, packaged as a standalone Unity
build that exposes a robot agent over Unity's **ml-agents** gRPC interface. The point
is sim-to-real RL: train a visual navigation policy in these video-reconstructed
scenes and transfer it to a real robot.

**The two environment sets you're downloading** (30 scenes each, one Unity build per
scene):

- **PointNav** (`envs/static/00–29`) — *point-goal navigation* in **static** scenes.
  The robot must drive from its start to a goal position while avoiding fixed
  obstacles (the GLB props — bollards, hydrants, mailboxes — randomly placed each
  episode). No moving agents.
- **SocialNav** (`envs/dynamic/00–29`) — *social navigation* in **dynamic** scenes.
  Same goal-reaching task, but with moving pedestrians/agents the robot must avoid,
  so the policy has to handle dynamic collision avoidance. (Enabled via
  `enable_dynamic_agent`; this demo leaves it off and uses a PointNav scene.)

**How they're normally trained.** In the full Vid2Sim pipeline a **SAC** policy
(stable-baselines3, see `config/rgb-30envs-pointnav.yaml`) trains across all 30
scenes in parallel for millions of steps. At each step the agent receives a stack of
recent **RGB camera frames** (a small CNN encodes them) plus a short **vector**
observation (goal direction/distance, collision flags), and outputs a continuous
**[throttle, steering]** action. Reward shapes the behavior: progress toward the goal
(`distanceRewardMultiplier`) and a large `goalReward` for reaching it, minus a
per-step `timePenalty`, a `collisionPenalty`, and a steering-smoothness term. The
result is a checkpoint (`.zip`) that navigates from pixels.

**What this demo does instead.** It skips training entirely — there is no checkpoint.
`demo.py` drives the *same* observation/action interface with a **forward-biased
random policy** (`action[0] = 0.8` throttle, random steering), purely to exercise the
full pipeline end-to-end: load a reconstructed scene, render the Gaussian-Splatting
view into the agent's camera, step the ml-agents loop, and record the RGB frames to
MP4. So it showcases the **photorealistic rendering + sim interface**, not learned
navigation — expect the robot to wander and bump into things, not reach goals. To see
*trained* behavior you'd load a SAC checkpoint and query its policy in the step loop
instead of sampling random actions.

> **Roadmap note (future direction, not yet implemented).** This demo runs on the
> upstream Vid2Sim **Unity** build. Going forward the plan is to move the rendering
> engine to **Unreal Engine**, replacing Unity as the simulator front-end. From
> Unreal we can export the reconstructed scenes as **USD/USDZ** and bring them into
> **Isaac Sim / Isaac Lab**, so the same video-reconstructed environments can be used
> for RL training inside the NVIDIA Omniverse stack (consistent with the rest of the
> SRX simulation workspace) rather than the Unity ml-agents path used here.
>
> **Richer reconstruction from our own robots.** Upstream Vid2Sim reconstructs a
> scene from a single monocular walking video. Our robots carry a **multi-camera rig
> (front, left, right, rear)**, so once we feed that footage into the reconstruction
> pipeline we get much wider coverage per pass — closer to a full surround view of
> the street — yielding a more complete, higher-fidelity scene. Those reconstructed
> scenes then become the simulation environments we train and evaluate policies in.

## System this was set up on

Reference host (the demo was verified working here):

| Component               | Value                                            |
|-------------------------|--------------------------------------------------|
| OS                      | Ubuntu 24.04.4 LTS                               |
| Kernel                  | 6.8.0-124-generic                                |
| GPU                     | NVIDIA GeForce RTX 4090, 24 GB                    |
| GPU compute capability  | **8.9** (Ada Lovelace, `sm_89`)                  |
| NVIDIA driver           | 595.71.05                                        |
| CUDA (host `nvcc`)      | 12.8 (V12.8.61)                                  |
| CUDA max (driver)       | 13.2 (per `nvidia-smi`)                          |
| Docker                  | 28.1.1                                           |
| nvidia-container-toolkit| 1.19.1                                           |
| Base image              | `nvidia/cuda:12.8.1-devel-ubuntu22.04`           |
| PyTorch wheel           | `cu128` (from download.pytorch.org)              |

The host needs: a recent NVIDIA driver, Docker with the **nvidia-container-toolkit**
(so `docker compose`'s `deploy.resources...devices: nvidia` works), and the NVIDIA
PRIME/GLVND libs for the host-side Unity launch (`__NV_PRIME_RENDER_OFFLOAD`,
`__GLX_VENDOR_LIBRARY_NAME=nvidia`).

### Building for a different GPU (e.g. RTX 6000 Pro / Blackwell)

The CUDA extensions are compiled for a specific GPU architecture. They're built at
runtime by `docker/setup_extensions.sh`, which currently pins this host's arch:
`TORCH_CUDA_ARCH_LIST="8.9"` (RTX 4090). **To run in a VM with an RTX 6000 Pro
(Blackwell, `sm_120` / compute capability 12.0), change all three lines in
`docker/setup_extensions.sh` from `8.9` to `12.0`** before running it:

```bash
# docker/setup_extensions.sh — for RTX 6000 Pro Blackwell
TORCH_CUDA_ARCH_LIST="12.0" pip install --no-build-isolation -e submodules/vid2sim-rasterizer
TORCH_CUDA_ARCH_LIST="12.0" pip install --no-build-isolation -e submodules/Grounded-SAM-2/grounding_dino
TORCH_CUDA_ARCH_LIST="12.0" pip install --no-build-isolation -e submodules/simple-knn
```

Notes for the Blackwell VM:
- `sm_120` requires CUDA ≥ 12.8 — the `nvidia/cuda:12.8.1` base image and the
  `cu128` torch wheel already satisfy this, so the Dockerfile itself is unchanged.
- The `.so` files compiled in `setup_extensions.sh` are arch-specific and persist
  in the mounted `/workspace`. If you reuse a `/workspace` that was previously
  built on the 4090 (`sm_89`), delete the stale builds and re-run the script, or
  set `TORCH_CUDA_ARCH_LIST="8.9;12.0"` to produce a fat binary that runs on both.
- The host VM still needs its own NVIDIA driver new enough for the RTX 6000 Pro and
  the nvidia-container-toolkit installed.
- Confirm the target arch in the VM with: `nvidia-smi --query-gpu=name,compute_cap --format=csv`.

## Why the split (Docker Python + host Unity)

Inside the Docker container, GLX routes to software rendering (nouveau), so the
Gaussian Splatting shaders render blank. Unity must therefore run on the **host**
with NVIDIA PRIME offload, while the Python ml-agents driver runs in the container.
`docker-compose.yml` uses `network_mode: host`, so container port 5004 and host
port 5004 are the same socket — they connect over localhost.

No `xhost`/X11 forwarding is needed: Unity renders on the host's own display, not
from inside the container. (`docker-compose.yml` still mounts `/tmp/.X11-unix` and
sets `DISPLAY` from a previous in-container approach, but the working flow doesn't
rely on them — ignore any `xhost: command not found` noise.)

## One-time asset setup (Google Drive → target dirs)

The Unity environments and obstacle meshes are NOT in the repo — they come from the
project's Google Drive, which has three folders: **mesh**, **PointNav**, **SocialNav**.

Download from the **interactive Unity environments** folder (the demo uses these):
https://drive.google.com/drive/folders/1LCruqb6M3mCgsjaqI1ON6WVoZ-9CmQDY?usp=sharing

(The separate source-video-data folder —
https://drive.google.com/drive/folders/1jGmKxZL6hKvjCg6qhM9wmW1_HjMwCUGa?usp=sharing
— is only needed for scene *reconstruction*, not for running this demo.)

**Disk space:** budget **at least ~150 GB** free. The steady-state footprint is
only ~71 GB (merged assets ~23 GB — PointNav `envs/static` ≈ 11 GB, SocialNav
`envs/dynamic` ≈ 12 GB, ~364 MB per scene, mesh ≈ 57 MB — plus the ~48 GB Docker
image). But **peak** usage during setup is far higher: `rsync` copies rather than
moves, so you transiently hold the data three times (zips + extracted parts +
merged target, ≈ 70 GB of assets alone), on top of the Docker build cache and the
~3 GB torch download. That stack-up is why ~150 GB is the safe number. It drops
back to ~71 GB once you delete the download parts (see "Reclaim space" below). If
you only need PointNav for this demo, skip SocialNav to save ~12 GB.

Google Drive splits large folders into multiple numbered zip parts on download, e.g.
`PointNav-<timestamp>-3-001.zip` … `-006.zip`. Each part extracts to its own
directory (`PointNav-<timestamp>-3-00N/PointNav/`) holding a *subset* of the scene
folders (`00`, `01`, …). You must merge all parts into one target directory.

Target mapping (relative to `src/vid2sim_rl/`):

| Drive folder | Target dir                 | Result                |
|--------------|----------------------------|-----------------------|
| PointNav     | `envs/static/`             | scenes `00`–`29` (30) |
| SocialNav    | `envs/dynamic/`            | scenes `00`–`29` (30) |
| mesh         | hardcoded path (see below) | obstacle `.glb` files |

Each scene folder contains `env.x86_64`, `UnityPlayer.so`, and `env_Data/`.

Merge the extracted parts with `rsync` (it overlays each part's scene folders into
the single target without clobbering the others). From `~/Downloads`:

```bash
# PointNav parts -> envs/static/
mkdir -p ~/code/srx/Vid2Sim/src/vid2sim_rl/envs/static
rsync -a PointNav-*/PointNav/  ~/code/srx/Vid2Sim/src/vid2sim_rl/envs/static/

# SocialNav parts -> envs/dynamic/
mkdir -p ~/code/srx/Vid2Sim/src/vid2sim_rl/envs/dynamic
rsync -a SocialNav-*/SocialNav/  ~/code/srx/Vid2Sim/src/vid2sim_rl/envs/dynamic/

# mesh -> the developer path hardcoded in the binary (see Gotchas)
mkdir -p /home/ziyangxie/Code/Video2Sim-RL/envs/mesh/sample_obj
rsync -a mesh-*/mesh/sample_obj/  /home/ziyangxie/Code/Video2Sim-RL/envs/mesh/sample_obj/
```

The trailing slashes on the source paths matter — they tell `rsync` to copy the
*contents* into the target rather than nesting another folder. Verify after merge:
`ls envs/static | wc -l` and `ls envs/dynamic | wc -l` should each print **30**.

**Make the Unity binaries executable.** Google Drive's zip download strips the
executable bit, so straight after the merge the `env.x86_64` files are not runnable
— launching one fails with `Permission denied`. Restore the bit on every scene in
both env trees:

```bash
chmod +x ~/code/srx/Vid2Sim/src/vid2sim_rl/envs/static/*/env.x86_64
chmod +x ~/code/srx/Vid2Sim/src/vid2sim_rl/envs/dynamic/*/env.x86_64
```

(`docker/setup_extensions.sh` is already committed with its `+x` bit, so it doesn't
need this — only the Drive-sourced Unity binaries do.)

**Reclaim space afterward.** `rsync -a` *copies* (it does not move), so once the
merge is done you have the data three times over: the `.zip` files, the extracted
part directories, and the merged target. Delete the first two to free space:

```bash
# only after verifying the 30/30 scene counts above
rm -f  ~/Downloads/PointNav-*.zip ~/Downloads/SocialNav-*.zip ~/Downloads/mesh-*.zip
rm -rf ~/Downloads/PointNav-*/    ~/Downloads/SocialNav-*/    ~/Downloads/mesh-*/
```

If disk is tight during setup, delete each `.zip` right after extracting it and each
extracted part right after its `rsync`, so you never hold all three copies at once.

## Run steps

You use **two terminals**, both opened on the host:
- **[HOST]** — a plain host shell (no venv activated).
- **[DOCKER]** — a shell inside the container, via `docker exec -it vid2sim_dev bash`.

Run the steps strictly in this order. The ordering matters: Python must be
listening *before* Unity launches.

1. **[HOST]** Start the container (if not already up):
   ```bash
   cd docker && docker compose up -d
   ```

2. **[HOST]** Clean slate — kill any stale processes from prior attempts (these
   are the #1 cause of connection timeouts):
   ```bash
   pkill -9 -f "env.x86_64"
   docker exec vid2sim_dev pkill -9 -f demo.py
   pgrep -af env.x86_64       # must print nothing
   ss -tlnp | grep 5004       # must print nothing
   ```

3. **[DOCKER]** Start the Python driver in the container. It listens on port 5004
   and waits (timeout_wait=300s) for Unity to connect:
   ```bash
   docker exec -it vid2sim_dev bash -c "cd /workspace/src/vid2sim_rl && python demo.py"
   ```
   Wait for: `Listening on port 5004...` — do NOT proceed until you see this.

4. **[HOST]** **Only after step 3 prints `Listening...`**, launch Unity on the
   host. The `--mlagents-port 5004` flag is required — it forces Unity to connect
   to the Python listener. (`--port` is the wrong flag and makes Unity crash.)
   `LANG`/`LC_ALL` avoid a Mono locale crash; the `__NV_*` vars route GLX to the
   NVIDIA GPU instead of software rendering. Run from the repo root:
   ```bash
   LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
     __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
     src/vid2sim_rl/envs/static/05/env.x86_64 \
     --mlagents-port 5004 \
     -screen-width 1280 -screen-height 720 &
   ```

5. **[DOCKER]** Watch the Python terminal: it advances past `Listening...` and
   prints `=== Episode 1/3 ===`. Output MP4s land in
   `demo_output/videos/episode_0{1,2,3}_rgb.mp4`, plus debug frame PNGs in
   `demo_output/` (inside the container, which is the mounted repo on host).

## Video quality — agent POV vs the Unity window

There are two different views, with very different quality:

- **Agent POV (what `demo.py` records).** These frames are the robot's
  *policy-input* camera observation. The Unity build renders that sensor at a fixed
  **128×72** (`obs_width`/`obs_height`), baked into the binary — it is the hard
  resolution ceiling, and no codec/bitrate setting recovers detail that was never
  rendered. `demo.py` writes these with high-quality x264 (`crf 16`, `quality=10`)
  and an `inference.upscale` factor (default 6 → 768×432) that enlarges via LANCZOS
  for viewability — but upscaling does **not** add real detail.

- **Unity window (the good-looking 1280×720 render).** To capture the quality you
  actually see on screen, screen-record the Unity window instead of the observation.
  Use the helper (run on the **host**, while Unity is open):
  ```bash
  # one-time host setup
  sudo apt-get install -y ffmpeg xdotool
  # in a host terminal, after Unity is running (step 4):
  ./src/vid2sim_rl/record_unity.sh unity_capture.mp4   # Ctrl-C to stop
  ```
  It finds the `Video2Sim` window, grabs its exact geometry, and encodes a visually
  lossless (`crf 18`) MP4 of the live render. Launch Unity at a higher resolution
  (`-screen-width 1920 -screen-height 1080`) for an even sharper capture.

## Retrying after a failure or crash

Any failed/timed-out attempt leaves a stale Unity (and sometimes Python) process
holding port 5004. **Before every retry, re-run the step-2 clean slate** — do not
just relaunch:

```bash
# [HOST] kill Unity, and Python inside the container
pkill -9 -f "env.x86_64"
docker exec vid2sim_dev pkill -9 -f demo.py

# verify both are clear (both must print nothing)
pgrep -af env.x86_64
ss -tlnp | grep 5004
```

Symptoms that mean you skipped this: `demo.py` hangs on `Listening on port 5004`
and never connects, or `Player.log` shows reward lines streaming on their own (a
stale Unity stepping without Python). Kill everything, then redo steps 3→4 in order.

## Gotchas

- **Hardcoded developer path (`/home/ziyangxie/...`) — manual workaround required.**
  The Unity binary was built on the original author's machine and the obstacle-GLB
  directory is *hardcoded into the compiled binary* as
  `/home/ziyangxie/Code/Video2Sim-RL/envs/mesh/sample_obj/`. It is NOT configurable
  from `demo.yaml` or any env var, and the path does not exist on our machine, so
  Unity throws `DirectoryNotFoundException` and crashes on the first episode reset
  (`OnActionReceived → EndEpisode`). The workaround applied here: recreate that exact
  absolute path locally and drop the Drive `mesh/sample_obj` GLBs into it —
  ```bash
  mkdir -p /home/ziyangxie/Code/Video2Sim-RL/envs/mesh/sample_obj
  rsync -a ~/Downloads/mesh-*/mesh/sample_obj/ \
           /home/ziyangxie/Code/Video2Sim-RL/envs/mesh/sample_obj/
  ```
  Anyone running this on a new machine (or the RTX 6000 Pro VM) must recreate this
  same `/home/ziyangxie/...` path — it is not relative to the repo or `$HOME`. The
  proper fix would be a rebuilt Unity binary that reads the mesh path from config.
- `random_obj_placement: True` + `max_loaded_obj_num: 1` in `config/demo.yaml`;
  setting these to 0/False crashes Unity during `OnEpisodeBegin`.
- CUDA extensions (vid2sim-rasterizer, groundingdino, simple-knn) are NOT built in
  the Dockerfile (no GPU at build time). Run `docker/setup_extensions.sh` once
  inside the container; the compiled .so files persist in the mounted /workspace.
