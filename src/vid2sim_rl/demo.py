import os
import imageio
import numpy as np
import hydra
from omegaconf import DictConfig
from env import build_env


@hydra.main(version_base=None, config_path="config", config_name="demo")
def main(cfg: DictConfig):
    env_path = cfg.env.test_env_paths[0]
    print(f"Loading environment: {env_path}")

    # Pass file_name=None to connect to a Unity instance already running on the host.
    # Launch the executable manually on the host first with NVIDIA GPU support.
    env = build_env(None, cfg, worker_id=0,
                    random_seed=0, inference_mode=True, no_graphics=False)

    os.makedirs(cfg.inference.output_dir, exist_ok=True)
    video_dir = os.path.join(cfg.inference.output_dir, cfg.inference.video_path)
    os.makedirs(video_dir, exist_ok=True)

    for ep in range(cfg.inference.num_episodes):
        print(f"\n=== Episode {ep+1}/{cfg.inference.num_episodes} ===")
        video_path = os.path.join(video_dir, f"episode_{ep:02d}_rgb.mp4")
        writer = imageio.get_writer(video_path, fps=cfg.inference.fps, macro_block_size=1)

        obs, _ = env.reset()
        done = False

        for step in range(cfg.inference.max_steps):
            # Forward-biased policy: mostly go straight, occasional turns
            action = env.action_space.sample()
            action[0] = 0.8  # forward speed (index 0 = throttle)
            obs, reward, done, _, info = env.step(action)

            if info.get('raw_img') is not None:
                image = info['raw_img'][-1].transpose(1, 2, 0)
                image = (image * 255).astype(np.uint8)
                if step in (0, 10, 50, 100, 150) and ep == 0:
                    print(f"  [debug] step={step} min={image.min()} max={image.max()} mean={image.mean():.2f}")
                    import imageio as iio
                    iio.imwrite(os.path.join(cfg.inference.output_dir, f'debug_frame_step{step:03d}.png'), image)
                writer.append_data(image)

            if done:
                print(f"  Episode finished at step {step+1}")
                break

        writer.close()
        print(f"  Saved: {video_path}")

    env.close()
    print(f"\nDemo complete. Videos saved to: {cfg.inference.output_dir}/")


if __name__ == "__main__":
    main()
