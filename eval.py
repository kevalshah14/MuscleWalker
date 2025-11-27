import imageio
import numpy as np
from stable_baselines3 import PPO
from RL_env import MuscleWalkerEnv
import time


def record_video(model_path, xml_path="assets/walker/muscle_new.xml", video_path="ppo_eval.mp4",
                 episode_len=2000, frame_skip=10):

    # Create environment in rgb_array mode
    env = MuscleWalkerEnv(xml_path=xml_path, frame_skip=frame_skip, render_mode="human")

    # Load trained model
    model = PPO.load(model_path)

    obs, _ = env.reset()
    frames = []

    for i in range(episode_len):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()
        time.sleep(0.02)
        frames.append(frame)
        if terminated or truncated:
            break

        if i==0:
            input("Press Enter to start recording...")

    env.close()

    # Save to video
    # print(f"Saving video ({len(frames)} frames) to {video_path}...")
    # imageio.mimsave(video_path, frames, fps=30)
    # print("Done!")


if __name__ == "__main__":
    record_video(
        model_path="ppo_muscle_walker.zip",
        xml_path="assets/walker/muscle_new.xml",
        video_path="ppo_eval.mp4",
        episode_len=2000
    )