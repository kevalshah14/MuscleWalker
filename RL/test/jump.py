import time
import mujoco
from stable_baselines3 import PPO


def main():
    import os
    # Import mujoco.viewer only inside main to avoid import issues if not under mjpython
    import mujoco.viewer

    model_path = "../../trained_models/ppo_muscle_kangaroo.zip"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run RL/train/jump.py first!")
        return

    print(f"Loading kangaroo model from {model_path}...")

    # Import the environment from the train script
    import sys
    import os
    # Add the RL directory to path, then import from train
    rl_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, rl_dir)
    from train.jump import MuscleKangarooEnv

    # Create environment
    env = MuscleKangarooEnv(render_mode="human", terminate_on_fall=False)

    # Load the trained model
    model = PPO.load(model_path)

    print("\nStarting kangaroo hopping simulation...")
    print("Press ESC in the viewer window to exit.")

    # Launch passive viewer with camera following the kangaroo
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Use tracking camera mode for smooth following with user control
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1  # Track the torso
        viewer.cam.distance = 4.0   # Slightly farther for hopping
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15

        obs, _ = env.reset()

        while viewer.is_running():
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Print hopping info occasionally
            if env.data.time % 1.0 < 0.05:  # Print every ~1 second
                print(".3f")

            # Sync viewer
            viewer.sync()

            # Slow down playback for better visualization
            time.sleep(0.03)  # ~30 FPS for hopping

            # Reset if episode ends
            if terminated or truncated:
                print("Episode finished. Resetting...")
                obs, _ = env.reset()


if __name__ == "__main__":
    main()
