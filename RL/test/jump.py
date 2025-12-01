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
        viewer.cam.distance = 3.5   # Closer for better hopping view
        viewer.cam.azimuth = 90     # Side view
        viewer.cam.elevation = -20  # Slightly lower angle
        viewer.cam.lookat[2] = 0.5  # Look at torso height

        obs, _ = env.reset()

        while viewer.is_running():
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Print hopping info occasionally
            if env.data.time % 0.5 < 0.05:  # Print every ~0.5 second
                height = env.data.qpos[1]
                forward_pos = env.data.qpos[0]
                # Show muscle activations: [hip_right, knee_right, ankle_right, hip_left, knee_left, ankle_left]
                muscle_activations = env.data.ctrl[:6] if hasattr(env.data, 'ctrl') else [0]*6
                reward_display = reward
                print(".3f")
                # Show if muscles are activated (should be > 0.1 for meaningful force)
                strong_muscles = sum(1 for m in muscle_activations if m > 0.1)
                print(f"Active muscles: {strong_muscles}/6, Max activation: {max(muscle_activations):.3f}")

            # Sync viewer
            viewer.sync()

            # Slow down playback for better visualization (slow)
            time.sleep(0.10)  # ~10 FPS for very clear hopping visualization

            # Reset if episode ends
            if terminated or truncated:
                print("Episode finished. Resetting...")
                obs, _ = env.reset()


if __name__ == "__main__":
    main()
