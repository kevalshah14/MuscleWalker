import time
import mujoco
import numpy as np
from stable_baselines3 import PPO


def main():
    import os
    # Import mujoco.viewer only inside main to avoid import issues if not under mjpython
    import mujoco.viewer

    model_path = "../../trained_models/ppo_muscle_trotter.zip"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run RL/train/trot.py first!")
        return

    print(f"Loading trotter model from {model_path}...")

    # Import the environment from the train script
    import sys
    import os
    # Add the RL directory to path, then import from train
    rl_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, rl_dir)
    from train.trot import MuscleTrotterEnv

    # Create environment
    env = MuscleTrotterEnv(render_mode="human", terminate_on_fall=False)

    # Load the trained model
    model = PPO.load(model_path)

    print("\nStarting trotting simulation...")
    print("Press ESC in the viewer window to exit.")

    # Launch passive viewer with camera following the trotter
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Use tracking camera mode for smooth following with user control
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1  # Track the torso
        viewer.cam.distance = 4.0    # Slightly further back for faster trotting view
        viewer.cam.azimuth = 90     # Side view
        viewer.cam.elevation = -20  # Slightly lower angle
        viewer.cam.lookat[2] = 0.7  # Look at torso height

        obs, _ = env.reset()

        while viewer.is_running():
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Print trotting info occasionally
            if env.data.time % 0.5 < 0.05:  # Print every ~0.5 second
                height = env.data.qpos[1]
                forward_pos = env.data.qpos[0]
                forward_vel = env.data.qvel[0]
                reward_display = reward
                trotting_phase = info.get('trotting_phase', 0.0)
                print(f"Time: {env.data.time:.1f}s | Pos: {forward_pos:.2f}m | Vel: {forward_vel:.2f}m/s | Height: {height:.2f}m | Reward: {reward_display:.3f}")
                
                # Show muscle activations and trotting rhythm
                muscle_activations = env.data.ctrl[:6] if hasattr(env.data, 'ctrl') else [0]*6
                active_muscles = sum(1 for m in muscle_activations if m > 0.1)
                left_power = np.mean(muscle_activations[:3])  # Left leg muscles
                right_power = np.mean(muscle_activations[3:])  # Right leg muscles
                power_balance = abs(left_power - right_power)
                
                # Determine which leg is dominant
                dominant_leg = "LEFT" if left_power > right_power else "RIGHT"
                print(f"  Muscles: {active_muscles}/6 active | Alternation: {power_balance:.3f} | Dominant: {dominant_leg}")

                # Show if achieving trotting speed target (1.2 m/s)
                speed_status = "✓ TROTTING" if forward_vel >= 1.0 else "~ FAST WALK" if forward_vel >= 0.7 else "✗ SLOW"
                print(f"  Status: {speed_status} | Phase: {trotting_phase:.2f}")

            # Sync viewer
            viewer.sync()

            # Slow down playback for better visualization (moderate speed for trotting)
            time.sleep(0.01)  # ~25 FPS for clear trotting visualization

            # Reset if episode ends
            if terminated or truncated:
                print("Episode finished. Resetting...")
                obs, _ = env.reset()


if __name__ == "__main__":
    main()
