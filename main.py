import mujoco
import mujoco.viewer
import numpy as np
import time
import os


def load_walker_model():
    """Load the walker model from the assets folder."""
    # Get the path to the walker.xml file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    walker_path = os.path.join(current_dir, "assets", "walker", "walker.xml")
    
    if not os.path.exists(walker_path):
        raise FileNotFoundError(f"Walker model not found at {walker_path}")
    
    print(f"Loading walker model from: {walker_path}")
    return mujoco.MjModel.from_xml_path(walker_path)


def main():
    """Main function to run the MuJoCo muscle walker environment."""
    print("Setting up MuJoCo Muscle Walker environment...")
    
    # Load the walker model from assets
    model = load_walker_model()
    data = mujoco.MjData(model)
    
    print(f"Model loaded with {model.nq} DOF and {model.nu} actuators")
    print(f"Simulation timestep: {model.opt.timestep}")
    print(f"Actuator names: {[model.actuator(i).name for i in range(model.nu)]}")
    
    # Initialize simulation
    mujoco.mj_resetData(model, data)
    
    # Create viewer using MuJoCo's built-in viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nStarting visual simulation...")
        print("Press ESC or close the window to exit")
        print("The walker will attempt to walk forward using simple control logic")
        print("\nCamera controls:")
        print("- Mouse: Rotate view")
        print("- Scroll: Zoom in/out")
        print("- Right-click + drag: Pan view")
        print("- Press 'r' to reset view")
        
        # Set initial camera view
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 45
        
        # Run simulation with viewer
        step = 0
        phase = 0  # For alternating leg movement
        last_position = data.qpos[0]  # Track x position
        
        while viewer.is_running():
            # Simple walking control logic
            if step % 50 == 0:  # Change control every 50 steps
                phase = (phase + 1) % 2
                
                # Reset all controls
                data.ctrl[:] = 0
                
                if phase == 0:  # Right leg forward, left leg back
                    data.ctrl[0] = 0.3   # right_hip
                    data.ctrl[1] = 0.2   # right_knee  
                    data.ctrl[2] = 0.1   # right_ankle
                    data.ctrl[3] = -0.3  # left_hip
                    data.ctrl[4] = -0.2  # left_knee
                    data.ctrl[5] = -0.1  # left_ankle
                else:  # Left leg forward, right leg back
                    data.ctrl[0] = -0.3  # right_hip
                    data.ctrl[1] = -0.2  # right_knee
                    data.ctrl[2] = -0.1  # right_ankle
                    data.ctrl[3] = 0.3   # left_hip
                    data.ctrl[4] = 0.2   # left_knee
                    data.ctrl[5] = 0.1   # left_ankle
                
                current_position = data.qpos[0]
                distance_moved = current_position - last_position
                last_position = current_position
                
                print(f"Step {step}: Phase {phase} - {'Right leg forward' if phase == 0 else 'Left leg forward'}")
                print(f"  Position: x={current_position:.3f}, Distance moved: {distance_moved:.3f}")
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer with current state
            viewer.sync()
            
            # Small delay to make it visible
            time.sleep(0.01)
            
            step += 1
    
    print("\nSimulation completed!")
    print("MuJoCo Walker simulation finished!")


if __name__ == "__main__":
    main()
