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
        print("Control the muscles manually from the viewer!")
        print("\nTo control actuators:")
        print("- Double-click on the model to open the UI")
        print("- Go to 'Control' tab to see all muscle actuators")
        print("- Adjust sliders to control each muscle (0=relaxed, 1=contracted)")
        print("\n12 Muscle Actuators:")
        print("  Right leg: right_hip_flex, right_hip_ext, right_knee_flex, right_knee_ext, right_ankle_flex, right_ankle_ext")
        print("  Left leg:  left_hip_flex, left_hip_ext, left_knee_flex, left_knee_ext, left_ankle_flex, left_ankle_ext")
        print("\nCamera controls:")
        print("- Mouse: Rotate view")
        print("- Scroll: Zoom in/out")
        print("- Right-click + drag: Pan view")
        print("- Press 'r' to reset view")
        
        # Set initial camera view
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 45
        
        # Run simulation with viewer (no automatic control)
        while viewer.is_running():
            # Step the simulation (muscles controlled from viewer UI)
            mujoco.mj_step(model, data)
            
            # Sync viewer with current state
            viewer.sync()
            
            # Small delay to make it visible
            time.sleep(0.01)
    
    print("\nSimulation completed!")
    print("MuJoCo Walker simulation finished!")


if __name__ == "__main__":
    main()
