import mujoco
import mujoco.viewer
import numpy as np
import time
import os


def load_walker_model():
    """Load the walker model from the assets folder."""
    # Get the path to the walker.xml file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    walker_path = os.path.join(current_dir, "assets", "walker", "muscle_walker.xml")
    
    if not os.path.exists(walker_path):
        raise FileNotFoundError(f"Walker model not found at {walker_path}")
    
    print(f"Loading walker model from: {walker_path}")
    return mujoco.MjModel.from_xml_path(walker_path)


def apply_stabilizing_control(model, data):
    """
    Apply baseline muscle activation to keep the walker standing.
    This provides automatic stabilization - you can override by adjusting
    controls in the viewer UI (though they may be overwritten each frame).
    """
    # Find joint IDs by name and get their qpos addresses
    def get_joint_angle(joint_name):
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                qpos_addr = model.jnt_qposadr[joint_id]
                return data.qpos[qpos_addr]
        except:
            pass
        return 0.0
    
    # Get joint angles
    hip_angles = np.array([
        get_joint_angle("right_hip"),
        get_joint_angle("left_hip")
    ])
    knee_angles = np.array([
        get_joint_angle("right_knee"),
        get_joint_angle("left_knee")
    ])
    ankle_angles = np.array([
        get_joint_angle("right_ankle"),
        get_joint_angle("left_ankle")
    ])
    
    # Baseline activation to keep joints from collapsing
    # Hip: slight flexion to keep legs forward (positive angle = flexion)
    hip_activation = 0.3 + 0.2 * np.clip(hip_angles / 0.5, -1, 1)
    
    # Knee: moderate activation to prevent hyperextension (negative angle = extension)
    knee_activation = 0.4 + 0.3 * np.clip(knee_angles / -1.0, -1, 1)
    
    # Ankle: slight activation to keep feet flat
    ankle_activation = 0.2 + 0.2 * np.clip(ankle_angles / 0.3, -1, 1)
    
    # Apply to all muscles
    ctrl = np.zeros(model.nu)
    ctrl[0] = np.clip(hip_activation[0], 0, 1)  # right_hip_flex
    ctrl[1] = np.clip(knee_activation[0], 0, 1)  # right_knee_flex
    ctrl[2] = np.clip(ankle_activation[0], 0, 1)  # right_ankle_flex
    ctrl[3] = np.clip(hip_activation[1], 0, 1)  # left_hip_flex
    ctrl[4] = np.clip(knee_activation[1], 0, 1)  # left_knee_flex
    ctrl[5] = np.clip(ankle_activation[1], 0, 1)  # left_ankle_flex
    
    data.ctrl[:] = ctrl


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
        print("\nMUSCLE WALKER: Full walker using muscle.xml pattern")
        print("  6 Muscle actuators (one per joint):")
        print("    - Right hip, knee, ankle flexors")
        print("    - Left hip, knee, ankle flexors")
        print("  Each joint has:")
        print("    - 1 Active tendon (stiffness=0) with muscle actuator")
        print("    - 1 Passive tendon (stiffness=100) for antagonistic action")
        print("\nTORSO ANCHORED: Like muscle.xml's fixed base, torso is welded")
        print("  to prevent falling. You can test muscles by adjusting sliders.")
        print("\nCamera controls:")
        print("- Mouse: Rotate view")
        print("- Scroll: Zoom in/out")
        print("- Right-click + drag: Pan view")
        print("- Press 'r' to reset view")
        
        # Set initial camera view
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 45
        
        # Run simulation (torso is anchored, so no automatic control needed)
        # You can control muscles manually via the viewer UI
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
