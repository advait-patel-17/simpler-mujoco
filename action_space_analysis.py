import mujoco
import numpy as np
from mujoco import MjModel


def analyze_action_space(model_path):
    # Load the model
    model = MjModel.from_xml_path(model_path)
    
    print(f"\nTotal number of actuators: {model.nu}")
    print("\nActuator Analysis:")
    print("-" * 80)
    print(f"{'Index':<6} {'Name':<30} {'Control Range':<20} {'Gear Ratio':<15} {'Controlled Joint'}")
    print("-" * 80)
    
    for i in range(model.nu):
        # Get actuator name
        actuator_name = model.actuator_names[i]
        
        # Get control range
        ctrl_range = f"[{model.actuator_ctrlrange[i][0]:.2f}, {model.actuator_ctrlrange[i][1]:.2f}]"
        
        # Get gear ratio (gives us information about the actuator's strength/scaling)
        gear = model.actuator_gear[i][0]
        
        # Get the joint that this actuator controls
        # We can get this from the transmission target (trnid)
        joint_id = model.actuator_trnid[i][0]
        if joint_id >= 0 and joint_id < len(model.joint_names):
            joint_name = model.joint_names[joint_id]
        else:
            joint_name = "N/A"
            
        print(f"{i:<6} {actuator_name:<30} {ctrl_range:<20} {gear:<15.2f} {joint_name}")
    
    print("\nControl Space Shape:", model.nu)
    print("\nControl Range Summary:")
    for i in range(model.nu):
        print(f"  {model.actuator_names[i]}: [{model.actuator_ctrlrange[i][0]:.2f}, {model.actuator_ctrlrange[i][1]:.2f}]")
        
    # Additional information about actuator dynamics
    print("\nActuator Dynamics:")
    print("  - Damping:", model.actuator_biasprm[:, 0])
    print("  - Friction Loss:", model.actuator_biasprm[:, 1])
    print("  - Time Constant:", model.actuator_biasprm[:, 2])

if __name__ == "__main__":
    model_path = "./aloha/robolab_setup.xml"
    analyze_action_space(model_path)