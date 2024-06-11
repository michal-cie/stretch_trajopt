from urchin import URDF, Joint, Link, JointLimit

# based on RE3 URDF
KEEP_JOINTS = [
    "joint_mast",
    "joint_lift",
    "joint_arm_l4",
    "joint_arm_l3",
    "joint_arm_l2",
    "joint_arm_l1",
    "joint_arm_l0",
    "joint_wrist_yaw",
    "joint_wrist_yaw_bottom",
    "joint_wrist_pitch",
    "joint_wrist_roll",
    "joint_grasp_center",
    "joint_gripper_s3_body",
    "joint_grasp_center",
    "joint_gripper_finger_left",
    "joint_gripper_fingertip_left",
    "joint_gripper_finger_right",
    "joint_gripper_fingertip_right",
]

KEEP_LINKS = [
    "base_link",
    "link_mast",
    "link_lift",
    "link_arm_l4",
    "link_arm_l3",
    "link_arm_l2",
    "link_arm_l1",
    "link_arm_l0",
    "link_wrist_yaw",
    "link_wrist_yaw_bottom",
    "link_wrist_pitch",
    "link_wrist_roll",
    "link_gripper_s3_body",
    "link_grasp_center",
    "link_gripper_finger_left",
    "link_gripper_fingertip_left",
    "link_gripper_finger_right",
    "link_gripper_fingertip_right",
]

def simplify_udrf(input_urdf_path, output_urdf_path=None):
    # Load the URDF
    robot = URDF.load(input_urdf_path)

    for joint in robot.joints:
        if joint.name not in KEEP_JOINTS:
            # print("Removing joint", joint.name)
            robot._joints.remove(joint)

    for link in robot.links:
        if link.name not in KEEP_LINKS:
            # print("Removing link", link.name)
            robot._links.remove(link)

    # fix finger joints
    robot._joint_map["joint_gripper_finger_right"].joint_type = "fixed"
    robot._joint_map["joint_gripper_finger_left"].joint_type = "fixed"

    # Save the updated URDF
    if output_urdf_path is not None:
        print("Saving simple urdf to " + output_urdf_path)
        robot.save(output_urdf_path)

    return robot

def add_virtual_joints_links(robot: URDF, output_urdf_path=None):
    robot.name = "stretch_full"
    
    # add stuff
    virtual_base_link = Link("virtual_base_link", None, None, None)
    x_virtual_link = Link("x_virtual_link", None, None, None)
    y_virtual_link = Link("y_virtual_link", None, None, None)
    z_virtual_link = Link("z_virtual_link", None, None, None)

    x_prismatic_limit = JointLimit(effort=10, velocity=1, lower=-100., upper=100.)
    y_prismatic_limit = JointLimit(effort=10, velocity=1, lower=-100., upper=100.)
    z_revolute_limit = JointLimit(effort=1000, velocity=10)

    x_prismatic_joint = Joint("x_prismatic_joint", "prismatic", parent="virtual_base_link", child="x_virtual_link", axis=[1,0,0], limit=x_prismatic_limit)
    y_prismatic_joint = Joint("y_prismatic_joint", "prismatic", parent="x_virtual_link", child="y_virtual_link", axis=[0,1,0], limit=y_prismatic_limit)
    z_revolute_joint = Joint("z_revolute_joint", "continuous", parent="y_virtual_link", child="z_virtual_link", axis=[0,0,1], limit=z_revolute_limit)

    virtual_base_joint = Joint("virtual_base_joint", "fixed", parent="z_virtual_link", child="base_link")

    new_links = [virtual_base_link, x_virtual_link, y_virtual_link, z_virtual_link]
    new_joints = [virtual_base_joint, x_prismatic_joint, y_prismatic_joint, z_revolute_joint]

    robot._links = new_links + robot._links
    robot._joints = new_joints + robot._joints

    if output_urdf_path is not None:
        print("Saving full urdf to " + output_urdf_path)
        robot.save(output_urdf_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=False)

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if output_path is not None:
        full_output_path = output_path[:-5] + "_full" + output_path[-5:]
    else:
        print("Warning - no output path specified! Not saving.")
        full_output_path = None

    simple_robot = simplify_udrf(input_path, output_path)
    add_virtual_joints_links(simple_robot, full_output_path)
