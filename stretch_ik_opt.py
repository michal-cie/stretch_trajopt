import os
import optas
from casadi import fabs
import numpy as np  

# Specify URDF filename
base_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(base_path, "urdf", "stretch.urdf")
urdf = os.path.join(base_path, "urdf", "stretch_full.urdf")

####################### Setup #######################
robot = optas.RobotModel(urdf_filename=urdf)
name = robot.get_name()

T = 1
builder = optas.OptimizationBuilder(T, robots=robot)

qn = builder.add_parameter("q_nominal", robot.ndof)
pg = builder.add_parameter("p_goal", 3)

# Nominal configuration for the robot
extension = 0.5
nominal_extension_prismatic = [extension/4, extension/4, extension/4, extension/4]

#################### Constraints ####################
### Constraint: end effector goal position
q = builder.get_model_state(name, 0)
end_effector_name = "link_grasp_center"
p = robot.get_global_link_position(end_effector_name, q)
builder.add_equality_constraint("end_goal", p, pg)

### Constraint: joint position limits
builder.enforce_model_limits(name)  # joint limits extracted from URDF

####################### Costs #######################
### Cost: absolute y displacement 
# - perfer to turn the base rather than extend the arm
builder.add_cost_term("y_displacement", 10000*fabs(q[1]))

### Cost: displacement from default arm extension
# - perfer to move base to reach goal rather than extend the arm
builder.add_cost_term("arm_extension", optas.sumsqr(optas.sum1(q[4:8]) - np.array(nominal_extension_prismatic)))

############# Build optimization problem #############
optimization = builder.build()
solver = optas.CasADiSolver(optimization).setup("ipopt")

#################### Solve problem ####################
### Initial guess - use current configuration of robot
# Nominal configuration for the robot, new each time you solve
# Note: you might not need to change the xy and theta and instead 
#       keep them fixed in the initialization because if you are 
#       solving with respect to the base frame.
xy_virtual_prismatic = [0.0, 0.0]
theta_virtual_revolute = optas.deg2rad([0.0]).full().flatten().tolist()
lift_prismatic = [0.5]
yaw_gripper_revolute = optas.deg2rad([0.0]).full().flatten().tolist()

qn = xy_virtual_prismatic + theta_virtual_revolute + lift_prismatic + nominal_extension_prismatic + yaw_gripper_revolute

### End-effector position in nominal configuration
p_nominal = robot.get_global_link_position(end_effector_name, qn)

### End-effector goal position
p_goal = p_nominal + optas.DM([2.0, 3.5, 0.1])

### Reset per each solution attempt
solver.reset_parameters({"q_nominal": qn, "p_goal": p_goal})
solver.reset_initial_seed({f"{name}/q": qn})

solution = solver.solve()
q_solution = solution[f"{name}/q"]


#### Visualization is broken in python3.10 ####

# Visualize the robot
vis = optas.Visualizer(quit_after_delay=20.0)

#### Visualization is broken in python3.10
# Draw goal position and start visualizer
vis.robot(robot, q=qn,display_link_names=True,show_links=True)   # nominal
vis.robot(robot, q=q_solution, display_link_names=True, show_links=True)  # solution

vis.start()