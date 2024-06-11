from test_imports import *
import hello_planners.whole_body_planner as wbp
from hello_planners.optas_visualizer import visualize_trajectory

import numpy as np

# instantiate planner
urdf_dir = get_urdf_dir()
planner = wbp.Planner(urdf_dir, mode=wbp.OptimizerMode.FULL)

# synthesize a trajectory
time_vector = np.linspace(0., 20., 30)
trajectory = get_triangle_trajectory(time_vector)

# send trajectory to planner and plan
planner.set_trajectory(trajectory)
stretch_full_plan = planner.plan()

# visualize results
visualize_trajectory(trajectory=trajectory, planner=planner, stretch_full_plan=stretch_full_plan, smooth=True)
