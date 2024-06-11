from hello_kinematics.trajectory import Trajectory
from hello_planners.whole_body_planner import Planner

import numpy as np
import casadi as cs
from optas import Visualizer

def get_actual_robot_trajectory(stretch_full_plan: cs.DM, planner: Planner):
    """
    TODO: this seems useful to move elsewhere, maybe for evaluating the accuracy of the optimization?

    Args:
        stretch_full_plan (cs.DM): array of full body configurations for Stretch.
        planner (Planner): planner used to compute trajectory (used for FK for Stretch)
    """

    n_points = stretch_full_plan.shape[1]
    path_actual = np.zeros((3, n_points))
    for k in range(n_points):
        q_sol = np.array(stretch_full_plan[:, k])
        pn = cs.DM(planner.stretch_full.get_global_link_position("link_grasp_center", 
                                                                 q_sol)).full()    
        path_actual[:, k] = pn.flatten()

    return path_actual

def interpolate_solution(stretch_full_plan: cs.DM, timestep_multiplier: int=1):
    """
    Interpolate between timesteps for smoother animation.

    Args:
        stretch_full_plan (cs.DM): array of full body configurations for Stretch.
        timestep_multiplier (int): number of frames to add between each body configuration for
        smoother rendering.
    """
    original_timesteps = np.linspace(0, 1, stretch_full_plan.size2())
    interpolated_timesteps = np.linspace(0, 1, timestep_multiplier * stretch_full_plan.size2())
    interpolated_solution = cs.DM.zeros(stretch_full_plan.size1(), len(interpolated_timesteps))
    for i in range(stretch_full_plan.size1()):
        interpolated_solution[i, :] = cs.interp1d(original_timesteps, 
                                                  stretch_full_plan[i, :].T, 
                                                  interpolated_timesteps)
        
    return interpolated_solution

def visualize_trajectory(trajectory: Trajectory, planner: Planner, stretch_full_plan: cs.DM, smooth: bool=True):
    """
    Visualizes a trajectory for Stretch.

    Args:
        trajectory (Trajectory): trajectory that Stretch is following.
        planner (Planner): planner used to compute trajectory.
        stretch_full_plan (cs.DM): array of full body configurations for Stretch.
        smooth (bool): whether to smooth the rendering for the trajectory
    """

    T = trajectory.get_length()
    duration = trajectory.get_duration()
    path = trajectory.get_xyz_series()

    path_actual = get_actual_robot_trajectory(stretch_full_plan=stretch_full_plan, planner=planner)

    smoothing_factor = 5 if smooth else 1  # 5 frames between each solution
    interpolated_solution = interpolate_solution(stretch_full_plan, smoothing_factor)

    print("Initializing Visualizer...")
    vis = Visualizer(camera_position=[2,3,3], camera_focal_point=[-1,-2,0])

    for i in range(len(path[0, :])):
        vis.sphere(position=path[:, i], radius=0.01, rgb=[1, 0, 0])
        vis.sphere(position=path_actual[:, i], radius=0.01, rgb=[0, 1, 0])

    vis.grid_floor()
    vis.robot_traj(planner.stretch_full, np.array(interpolated_solution), 
                   animate=True, duration=duration)
    
    print("Launching Visualizer...")
    vis.start()