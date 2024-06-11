from test_imports import *
import hello_planners.whole_body_planner as wbp
from hello_planners.optas_visualizer import visualize_trajectory

import numpy as np

import time
import json

def test_planner_init():
    print("\nTest Planner Initialization")
    urdf_dir = get_urdf_dir()
    print("Description path: ")
    print(urdf_dir, "\n")

    p = wbp.Planner(urdf_dir)
    print("Loading robots...")
    p.init_robot(urdf_dir)
    print("Stretch Simplified # DOF: ", p.stretch.ndof)
    print("Stretch Full # DOF: ", p.stretch_full.ndof, "\n")

    print(p.solver)

def test_planner_plan(visualize: bool=False):
    urdf_dir = get_urdf_dir()
    planner = wbp.Planner(urdf_dir)

    time_vector = np.linspace(0., 20., 200)
    trajectory = get_figure_eight_trajectory(time_vector, 0.3, 0.5, 0.1)

    planner.set_trajectory(trajectory)

    stretch_full_plan = planner.plan()

    if visualize:
        visualize_trajectory(trajectory=trajectory, planner=planner, stretch_full_plan=stretch_full_plan)

def test_planner_orientation_tracking(visualize: bool=False):
    urdf_dir = get_urdf_dir()
    planner = wbp.Planner(urdf_dir, mode=wbp.OptimizerMode.FULL)

    dt = 0.2
    n = 50
    time_vector = np.linspace(0., n * dt, n)
    trajectory = get_triangle_trajectory(time_vector)
    
    planner.set_trajectory(trajectory)

    stretch_full_plan = planner.plan()

    if visualize:
        visualize_trajectory(trajectory=trajectory, planner=planner, stretch_full_plan=stretch_full_plan)

def test_planner_sparse_trajectory(visualize: bool=False):
    urdf_dir = get_urdf_dir()
    planner = wbp.Planner(urdf_dir, mode=wbp.OptimizerMode.FULL)

    # trajectory = get_sparse_trajectory()
    trajectory = get_sparse_trajectory_parametric(20, 0.05, 0.5)

    planner.set_trajectory(trajectory)

    stretch_full_plan = planner.plan()

    if visualize:
        visualize_trajectory(trajectory=trajectory, planner=planner, stretch_full_plan=stretch_full_plan)

def test_planner_sparse_trajectories(visualize: bool=False):
    urdf_dir = get_urdf_dir()
    planner = wbp.Planner(urdf_dir, mode=wbp.OptimizerMode.FULL)

    n_points = 75
    # distances_between_points = np.arange(0.1, 1.01, 0.1)
    distances_between_points = np.array([0.1, 0.25, 0.5, 0.75, 1])
    max_m_s = 0.1
    dts = distances_between_points / max_m_s

    for dp, dt in zip(distances_between_points, dts):
        trajectory = get_sparse_trajectory_parametric(n_points, dp, dt)

        planner.set_trajectory(trajectory)

        stretch_full_plan = planner.plan()

        if visualize:
            visualize_trajectory(trajectory=trajectory, planner=planner, stretch_full_plan=stretch_full_plan, smooth=False)

def test_planner_short_trajectories(n=1, save_data: bool=True):
    urdf_dir = get_urdf_dir()
    planner = wbp.Planner(urdf_dir, mode=wbp.OptimizerMode.FULL)

    # config
    horizons = np.arange(0.25, 5, 0.125)
    dts = [0.0125, 0.025, 0.05, 0.1, 0.25]

    all_times = {str(h): {} for h in horizons}

    for horizon in horizons:
        for dt in dts:
            times = np.zeros(n)

            for i in range(n):
                time_start = time.time()
                trajectory = get_small_trajectory(horizon, dt)
                planner.set_trajectory(trajectory)
                planner.plan()
                
                times[i] = time.time() - time_start

            all_times[str(horizon)][str(dt)] = times.tolist()

    if save_data:
        out_file = open("times.json", "w") 
        json.dump(all_times, out_file, indent = 2)
        out_file.close() 

def test_all():
    test_planner_init()
    test_planner_plan()
    test_planner_orientation_tracking()
    test_planner_sparse_trajectory()
    test_planner_short_trajectories(save_data=False)
    test_planner_sparse_trajectories()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action='store_true')

    args = parser.parse_args()

    if args.all:
        test_all()
    else:
        # edit this to choose a test to run
        
        # test_planner_plan(visualize=True)
        test_planner_orientation_tracking(visualize=True)
        # test_planner_sparse_trajectory(visualize=True)
        # test_planner_short_trajectories()
        # test_planner_sparse_trajectories(visualize=True)
