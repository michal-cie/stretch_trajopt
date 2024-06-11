from hello_kinematics.trajectory import *
from hello_planners import whole_body_planner as wbp
import numpy as np
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import os

def get_urdf_dir():
    # kind of hacky
    package_path = os.path.dirname(wbp.__file__)
    parent_path = os.path.dirname(os.path.dirname(package_path))
    urdf_dir = os.path.join(parent_path, "description")
    return urdf_dir

# POSITION AND ORIENTATION TRAJECTORIES
def get_sparse_trajectory():
    trajectory = Trajectory()

    # positions relative to end effector start location
    p1 = Waypoint(position=[0., 0., 0.], orientation=[0, 0, 0, 1], time=0.)
    p2 = Waypoint(position=[1., 0., 0.], orientation=[0, 0, 0.7071068, 0.7071068], time=10.)
    p3 = Waypoint(position=[1., 1., 0.], orientation=[0, 0, -0.7071068, 0.7071068], time=20.)
    p4 = Waypoint(position=[0.75, 0.25, 0.], orientation=[0, 0, 0, 1], time=20.)

    trajectory.set_waypoints([p1, p2, p3, p4])
    return trajectory

def get_sparse_trajectory_parametric(n_points: int, distance_between_points: float, dt: float):
    theta_min = np.deg2rad(-180.)
    theta_max = np.deg2rad(-90.)

    theta_range = np.linspace(theta_min, theta_max, n_points)
    
    d_theta = theta_range[1] - theta_range[0]
    radius = distance_between_points / d_theta

    x = radius * np.cos(theta_range) + radius
    y = radius * np.sin(theta_range)

    # o = [0, 0, 0.7071068, 0.7071068]  # orientation
    o = [ 0, 0.3826834, 0, 0.9238795 ]
    # o = [0, 0, 0, 1]
    points = [Waypoint(position=[x[i], y[i], 0.], orientation=o, time=dt * i) for i in range(n_points)]
    trajectory = Trajectory(points)
    return trajectory

def get_small_trajectory(time_horizon: float, dt: float, x_final: list=[0.1, 0.1, 0.]):
    points = [Waypoint(position=[0., 0., 0.], orientation=[0, 0, 0, 1], time=0.)]
    n_points = int((time_horizon // dt) + 1)
    t = np.linspace(0, time_horizon, n_points)

    x0 = [0, 0, 0]
    x_traj = np.array([np.linspace(x0[i], x_final[i], n_points) for i in range(3)])
    
    o = [0, 0, 0, 1]
    for i in range(n_points):
        x_intermediate = x_traj[:, i]
        points.append(Waypoint(position=x_intermediate, orientation=o, time=t[i]))

    return Trajectory(points)


def get_triangle_trajectory(time_vector, point1_pos=[0,0,0], point2_pos=[0.25,0.25,0.25], point3_pos=[0.4,-0.25,-0.25]):
    """
    Wraps trajectory from get_triangle_path() in a Trajectory object.

    Args:
        time_vector (np.array): 1 x N time vector
        point1_pos (list): 3x1 coordinate
        point2_pos (list): 3x1 coordinate
        point3_pos (list): 3x1 coordinate

    Returns:
        Trajectory
    """
    
    path = get_triangle_path(time_vector, point1_pos, point2_pos, point3_pos)
    pos = path[:3, :]
    ori = path[3:, :]
    waypoints = [Waypoint(pos[:, i], ori[:, i], time_vector[i]) for i in range(len(time_vector))]
    trajectory = Trajectory(waypoints)
    return trajectory

def get_triangle_path(time_vector, point1_pos=[0,0,0], point2_pos=[0.25,0.25,0.25], point3_pos=[0.4,-0.25,-0.25]):
    """
    Generates a triangular path with continuously varying orientation along it.

    Args:
        time_vector (np.array): 1 x N time vector
        point1_pos (list): 3x1 coordinate
        point2_pos (list): 3x1 coordinate
        point3_pos (list): 3x1 coordinate

    Returns:
        np.ndarray (7 x N): XYZ QwQxQyQz path, N points
    """

    n_points = len(time_vector)
    points_per_side = n_points // 3
    remainder = n_points % 3

    # pos
    side1_pos = np.linspace(point1_pos, point2_pos, points_per_side).T
    side2_pos = np.linspace(point2_pos, point3_pos, points_per_side).T
    side3_pos = np.linspace(point3_pos, point1_pos, points_per_side).T

    # ori
    ori1 = R.from_quat([0, 0, 0, 1])
    # ori2 = R.from_quat([0, 0, 0, 1])
    # ori3 = R.from_quat([0, 0, 0, 1])
    ori2 = R.from_quat([ 0.1464466, 0.3535534, 0.3535534, 0.8535534 ])  # XYZ euler 0, 45, 45
    ori3 = R.from_quat([ -0.3696438, -0.0990458, -0.2391176, 0.8923991 ])  # XYZ euler -45, 0, -30

    # interpolate orientations
    slerp_12 = Slerp([0, 1], R.concatenate([ori1, ori2]))
    slerp_23 = Slerp([0, 1], R.concatenate([ori2, ori3]))
    slerp_31 = Slerp([0, 1], R.concatenate([ori3, ori1]))
    
    interp_t = np.linspace(0, 1, points_per_side)
    side1_ori = slerp_12(interp_t).as_quat().T
    side2_ori = slerp_23(interp_t).as_quat().T
    side3_ori = slerp_31(interp_t).as_quat().T

    # combine positions and orientations
    side1_traj = np.concatenate((side1_pos, side1_ori), axis=0)
    side2_traj = np.concatenate((side2_pos, side2_ori), axis=0)
    side3_traj = np.concatenate((side3_pos, side3_ori), axis=0)

    # extend to match number of time points
    while remainder > 0:
        end_row = side3_traj[:, -1]
        end_row = np.expand_dims(end_row, 1)
        side3_traj = np.concatenate((side3_traj, end_row), axis=1)
        remainder -= 1

    # assemble trajectory points
    trajectory = np.concatenate((side1_traj, side2_traj, side3_traj), axis=1)

    return trajectory


# POSITION ONLY TRAJECTORIES
def get_figure_eight_trajectory(time_vector: np.ndarray, x_amp: float, y_amp: float, z_amp: float) -> Trajectory:
    """
    Wraps the path from get_figure_eight_path in a Trajectory object.

    Args:
        time_vector (np.array): 1 x N time vector
        x_amp (float): global x component amplitude (m)
        y_amp (float): global y component amplitude (m)
        z_amp (float): global z component amplitude (m)

    Returns:
        Trajectory
    """
    
    path = get_figure_eight_path(time_vector, x_amp, y_amp, z_amp)
    waypoints = [Waypoint(path[:, i], time=time_vector[i]) for i in range(len(time_vector))]
    trajectory = Trajectory(waypoints)
    return trajectory

def get_figure_eight_path(time_vector: np.ndarray, x_amp: float, y_amp: float, z_amp: float) -> np.ndarray:
    """
    Builds a figure-eight shaped trajectory with sine wiggles.

    Args:
        time_vector (np.array): 1 x N time vector
        x_amp (float): global x component amplitude (m)
        y_amp (float): global y component amplitude (m)
        z_amp (float): global z component amplitude (m)

    Returns:
        np.ndarray (3 x N): XYZ path, N points
    """

    n_points = len(time_vector)

    path = np.zeros((3, n_points))
    path[0, :] = x_amp * np.sin(time_vector * np.pi/4).T  # need .T since t is col vec
    path[1, :] = y_amp * np.sin(time_vector * np.pi/2).T  # need .T since t is col vec
    path[2, :] = z_amp * np.sin(time_vector * 2*np.pi).T
    
    return path

def get_three_point_path(T, x_end1, y_end1, z_end1, 
                      x_end2, y_end2, z_end2, 
                      x_end3, y_end3, z_end3):
    """
    TODO: refactor this boi
    """

    path = np.zeros((3, T))
    one_third = T // 3
    two_third = 2 * one_third
    for k in range(T):
        if k < one_third:
            # Interpolate from 0 to point1
            fraction = k / one_third
            path[0, k] = 0 + fraction * x_end1
            path[1, k] = 0 + fraction * y_end1
            path[2, k] = 0 + fraction * z_end1
        elif k < two_third:
            # Interpolate from point1 to point2
            fraction = (k - one_third) / (two_third - one_third)
            path[0, k] = x_end1 + fraction * (x_end2 - x_end1)
            path[1, k] = y_end1 + fraction * (y_end2 - y_end1)
            path[2, k] = z_end1 + fraction * (z_end2 - z_end1)
        else:
            # Interpolate from point2 to point3
            fraction = (k - two_third) / (T - two_third - 1)
            path[0, k] = x_end2 + fraction * (x_end3 - x_end2)
            path[1, k] = y_end2 + fraction * (y_end3 - y_end2)
            path[2, k] = z_end2 + fraction * (z_end3 - z_end2)
    return path
