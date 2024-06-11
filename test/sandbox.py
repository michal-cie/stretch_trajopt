from test_imports import *
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def plot_coordinate_frames(a: plt.Axes, positions, orientations, axis_length: float=0.1):
    l = axis_length
    
    r = R.from_quat(orientations.T).as_matrix() * l
    # print(r.shape)
    x = r[:, :, 0]
    y = r[:, :, 1]
    z = r[:, :, 2]
    
    # print(positions.shape)
    for i in range(positions.shape[1]):
        p = positions[:, i]
        # print(p)
        x0 = p[0]
        y0 = p[1]
        z0 = p[2]
        a.plot([x0, x0 + x[i, 0]], [y0, y0 + x[i, 1]], [z0, z0 + x[i, 2]], color='r')
        a.plot([x0, x0 + y[i, 0]], [y0, y0 + y[i, 1]], [z0, z0 + y[i, 2]], color='g')
        a.plot([x0, x0 + z[i, 0]], [y0, y0 + z[i, 1]], [z0, z0 + z[i, 2]], color='b')
    

def test_trajectory_generator():
    # t = np.linspace(0., 20., 50)
    # traj = get_triangle_path(t)
    # traj = get_sparse_trajectory().get_pose_series()
    # traj = get_sparse_trajectory_parametric(10, 0.1, 5.0).get_pose_series()
    traj = get_small_trajectory(1, 0.1).get_pose_series()
    # print(traj)

    f = plt.figure()
    a = f.add_subplot(projection='3d')

    a.scatter(traj[0, :], traj[1, :], traj[2, :], color='k')
    plot_coordinate_frames(a, traj[:3, :], traj[3:])

    # config plot
    a.set_xlabel('x (m)')
    a.set_ylabel('y (m)')
    a.set_zlabel('z (m)')
    a.set_aspect('equal')
    plt.show()

if __name__ == '__main__':
    test_trajectory_generator()