from test_imports import *
import hello_kinematics.trajectory as tj

import numpy as np

def test_figure_eight():
    time_vector = np.linspace(0, 20., 200)
    figure_eight_trajectory = get_figure_eight_trajectory(time_vector, 1., 1., 1.)

    print([w.time for w in figure_eight_trajectory.get_waypoints()])

if __name__ == '__main__':
    test_figure_eight()