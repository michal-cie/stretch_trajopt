from typing import List
import numpy as np

class Waypoint:
    def __init__(self, position=None, orientation=[0, 0, 0, 1], time=None):
        self.position = position
        self.orientation = orientation
        self.time = time

    def set_position(self, pose: list):
        if len(pose) != 3:
            raise ValueError
        
        self.position = pose

    def get_pose(self) -> list:
        if self.position is not None:
            return self.position
        else:
            raise ValueError
    
    def set_orientation(self, orientation: list):
        if len(orientation) != 4:
            raise ValueError
        
        self.orientation = orientation

    def get_orientation(self) -> list:
        return self.orientation

    def set_time(self, time: float):
        self.time = time

    def get_time(self):
        if self.time is not None:
            return self.time
        else:
            raise ValueError
    
    # helpers
    def set_state(self, position: list, orientation: list, time: float):
        """
        Sets both the position and time of the waypoint.

        Args:
            position (list): 3-length XYZ coordinate
            orientation (list): 4-length [qw qx qy qz] quaternion
            time (float): time (s) from beginning of trajectory
        """

        self.set_position(position)
        self.set_orientation(orientation)
        self.set_time(time)

    def get_state(self):
        return self.get_pose() + self.get_orientation()

class Trajectory:
    def __init__(self, waypoints=None):
        self.waypoints = waypoints

    def set_waypoints(self, waypoints: List[Waypoint]):
        self.waypoints = waypoints

    def set_xyz(self, xyz: np.ndarray):
        """
        Updates waypoints to have positions xyz

        Args:
            xyz (np.ndarray): 3xN list of waypoint locations.
        """

        if xyz.shape[1] != len(self.waypoints):
            raise ValueError
        
        wp = self.get_waypoints()
        for i in range(len(wp)):
            wp[i].set_position(xyz[:, i].tolist())

    def set_orientation(self, orientation):
        wp = self.get_waypoints()
        for i in range(len(wp)):
            wp[i].set_orientation(orientation[:, i].tolist())

    def get_waypoints(self) -> List[Waypoint]:
        if self.waypoints is not None:
            return self.waypoints
        else:
            raise ValueError
        
    # helpers
    def get_length(self) -> int:
        """
        Returns the length (number of points) of the trajectory.
        """

        return len(self.waypoints)
    
    def get_duration(self) -> float:
        """
        Returns the duration of the trajectory in seconds.
        """

        pts = self.get_waypoints()
        start_time = pts[0].time
        end_time = pts[-1].time
        return end_time - start_time
    
    def get_times(self) -> np.ndarray:
        return np.array([w.time for w in self.waypoints])

    def get_xyz_series(self) -> np.ndarray:
        """
        Returns the trajectory as a 3xN np.ndarray of XYZ coordinates.
        """

        return np.array([w.position for w in self.waypoints]).T
    
    def get_orientation_series(self) -> np.ndarray:
        """
        Returns the trajectory as a 4xN np.ndarray of Qw Qx Qy Qz coordinates.
        """

        return np.array([w.orientation for w in self.waypoints]).T
    
    def get_pose_series(self):
        """
        Returns the trajectory as a 7xN np.ndarray of XYZ Qw Qx Qy Qz coordinates.
        """

        xyz = self.get_xyz_series()
        ori = self.get_orientation_series()
        return np.concatenate((xyz, ori), axis=0)
