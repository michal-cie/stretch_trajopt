import casadi as cs
import optas

class PlanarMobileBase(optas.TaskModel):
    def __init__(self):
        """
        Contains the motion model for Stretch's differential drive base.
        """
        super().__init__(
            "diff_drive", 3, time_derivs=[0, 1, 2],
            dlim={0: [-100, 100], 1: [-0.8, 0.8], 2:[-1, 1]}  # limits for position and velocity
        )
        
    def diff_drive_kinematics(self, v: float, w: float, theta: float) -> list:
        """
        Computes the change in the mobile base's state in SE(2) based on
        the current velocities and heading of the base.

        Args:
            v (float): base translational velocity (m/s)
            w (float): base rotational velocity (rad/s)
            theta (float): current base heading (rad)

        Returns:
            list: [dx, dy, dtheta] change in translation and rotation in SE(2)
        """

        dx = v * cs.cos(theta)
        dy = v * cs.sin(theta)
        dtheta = w
        return [dx, dy, dtheta]
