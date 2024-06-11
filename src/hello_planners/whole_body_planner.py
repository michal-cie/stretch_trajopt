from typing import List
from enum import Enum

from hello_kinematics.motion_models import PlanarMobileBase
from hello_kinematics.trajectory import Trajectory, Waypoint

import os
import casadi  as cs
import optas
from optas.templates import Manager
from optas.spatialmath import Quaternion
from scipy.spatial.transform import Rotation

class OptimizerMode(Enum):
    POSITION = 0
    FULL = 1

class Optimizer():
    def __init__(self):
        pass

    def init_optimization_builder(self, stretch_simplified: optas.RobotModel, stretch_full: optas.RobotModel, planar_mobile_base: PlanarMobileBase, n_points: int, dt: float, mode: OptimizerMode=OptimizerMode.POSITION) -> optas.OptimizationBuilder:
        """
        Configures the optimizer to solve whole-body trajectory optimization for Stretch.

        Args:
            stretch_simplified (optas.RobotModel): URDF for Stretch only including arm / wrist joints
            stretch_full (optas.RobotModel): URDF for Stretch with virtual base joints (x, y, th)
            planar_mobile_base (motion_models.PlanarMobileBase): model for Stretch's base
            n_points (int): number of points in the trajectory
            dt (float): time between steps (s)
            mode (OptimizerMode): mode to run in (position only or full position + orientation)

        Returns:
            optas.OptimizationBuilder: configuration for the optimization problem.
        """
        
        print("Optimizer::init_optimization_builder: starting")
        # inits
        stretch_name = stretch_simplified.name
        base_name = planar_mobile_base.name
        link_ee = 'link_grasp_center'

        # formulate optimization problem: create OptimizationBuilder and populate constraints
        _builder = optas.OptimizationBuilder(T=n_points, tasks=[planar_mobile_base], robots=[stretch_simplified])
        self.add_stretch_arm_constraints(_builder, stretch=stretch_simplified)
        self.add_mobile_base_constraints(_builder, planar_mobile_base=planar_mobile_base)
        
        [q_current, q_nominal, base_state, path] = self.init_optimization_parameters(_builder, stretch_ndof=stretch_simplified.ndof, n_points=n_points)

        self.add_initial_constraints(_builder, stretch_name=stretch_name, qc=q_current, planar_mobile_base_name=base_name, base_init=base_state)
        self.add_dynamics_constraints(_builder, stretch_name=stretch_name, planar_mobile_base_name=base_name, dt=dt)

        [Q, Q_full] = self.get_symbolic_trajectory_variables(_builder, stretch_name=stretch_name, planar_mobile_base_name=base_name)

        self.add_differential_drive_constraints(_builder, planar_mobile_base=planar_mobile_base, n_points=n_points, dt=dt)
        self.add_end_effector_costs(_builder, stretch_full=stretch_full, Q_full=Q_full, path=path, link_ee=link_ee, n_points=n_points)

        if mode == OptimizerMode.FULL:
            self.add_orientation_costs(_builder, stretch_full=stretch_full, Q_full=Q_full, link_ee=link_ee, n_points=n_points)

        self.add_other_costs(_builder, qn=q_nominal, Q=Q, stretch_name=stretch_name, planar_mobile_base_name=base_name, n_points=n_points)

        print("Optimizer::init_optimization_builder: complete!")
        return _builder
        
    @staticmethod
    def add_stretch_arm_constraints(builder: optas.OptimizationBuilder, stretch: optas.RobotModel):
        """
        Adds robot arm constraints to the optimization builder in-place.

        Args:
            builder (optas.OptimizationBuilder): the optimization builder to update.
            stretch (optas.RobotModel): [simplified] stretch robot model
        """

        stretch_name = stretch.name

        # Get stretch position and velocity limits
        lower_pos_limits, upper_pos_limits = stretch.get_limits(0)
        lower_vel_limits, upper_vel_limits = stretch.get_limits(1)
        
        # Stretch arm limits
        builder.enforce_model_limits(stretch_name,
                                     0,
                                     lower_pos_limits,
                                     upper_pos_limits)

        builder.enforce_model_limits(stretch_name,
                                     1,
                                     lower_vel_limits,
                                     upper_vel_limits)

    @staticmethod
    def add_mobile_base_constraints(builder: optas.OptimizationBuilder, planar_mobile_base: PlanarMobileBase):
        """
        Adds mobile base constraints to the optimization builder in-place.

        Args:
            builder (optas.OptimizationBuilder): the optimization builder to update.
            planar_mobile_base (motion_models.PlanarMobileBase): mobile base motion model
        """

        # Get base velocity and acceleration limits
        planar_mobile_base_name = planar_mobile_base.name
        base_lower_vel_limits, base_upper_vel_limits = planar_mobile_base.get_limits(1)
        base_lower_acc_limits, base_upper_acc_limits = planar_mobile_base.get_limits(2)

        # Mobile base limits
        builder.enforce_model_limits(planar_mobile_base_name,
                                     1, 
                                     base_lower_vel_limits,
                                     base_upper_vel_limits)
        builder.enforce_model_limits(planar_mobile_base_name,
                                     2, 
                                     base_lower_acc_limits,
                                     base_upper_acc_limits)

    @staticmethod
    def init_optimization_parameters(builder: optas.OptimizationBuilder, stretch_ndof: int, n_points: int) -> List[cs.SX]:
        """
        Adds and returns the target optimization parameters for the robot and the path in-place.

        Args:
            builder (optas.OptimizationBuilder): the optimization builder to update.
            stretch_ndof (int): number of degrees of freedom that Stretch's URDF has.
            n_points (int): number of points in the path.
        
        Returns (all as CasADi SX):
            q_current (SX): the current robot joint configuration
            q_nominal (SX): the nominal robot joint configuration
            base_state (SX): mobile base state [X, Y, theta]
            path (SX): the trajectory waypoints
        """

        # Current robot joint configuration
        q_current = builder.add_parameter("qc", stretch_ndof)

        # Nominal robot joint configuration
        q_nominal = builder.add_parameter("qn", stretch_ndof)

        # Mobile base state
        base_state = builder.add_parameter("mobile_base_state", 3)

        # Path state
        path = builder.add_parameter("path", 3, n_points)

        return [q_current, q_nominal, base_state, path]

    @staticmethod
    def add_initial_constraints(builder: optas.OptimizationBuilder, stretch_name: str, qc: cs.SX, planar_mobile_base_name: str, base_init: cs.SX):
        """
        Adds initial configuration, initial velocity, and final velocity constraints in-place.

        Args:
            builder (optas.OptimizationBuilder): the optimization builder to update.
            stretch_name (str): name of the stretch optas variable
            planar_mobile_base_name (str): name of the mobile base optas variable
            base_init (SX): symbolic variable for the mobile base state
        """

        # Constraint: initial arm configuration
        builder.fix_configuration(stretch_name, config=qc)
        
        # Constraint: initial joint vel is zero
        builder.fix_configuration(stretch_name, time_deriv=1)

        # Constraint: final joint vel is zero
        dqF = builder.get_model_state(stretch_name, -1, time_deriv=1)
        builder.add_equality_constraint("final_joint_velocity", dqF)

        # Constraint: mobile base init position
        builder.fix_configuration(planar_mobile_base_name, config=base_init)

        # Constraint: mobile base init velocity
        builder.fix_configuration(planar_mobile_base_name, time_deriv=1)

        # Constraint: mobile base final velocity
        dxF = builder.get_model_state(planar_mobile_base_name, -1, time_deriv=1)
        builder.add_equality_constraint("final_velocity", dxF)

    @staticmethod
    def add_dynamics_constraints(builder: optas.OptimizationBuilder, stretch_name: str, planar_mobile_base_name: str, dt: float):
        """
        Adds dynamics constraints to the builder in-place.

        Args:
            builder (optas.OptimizationBuilder): the optimization builder to update.
            stretch_name (str): name of the stretch optas variable
            planar_mobile_base_name (str): name of the mobile base optas variable
            dt (float): trajectory timestep
        """

        # Constraint: dynamics
        builder.integrate_model_states(
            stretch_name,
            time_deriv=1,  # i.e. integrate velocities to positions
            dt=dt,
        )

        builder.integrate_model_states(planar_mobile_base_name, time_deriv=1, dt=dt)
            
    @staticmethod
    def get_symbolic_trajectory_variables(builder: optas.OptimizationBuilder, stretch_name: str, planar_mobile_base_name: str):
        """
        Returns the state matrices for the optimization.

        Args:
            builder (optas.OptimizationBuilder): the optimization builder to update.
            stretch_name (str): name of the stretch optas variable
            planar_mobile_base_name (str): name of the mobile base optas variable

        Returns:
            Q (SX): symbolic NxT (N = # DOF for arm, T = # time steps)
            Q_full (SX): symbolix NxT (N = # DOF for full robot, T = # time steps)
        """

        # Get joint trajectory
        Q = builder.get_model_states(stretch_name)  # ndof-by-T symbolic array for robot trajectory
        X = builder.get_model_states(planar_mobile_base_name)  # ndof-by-T symbolic array for base trajectory
        
        # Configuration of the full system i.e. mobile base and stretch arm
        Q_full = cs.vertcat(X, Q)

        return [Q, Q_full]
    
    @staticmethod
    def add_differential_drive_constraints(builder: optas.OptimizationBuilder, planar_mobile_base: PlanarMobileBase, n_points: int, dt: float):
        """
        Adds differential drive constraints to the optimization in-place.

        Args:
            builder (optas.OptimizationBuilder): the optimization builder to update.
            planar_mobile_base (motion_models.PlanarMobileBase): motion model for the mobile base
            n_points (int): number of points in trajectory
            dt (float): trajectory timestep
        """

        # Add decision variables for linear and angular velocities
        builder.add_decision_variables("v", n=n_points)  # Linear velocity
        builder.add_decision_variables("w", n=n_points)  # Angular velocity

        # Constraint: Add diff drive kinematics
        for t in range(n_points - 1):
            x = builder.get_model_state(planar_mobile_base.name, t)
            x_next = builder.get_model_state(planar_mobile_base.name, t + 1)
            v = builder._decision_variables["v"][t]
            w = builder._decision_variables["w"][t]
            dx, dy, dtheta = planar_mobile_base.diff_drive_kinematics(v, w, x[2])
            builder.add_equality_constraint(f"kinematics_{t}", x_next, 
                                            x + dt * cs.vertcat(dx, dy, dtheta),
                                            reduce_constraint=True)
                
    @staticmethod
    def add_end_effector_costs(builder: optas.OptimizationBuilder, stretch_full: optas.RobotModel, Q_full: cs.SX, path: cs.SX, link_ee: str, n_points: int):
        """
        Adds end effector costs to the optimization in-place.

        Args:
            builder (optas.OptimizationBuilder): the optimization builder to update.
            stretch_full (optas.RobotModel): full Stretch URDF RobotModel
            Q_full (SX): symbolic full state matrix
            path (SX): symbolic path matrix
            link_ee (str): target URDF link to track the trajectory
            n_points (int): length of trajectory (TODO: this is implicit in the SX variable sizes)
        """

        # End effector position trajectory
        pos_full = stretch_full.get_global_link_position_function(link_ee, n=n_points)
        pos_ee = pos_full(Q_full)  # 3-by-T position trajectory for end-effector (FK)

        builder.add_cost_term("ee_path", 1e4 * optas.sumsqr(path - pos_ee))
        #builder.add_cost_term("ee_terminal", 1e4 * optas.sumsqr(path[:,-1] - pos_ee[:,-1]))

    @staticmethod
    def add_orientation_costs(builder: optas.OptimizationBuilder, stretch_full: optas.RobotModel, Q_full: cs.SX, link_ee: str, n_points):
        """
        Adds end effector orientation costs to the optimization in-place.

        Args:
            builder (optas.OptimizationBuilder): the optimization builder to update.
            stretch_full (optas.RobotModel): full Stretch URDF RobotModel
            Q_full (SX): symbolic full state matrix
            link_ee (str): target URDF link to track the trajectory
            n_points (int): length of trajectory (TODO: this is implicit in the SX variable sizes)
        """
        
        # add orientation state
        orientation = builder.add_parameter("orientation", 4, n_points)

        # add quaternion error
        for i in range(n_points):
            # get desired quaternion from trajectory and the current end effector quaternion
            quat_desired = Quaternion(orientation[0, i], orientation[1, i], orientation[2, i], orientation[3, i])
            quat_current_ = stretch_full.get_global_link_quaternion(link_ee, Q_full[:, i])
            quat_current = Quaternion(quat_current_[0], quat_current_[1], quat_current_[2], quat_current_[3])

            # compute quaternion difference
            quat_error = quat_desired * quat_current.inv()

            # add cost as error RPY angle magnitude
            builder.add_cost_term(f"orientation_error{i}", 1e6 * optas.sumsqr(quat_error.getrpy()))

        # TODO: add rotational velocity?

    @staticmethod
    def add_other_costs(builder, qn: cs.SX, Q: cs.SX, stretch_name: str, planar_mobile_base_name: str, n_points: int):
        """
        Adds other costs to the optimization in-place.

        Args:
            builder (optas.OptimizationBuilder): the optimization builder to update.
            qn (SX): nominal configuration for the robot's joints
            Q (SX): symbolic variable for the robot arm state
            stretch_name (str): name of the stretch optas variable
            planar_mobile_base_name (str): name of the mobile base optas variable
            n_points (int): length of the trajectory
        """

        # Cost: deviation from nominal configuration
        Qn = cs.repmat(qn, 1, n_points)
        builder.add_cost_term("deviation_nominal_config", 0.1 * optas.sumsqr(Qn - Q))

        # Cost: minimize joint and base velocities
        dQ = builder.get_model_states(stretch_name, time_deriv=1)
        dX = builder.get_model_states(planar_mobile_base_name, time_deriv=1)
        builder.add_cost_term("min_join_vel", 0.01 * optas.sumsqr(dQ))
        builder.add_cost_term("min_base_vel", 0.01 * optas.sumsqr(dX))

def placeholder_trajectory(n: int=10) -> Trajectory:
    """
    Returns a trajectory with one position.

    Args:
        n (int): number of points (> 3)
    """
    position = [0., 0.3, 0.5]
    points = [Waypoint(position, i, 0.) for i in range(n)]
    trajectory = Trajectory(points)
    return trajectory

class Planner(Manager):
    def __init__(self, robot_urdf_dir: str, mode: OptimizerMode=OptimizerMode.POSITION):
        """
        Creates a planner for trajectories using OpTaS.

        Args:
            robot_urdf_dir (str): absolute path to a directory containing simplified and full 
            stretch URDFs.
            mode (OptimizerMode): mode to run in (position only or full position + orientation)
        """

        self.mode = mode
        self.init_robot(robot_urdf_dir)
        self.trajectory = placeholder_trajectory()
        
        # finalize init
        super().__init__()

    def _get_initial_robot_configuration(self):
        """
        Provides a nominal starting configuration for the robot.
        This is about midpoint for the major arm joints.

        Returns:
            list [q_current, q_nominal]: the same vector, twice, containing the starting config.
        """

        lift_height = 0.5
        arm_extention = 0.25
        wrist_yaw = optas.np.deg2rad(0.)
        wrist_pitch = optas.np.deg2rad(0.)
        wrist_roll = optas.np.deg2rad(0.)
        q_current = optas.np.array([lift_height, arm_extention/4, arm_extention/4,
                             arm_extention/4, arm_extention/4, wrist_yaw, wrist_pitch, wrist_roll])
        q_nominal = q_current

        return [q_current, q_nominal]

    def _transform_trajectory_global_to_ee(self, trajectory: Trajectory) -> Trajectory:
        """
        Transforms a trajectory specified in the global frame to the end effector frame.

        Args:
            trajectory (Trajectory): trajectory to transform.

        Returns:
            Trajectory: updated trajectory
        """

        # Get initial robot state qc, extract end effector position and rotation information
        qc, _ = self._get_initial_robot_configuration()

        ee_pos = cs.DM(self.stretch.get_global_link_position("link_grasp_center", qc)).full()
        ee_ori = cs.DM(self.stretch.get_global_link_rotation("link_grasp_center", qc)).full()  # rotation matrix
        ee_q = cs.DM(self.stretch.get_global_link_quaternion("link_grasp_center", qc)).full()
        ee_q = Quaternion(*ee_q)

        # Extract trajectory waypoints
        path = trajectory.get_xyz_series()
        ori = trajectory.get_orientation_series()
        n_points = path.shape[1]

        # apply rotations and translations
        for i in range(n_points):
            path[:, i] = ee_pos.flatten() + ee_ori @ path[:, i]
            ori_R = cs.DM(Rotation.from_quat(ori[:, i]).as_matrix())
            new_R = ee_ori @ ori_R
            new_q = Rotation.from_matrix(new_R).as_quat()
            ori[:, i] = new_q

        # update trajectory information
        trajectory.set_xyz(path)
        trajectory.set_orientation(ori)
        return trajectory

    def init_robot(self, robot_urdf_dir: str):
        """
        Loads the simplified and full robot models into member variables.

        Args:
            robot_urdf_dir (str): absolute path to the folder containing stretch.urdf and stretch_full.urdf
        """

        # Setup robot
        robot_model_input = {}
        robot_model_input["time_derivs"] = [
            0,
            1,
        ]  # i.e. joint position/velocity trajectory

        # TODO: figure out why there are two robot models
        simplified_robot_urdf = os.path.join(robot_urdf_dir, "stretch_re3.urdf")
        full_robot_urdf = os.path.join(robot_urdf_dir, "stretch_re3_full.urdf")
        robot_model_input["urdf_filename"] = simplified_robot_urdf
        stretch_robot_model_input = robot_model_input.copy()
        stretch_robot_model_input["urdf_filename"] = full_robot_urdf

        self.stretch = optas.RobotModel(**robot_model_input)
        self.stretch_full = optas.RobotModel(**stretch_robot_model_input)
        self.planar_mobile_base = PlanarMobileBase()

    def set_trajectory(self, trajectory: Trajectory, transform_to_ee: bool=True, refresh: bool=True):
        """
        Sets the trajectory. Do this before planning.

        Args:
            trajectory (Trajectory): new trajectory.
            transform_to_ee (bool): whether to transform a trajectory in the global reference frame
            to begin at the end effector's current frame.
            reset (bool): [recommended True] whether to automatically refresh the planner.
        """

        if transform_to_ee:
            trajectory = self._transform_trajectory_global_to_ee(trajectory)

        self.trajectory = trajectory
        
        if refresh:
            self.refresh()

    def setup_solver(self) -> optas.CasADiSolver:
        """
        Overloads Manager::setup_solver. Runs on init. Sets up the optimization variables.

        Returns:
            optas.CasADiSolver: numerical solver for the problem specified in the Optimizer class.
        """

        # get path information
        n_points = self.trajectory.get_length()
        _wp = self.trajectory.get_waypoints()
        dt = _wp[1].time - _wp[0].time

        # set up optimization builder
        optimizer = Optimizer()
        builder = optimizer.init_optimization_builder(stretch_simplified=self.stretch, stretch_full=self.stretch_full, planar_mobile_base=self.planar_mobile_base, n_points=n_points, dt=dt, mode=self.mode)

        # Setup solver
        optimization = builder.build()
        solver = optas.CasADiSolver(optimization).setup("ipopt")

        return solver

    def is_ready(self):
        return True

    def reset(self, qc, qn, path, orientation=None, mobile_base_init=[0, 0, 0]):
        """
        Resets the solver.

        Args:
            qc: current robot state.
            qn: nominal robot state.
            path: 3xN path for the robot to follow.
            mobile_base_init: initial [x,y,th] state for mobile base.
        """

        parameters_to_set = {"qc": optas.DM(qc), "qn": optas.DM(qn),
                                      "path": optas.DM(path),
                                      "mobile_base_init": optas.DM(mobile_base_init)}
        
        if self.mode == OptimizerMode.FULL and orientation is not None:
            parameters_to_set["orientation"] = optas.DM(orientation)

        self.solver.reset_parameters(parameters_to_set)
    
        # Set initial seed, note joint velocity will be set to zero
        n_points = self.trajectory.get_length()
        Q0 = optas.diag(qc) @ optas.DM.ones(self.stretch.ndof, n_points)
        self.solver.reset_initial_seed({f"{self.stretch.name}/q/x": Q0})

    def refresh(self):
        """
        Resets the Planner to its default state.
        TODO: figure out why we have to super().__init__() to fix state.
        """

        super().__init__()
        [q_current, q_nominal] = self._get_initial_robot_configuration()
        path = self.trajectory.get_xyz_series()
        
        orientation = None
        if self.mode == OptimizerMode.FULL:
            orientation = self.trajectory.get_orientation_series()

        self.reset(qc=q_current, qn=q_nominal, path=path, orientation=orientation)

    def get_target(self):
        return self.solution

    def plan(self):
        """
        Solves the optimization problem as configured by setup_solver().
        """

        self.solve()
        solution = self.get_target()
        stretch_plan, mobile_base_plan = solution[f"{self.stretch.name}/q"], \
               solution[f"{self.planar_mobile_base.name}/y"]
        
        stretch_full_plan = cs.vertcat(mobile_base_plan, stretch_plan)
        return stretch_full_plan