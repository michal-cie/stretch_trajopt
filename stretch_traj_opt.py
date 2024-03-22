import os
import sys
import argparse
import numpy as np
import casadi as cs

import optas
from optas.spatialmath import *
from optas.visualize import Visualizer
from optas.templates import Manager

class PlanarMobileBase(optas.TaskModel):
    def __init__(self):
        super().__init__(
            "diff_drive", 3, time_derivs=[0, 1, 2],
            dlim={0: [-100, 100], 1: [-0.8, 0.8], 2:[-1, 1]}  # limits for position and velocity
        )
        
    def diff_drive_kinematics(self, v, w, theta):
        dx = v * cs.cos(theta)
        dy = v * cs.sin(theta)
        dtheta = w
        return [dx, dy, dtheta]

class Planner(Manager):
    def setup_solver(self):
        # Setup robot  ========================

        base_path = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(base_path, "urdf", "stretch.urdf")
        filename_stretch_full = os.path.join(base_path, "urdf", "stretch_full.urdf")
        link_ee = "link_grasp_center"

        # =====================================

        # Setup
        self.T = 40  # no. time steps in trajectory
        self.Tmax = 2.0  # total trajectory time
        t_ = optas.linspace(0, self.Tmax, self.T)
        self.t_ = t_
        self.dt = float((t_[1] - t_[0]).toarray()[0, 0])

        # Setup robot
        robot_model_input = {}
        robot_model_input["time_derivs"] = [
            0,
            1,
        ]  # i.e. joint position/velocity trajectory

        robot_model_input["urdf_filename"] = filename
        stretch_robot_model_input = robot_model_input.copy()
        stretch_robot_model_input["urdf_filename"] = filename_stretch_full

        self.stretch = optas.RobotModel(**robot_model_input)
        self.stretch_full = optas.RobotModel(**stretch_robot_model_input)
        lower_pos_limits, upper_pos_limits = self.stretch.get_limits(0)
        lower_vel_limits, upper_vel_limits = self.stretch.get_limits(1)

        self.stretch_name = self.stretch.get_name()

        self.planar_mobile_base = PlanarMobileBase()
        self.planar_mobile_base_name = self.planar_mobile_base.name
        base_lower_vel_limits, base_upper_vel_limits = self.planar_mobile_base.get_limits(1)
        base_lower_acc_limits, base_upper_acc_limits = self.planar_mobile_base.get_limits(2)

        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=self.T, tasks=[self.planar_mobile_base], robots=[self.stretch])

        # Stretch arm limits
        builder.enforce_model_limits(self.stretch_name,
                                     0,
                                     lower_pos_limits,
                                     upper_pos_limits)

        builder.enforce_model_limits(self.stretch_name,
                                     1,
                                     lower_vel_limits,
                                     upper_vel_limits)
        # Mobile base limits
        builder.enforce_model_limits(self.planar_mobile_base_name,
                                     1, 
                                     base_lower_vel_limits,
                                     base_upper_vel_limits)
        builder.enforce_model_limits(self.planar_mobile_base_name,
                                     2, 
                                     base_lower_acc_limits,
                                     base_upper_acc_limits)

        # Setup parameters
        qc = builder.add_parameter(
            "qc", self.stretch.ndof
        )  # current robot joint configuration
        qn = builder.add_parameter(
            "qn", self.stretch.ndof
        )  # nominal robot joint configuration
        base_init = builder.add_parameter("mobile_base_init", 3)
        path = builder.add_parameter("path", 3, self.T)
        q_full_init = cs.vertcat(base_init, qc)
        quat_init = self.stretch_full.get_global_link_quaternion("link_grasp_center", q_full_init)
        # Constraint: initial arm configuration
        builder.fix_configuration(self.stretch_name, config=qc)
        # Constraint: initial joint vel is zero
        builder.fix_configuration(
            self.stretch_name, time_deriv=1
        )  
        # Constraint: final joint vel is zero
        dqF = builder.get_model_state(self.stretch_name, -1, time_deriv=1)
        builder.add_equality_constraint("final_joint_velocity", dqF)

        # Constraint: mobile base init position
        builder.fix_configuration(self.planar_mobile_base.name, config=base_init)
        # Constraint: mobile base init velocity
        builder.fix_configuration(self.planar_mobile_base.name, time_deriv=1)
        # Constraint: mobile base final velocity
        dxF = builder.get_model_state(self.planar_mobile_base.name, -1, time_deriv=1)
        builder.add_equality_constraint("final_velocity", dxF)

        # Constraint: dynamics
        builder.integrate_model_states(
            self.stretch_name,
            time_deriv=1,  # i.e. integrate velocities to positions
            dt=self.dt,
        )
        builder.integrate_model_states(self.planar_mobile_base.name, 
                                       time_deriv=1, 
                                       dt=self.dt)

        # Get joint trajectory
        Q = builder.get_model_states(
            self.stretch_name
        )  # ndof-by-T symbolic array for robot trajectory
        X = builder.get_model_states(
            self.planar_mobile_base.name
        )  # ndof-by-T symbolic array for base trajectory
        # Configuration of the full system i.e. mobile base and stretch arm
        Q_full = cs.vertcat(X, Q)

        ################# Differntial Drive Constraints #################
        # Add decision variables for linear and angular velocities
        builder.add_decision_variables("v", n=self.T)  # Linear velocity
        builder.add_decision_variables("w", n=self.T)  # Angular velocity

        # Constraint: Add diff drive kinematics
        for t in range(self.T - 1):
            x = builder.get_model_state(self.planar_mobile_base.name, t)
            x_next = builder.get_model_state(self.planar_mobile_base.name, t + 1)
            v = builder._decision_variables["v"][t]
            w = builder._decision_variables["w"][t]
            dx, dy, dtheta = self.planar_mobile_base.diff_drive_kinematics(v, w, x[2])
            builder.add_equality_constraint(f"kinematics_{t}", x_next, 
                                            x + self.dt * cs.vertcat(dx, dy, dtheta),
                                            reduce_constraint=True)
        #################################################################
        
        ### example orientation constraint
        q_full_init = cs.vertcat(base_init, qc)
        quat_init = self.stretch_full.get_global_link_quaternion("link_grasp_center", q_full_init)
        quat_des = Quaternion(quat_init[0], quat_init[1], quat_init[2], quat_init[3])
        for t in range(self.T):
            quat_current_ = self.stretch_full.get_global_link_quaternion("link_grasp_center", Q_full[:, t])
            quat_current = Quaternion(quat_current_[0], quat_current_[1], quat_current_[2], quat_current_[3])
            quat_error = quat_des * quat_current.inv()
            builder.add_cost_term(f"orientation_error{t}", 1e4 * optas.sumsqr(quat_error.getrpy()))
        
        # End effector position trajectory
        pos_full = self.stretch_full.get_global_link_position_function(link_ee, n=self.T)
        pos_ee = pos_full(Q_full)  # 3-by-T position trajectory for end-effector (FK)

        builder.add_cost_term("ee_path", 1e4 * optas.sumsqr(path - pos_ee))
        #builder.add_cost_term("ee_terminal", 1e4 * optas.sumsqr(path[:,-1] - pos_ee[:,-1]))

        Qn = cs.repmat(qn, 1, self.T)
        builder.add_cost_term("deviation_nominal_config", 0.1 * optas.sumsqr(Qn - Q))

        # Cost: minimize joint and base velocities
        dQ = builder.get_model_states(self.stretch_name, time_deriv=1)
        dX = builder.get_model_states(self.planar_mobile_base_name, time_deriv=1)
        builder.add_cost_term("min_join_vel", 0.01 * optas.sumsqr(dQ))
        builder.add_cost_term("min_base_vel", 0.01 * optas.sumsqr(dX))

        # # Prevent rotation in end-effector
        # quatc = self.stretch.get_global_link_quaternion(link_ee, qc)
        # quat = self.stretch_full.get_global_link_quaternion_function(link_ee, n=self.T)
        # builder.add_equality_constraint("no_eff_rot", quat(Q_full), quatc)

        # Setup solver
        optimization = builder.build()
        solver = optas.CasADiSolver(optimization).setup("ipopt")

        return solver

    def is_ready(self):
        return True

    def reset(self, qc, qn, path, mobile_base_init=[0, 0, 0]):
        self.solver.reset_parameters({"qc": optas.DM(qc), "qn": optas.DM(qn),
                                      "path": optas.DM(path),
                                      "mobile_base_init": optas.DM(mobile_base_init)})

        # Set initial seed, note joint velocity will be set to zero
        Q0 = optas.diag(qc) @ optas.DM.ones(self.stretch.ndof, self.T)
        self.solver.reset_initial_seed({f"{self.stretch_name}/q/x": Q0})

    def get_target(self):
        return self.solution

    def plan(self):
        self.solve()
        solution = self.get_target()
        return solution[f"{self.stretch_name}/q"], \
               solution[f"{self.planar_mobile_base_name}/y"]

def main(arg="figure_eight"):
    planner = Planner()

    # Initial Arm Configuration
    lift_height = 0.5
    arm_extention = 0.25
    wrist_yaw = optas.np.deg2rad(0.0)
    qc = optas.np.array([lift_height, arm_extention/4, arm_extention/4,
                         arm_extention/4, arm_extention/4, wrist_yaw])
    qn = qc

    pn = cs.DM(planner.stretch.get_global_link_position("link_grasp_center", qc)).full()
    Rn = cs.DM(planner.stretch.get_global_link_rotation("link_grasp_center", qc)).full()

    t  = cs.DM(planner.t_).full()

    if arg == "figure_eight":
        path = get_figure_eight_path(planner.T, t, 0.3, 0.5, 0.1)
    elif arg == "three_point":
        path = get_three_point_path(planner.T, 
                                 -1.0, -1.0, -0.2, 
                                 0.0, -1.0, 0.0, 
                                 0.0, 0.0, 0.0)
    else:
        raise ValueError("Invalid option: choose between figure_eight and three_point")

    # Transform path to end effector in global frame
    for k in range(planner.T):
        path[:, k] = pn.flatten() + Rn @ path[:, k]

    planner.reset(qc, qn, path)
    stretch_plan, mobile_base_plan = planner.plan()
    stretch_full_plan = cs.vertcat(mobile_base_plan, stretch_plan)

    path_actual = np.zeros((3, planner.T))
    for k in range(planner.T):
        q_sol = np.array(stretch_full_plan[:, k])
        pn = cs.DM(planner.stretch_full.get_global_link_position("link_grasp_center", 
                                                                 q_sol)).full()    
        path_actual[:, k] = pn.flatten()

    # Optionally: interpolate between timesteps for smoother animation
    timestep_mult = 1 # 1 means no interpolation
    original_timesteps = np.linspace(0, 1, stretch_full_plan.size2())
    interpolated_timesteps = np.linspace(0, 1, 
                                         timestep_mult * stretch_full_plan.size2())
    interpolated_solution = cs.DM.zeros(stretch_full_plan.size1(), 
                                        len(interpolated_timesteps))
    for i in range(stretch_full_plan.size1()):
        interpolated_solution[i, :] = cs.interp1d(original_timesteps, 
                                                  stretch_full_plan[i, :].T, 
                                                  interpolated_timesteps)

    vis = Visualizer(camera_position=[3, 3, 3])

    for i in range(len(path[0, :])):
        vis.sphere(position=path[:, i], radius=0.01, rgb=[1, 0, 0])
        vis.sphere(position=path_actual[:, i], radius=0.01, rgb=[0, 1, 0])

    vis.grid_floor()
    vis.robot_traj(planner.stretch_full, np.array(interpolated_solution), 
                   animate=True, duration=planner.Tmax)
    vis.start()

    return 0

def get_figure_eight_path(T, t, x_amp, y_amp, z_amp):
    path = np.zeros((3, T))
    path[0, :] = x_amp * np.sin(t * np.pi/4).T  # need .T since t is col vec
    path[1, :] = y_amp * np.sin(t * np.pi/2).T  # need .T since t is col vec
    path[2, :] = z_amp * np.sin(t * 2*np.pi).T
    return path

def get_three_point_path(T, x_end1, y_end1, z_end1, 
                      x_end2, y_end2, z_end2, 
                      x_end3, y_end3, z_end3):
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

if __name__ == "__main__":
    main_arg = sys.argv[1] if len(sys.argv) > 1 else "figure_eight"
    sys.exit(main(main_arg))
