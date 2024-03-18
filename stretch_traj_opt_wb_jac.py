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

class SimpleLift(optas.TaskModel):
    def __init__(self):
        super().__init__(
            "simple_arm", 1, time_derivs=[0, 1, 2],
            dlim={0: [0, 1.1], 1: [-1, 1], 2:[-1, 1]}  # limits for position and velocity
        )
class SimpleExtension(optas.TaskModel):
    def __init__(self):
        super().__init__(
            "simple_extension", 1, time_derivs=[0, 1, 2],
            dlim={0: [0, 0.52], 1: [-1, 1], 2:[-1, 1]}  # limits for position and velocity
        )
class SimpleWristYaw(optas.TaskModel):
    def __init__(self):
        super().__init__(
            "simple_wrist_yaw", 1, time_derivs=[0, 1, 2],
            dlim={0: [-1.75, 4.0], 1: [-1, 1], 2:[-1, 1]}  # limits for position and velocity
        )

class Planner(Manager):
    def setup_solver(self):
        # Setup robot  ========================

        base_path = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(base_path, "urdf", "stretch.urdf")
        filename_stretch_full = os.path.join(base_path, "urdf", "stretch_full.urdf")
        link_ee = "link_grasp_center"

        # =====================================

        # Setup
        self.T = 200  # no. time steps in trajectory
        self.Tmax = 2  # total trajectory time
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

        self.stretch_name = self.stretch.get_name()

        self.planar_mobile_base = PlanarMobileBase()
        self.planar_mobile_base_name = self.planar_mobile_base.name
        base_lower_vel_limits, base_upper_vel_limits = self.planar_mobile_base.get_limits(1)
        base_lower_acc_limits, base_upper_acc_limits = self.planar_mobile_base.get_limits(2)

        self.simple_lift = SimpleLift()
        self.simple_lift_name = self.simple_lift.name
        lift_lower_pos_limits, lift_upper_pos_limits = self.simple_lift.get_limits(0)
        lift_lower_vel_limits, lift_upper_vel_limits = self.simple_lift.get_limits(1)

        self.simple_extension = SimpleExtension()
        self.simple_extension_name = self.simple_extension.name
        extension_lower_pos_limits, extension_upper_pos_limits = self.simple_extension.get_limits(0)
        extension_lower_vel_limits, extension_upper_vel_limits = self.simple_extension.get_limits(1)

        self.simple_wrist_yaw = SimpleWristYaw()
        self.simple_wrist_yaw_name = self.simple_wrist_yaw.name
        wrist_yaw_lower_pos_limits, wrist_yaw_upper_pos_limits = self.simple_wrist_yaw.get_limits(0)
        wrist_yaw_lower_vel_limits, wrist_yaw_upper_vel_limits = self.simple_wrist_yaw.get_limits(1)

        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=self.T, tasks=[self.planar_mobile_base,
                                                             self.simple_lift,
                                                             self.simple_extension,
                                                             self.simple_wrist_yaw], robots=[self.stretch])

        # Mobile base limits
        builder.enforce_model_limits(self.planar_mobile_base_name,
                                     1, 
                                     base_lower_vel_limits,
                                     base_upper_vel_limits)
        builder.enforce_model_limits(self.planar_mobile_base_name,
                                     2, 
                                     base_lower_acc_limits,
                                     base_upper_acc_limits)
        builder.enforce_model_limits(self.simple_lift_name,
                                     0, 
                                     lift_lower_pos_limits,
                                     lift_upper_pos_limits)
        builder.enforce_model_limits(self.simple_lift_name,
                                     1, 
                                     lift_lower_vel_limits,
                                     lift_upper_vel_limits)
        builder.enforce_model_limits(self.simple_extension_name,
                                     0, 
                                     extension_lower_pos_limits,
                                     extension_upper_pos_limits)
        builder.enforce_model_limits(self.simple_extension_name,
                                     1, 
                                     extension_lower_vel_limits,
                                     extension_upper_vel_limits)
        builder.enforce_model_limits(self.simple_wrist_yaw_name,
                                     0, 
                                     wrist_yaw_lower_pos_limits,
                                     wrist_yaw_upper_pos_limits)
        builder.enforce_model_limits(self.simple_wrist_yaw_name,
                                     1, 
                                     wrist_yaw_lower_vel_limits,
                                     wrist_yaw_upper_vel_limits)

        # Setup parameters
        qc = builder.add_parameter(
            "qc", self.stretch.ndof
        )  # current robot joint configuration
        qn = builder.add_parameter(
            "qn", self.stretch.ndof
        )  # nominal robot joint configuration
        base_init = builder.add_parameter("mobile_base_init", 3)
        path = builder.add_parameter("path", 3, self.T)

        # Constraint: mobile base init position
        builder.fix_configuration(self.planar_mobile_base.name, config=base_init)
        # Constraint: mobile base init velocity
        builder.fix_configuration(self.planar_mobile_base.name, time_deriv=1)
        # Constraint: mobile base final velocity
        dxF = builder.get_model_state(self.planar_mobile_base.name, -1, time_deriv=1)
        builder.add_equality_constraint("final_base_velocity", dxF)

        # Constraint: lift init position
        builder.fix_configuration(self.simple_lift.name, config=qc[0])
        # Constraint: lift init velocity
        builder.fix_configuration(self.simple_lift.name, time_deriv=1)
        # Constraint: lift final velocity
        dqlF = builder.get_model_state(self.simple_lift.name, -1, time_deriv=1)
        builder.add_equality_constraint("final_lift_velocity", dqlF)

        # Constraint: extension init position
        builder.fix_configuration(self.simple_extension.name, config=qc[1]+qc[2]+qc[3]+qc[4])
        # Constraint: extension init velocity
        builder.fix_configuration(self.simple_extension.name, time_deriv=1)
        # Constraint: extension final velocity
        dqeF = builder.get_model_state(self.simple_extension.name, -1, time_deriv=1)
        builder.add_equality_constraint("final_extension_velocity", dqeF)

        # Constraint: wrist_yaw init position
        builder.fix_configuration(self.simple_wrist_yaw.name, config=qc[5])
        # Constraint: wrist_yaw init velocity
        builder.fix_configuration(self.simple_wrist_yaw.name, time_deriv=1)
        # Constraint: wrist_yaw final velocity
        dqwF = builder.get_model_state(self.simple_wrist_yaw.name, -1, time_deriv=1)
        builder.add_equality_constraint("final_wrist_yaw_velocity", dqwF)

        # Decision variables
        builder.add_decision_variables("v", n=self.T)  # Linear velocity
        builder.add_decision_variables("w", n=self.T)  # Angular velocity

        #builder.integrate_model_states(self.planar_mobile_base_name, time_deriv=1, dt=self.dt)
        builder.integrate_model_states(self.simple_lift_name, time_deriv=1, dt=self.dt)
        builder.integrate_model_states(self.simple_extension_name, time_deriv=1, dt=self.dt)
        builder.integrate_model_states(self.simple_wrist_yaw_name, time_deriv=1, dt=self.dt)

        ################# Differntial Drive Constraints #################
        # Constraint: Add diff drive kinematics
        for t in range(self.T - 1):
            x = builder.get_model_state(self.planar_mobile_base.name, t)
            x_next = builder.get_model_state(self.planar_mobile_base.name, t + 1)
            v = builder._decision_variables["v"][t]
            w = builder._decision_variables["w"][t]
            dx, dy, dtheta = self.planar_mobile_base.diff_drive_kinematics(v, w, x[2])
            builder.add_equality_constraint(f"base_kinematics_{t}", x_next, 
                                            x + self.dt * cs.vertcat(dx, dy, dtheta),
                                            reduce_constraint=True)
        ############ Integrate Joint velocities into positions ###########

        for t in range(self.T - 1):
            v = builder._decision_variables["v"][t]
            w = builder._decision_variables["w"][t]

            x = builder.get_model_state(self.planar_mobile_base.name, t)
            x_next = builder.get_model_state(self.planar_mobile_base.name, t + 1)

            q1 = builder.get_model_state(self.simple_lift.name, t)
            q2 = builder.get_model_state(self.simple_extension.name, t)
            q3 = builder.get_model_state(self.simple_wrist_yaw.name, t)
            qd1 = builder.get_model_state(self.simple_lift.name, t, time_deriv=1)
            qd2 = builder.get_model_state(self.simple_extension.name, t, time_deriv=1)
            qd3 = builder.get_model_state(self.simple_lift.name, t, time_deriv=1)
            
            path_vel = (path[:, t + 1] - path[:, t]) / self.dt
            q_arm = cs.vertcat(q1, q2/4, q2/4, q2/4, q2/4, q3)
            full_body_jac = self.get_full_body_jacobian(q_arm)
            ee_vel = full_body_jac @ cs.vertcat(v, w, qd1, qd2/4, qd2/4, qd2/4, qd2/4, qd3)
            
            T_bw = cs.vertcat(cs.horzcat(cs.cos(x[2]), -cs.sin(x[2]), 0, x[0]),
                              cs.horzcat(cs.sin(x[2]), cs.cos(x[2]),  0, x[1]),
                              cs.horzcat(0,            0,             1, 0),
                              cs.horzcat(0,            0,             0, 1))
            Ad_Tbw = self.adjoint_matrix(T_bw)
            ee_vel_global = Ad_Tbw @ ee_vel

            q_arm_full = cs.vertcat(x[0], x[1], x[2], q_arm)
            p_ee = self.stretch_full.get_global_link_position("link_grasp_center", q_arm_full)
            p_ee_next = p_ee + self.dt * ee_vel_global[0:3]
            builder.add_cost_term(f"ee_pos_error{t}", 1e4 * optas.sumsqr(path[:,t+1] - p_ee_next))
            # builder.add_cost_term(f"ee_vel_error{t}", 1e3 * optas.sumsqr(ee_vel_global[0:3] - path_vel))

        # Q = cs.SX.zeros(self.stretch.ndof, self.T)
        # for t in range(self.T):
        #     q1 = builder.get_model_state(self.simple_lift.name, t)
        #     q2 = builder.get_model_state(self.simple_extension.name, t)
        #     q3 = builder.get_model_state(self.simple_wrist_yaw.name, t)
        #     Q[:, t] = cs.vertcat(q1, q2/4, q2/4, q2/4, q2/4, q3)
        # Qn = cs.repmat(qn, 1, self.T)
        # builder.add_cost_term("deviation_nominal_config", 0.1 * optas.sumsqr(Qn - Q))

        # Cost: minimize joint and base velocities
        dX = builder.get_model_states(self.planar_mobile_base_name, time_deriv=1)
        dql = builder.get_model_states(self.simple_lift_name, time_deriv=1)
        dqe = builder.get_model_states(self.simple_extension_name, time_deriv=1)
        dqw = builder.get_model_states(self.simple_wrist_yaw_name, time_deriv=1)
        #builder.add_cost_term("min_join_vel", 0.01 * optas.sumsqr(dQ))
        builder.add_cost_term("min_base_vel", 0.01 * optas.sumsqr(dX))
        builder.add_cost_term("min_lift_vel", 0.01 * optas.sumsqr(dql))
        builder.add_cost_term("min_extension_vel", 0.01 * optas.sumsqr(dqe))
        builder.add_cost_term("min_wrist_vel", 0.01 * optas.sumsqr(dqw))

        # Prevent rotation in end-effector
        # quatc = self.stretch.get_global_link_quaternion(link_ee, qc)
        # quat = self.stretch.get_global_link_quaternion_function(link_ee, n=self.T)
        # builder.add_equality_constraint("no_eff_rot", quat(Q), quatc)

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
        return solution[f"{self.planar_mobile_base_name}/y"], \
               solution[f"{self.simple_lift_name}/y"], \
               solution[f"{self.simple_extension_name}/y"], \
               solution[f"{self.simple_wrist_yaw_name}/y"]

    def adjoint_matrix(self, T):
        R = T[0:3, 0:3]
        p = T[0:3, 3]

        # Construct the skew-symmetric matrix [p]_x
        px = cs.vertcat(cs.horzcat(0, -p[2], p[1]),
                        cs.horzcat(p[2], 0, -p[0]),
                        cs.horzcat(-p[1], p[0], 0))

        # Construct the adjoint matrix Ad_T
        Ad_T = cs.vertcat(cs.horzcat(R, px @ R),
                          cs.horzcat(cs.SX.zeros(3, 3), R))

        return Ad_T

    def get_base_jacobian(self, q):
        T_0e = self.stretch.get_global_link_transform("link_grasp_center", q)
        T_0e_inv = invt(T_0e)

        F_6 = cs.vertcat(cs.horzcat(1, 0),
                         cs.horzcat(0, 0),
                         cs.horzcat(0, 0),
                         cs.horzcat(0, 0),
                         cs.horzcat(0, 0),
                         cs.horzcat(0, 1))
        
        Ad_T_0e = self.adjoint_matrix(T_0e_inv)


        jacobian_base = Ad_T_0e @ F_6
        return jacobian_base
    
    def get_full_body_jacobian(self, q):
        jacobian_arm = self.stretch.get_global_link_geometric_jacobian("link_grasp_center", q)
        jacobian_base = self.get_base_jacobian(q)
        return cs.horzcat(jacobian_base, jacobian_arm)



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
    mobile_base_plan, lift_plan, extension_plan, wrist_yaw_plan = planner.plan()
    stretch_full_plan = cs.vertcat(mobile_base_plan, lift_plan, extension_plan/4, extension_plan/4, extension_plan/4, extension_plan/4, wrist_yaw_plan)

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