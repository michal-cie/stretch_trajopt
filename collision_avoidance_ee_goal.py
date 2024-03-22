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
        self.Tmax = 4  # total trajectory time
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

        self.stretch_full = optas.RobotModel(**stretch_robot_model_input)
        
        self.stretch = optas.RobotModel(**robot_model_input)
        self.stretch_name = self.stretch.get_name()
        lower_pos_limits, upper_pos_limits = self.stretch.get_limits(0)
        lower_vel_limits, upper_vel_limits = self.stretch.get_limits(1)

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
        ee_goal = builder.add_parameter("ee_goal", 3, 1)

        # Constraint: mobile base init position
        builder.fix_configuration(self.planar_mobile_base.name, config=base_init)
        # Constraint: mobile base init velocity
        builder.fix_configuration(self.planar_mobile_base.name, time_deriv=1)

        # Constraint: initial arm configuration
        builder.fix_configuration(self.stretch_name, config=qc)
        # Constraint: initial joint vel is zero
        builder.fix_configuration(
            self.stretch_name, time_deriv=1
        )

        builder.integrate_model_states(self.planar_mobile_base_name, time_deriv=1, dt=self.dt)
        builder.integrate_model_states(
            self.stretch_name,
            time_deriv=1,  # i.e. integrate velocities to positions
            dt=self.dt,
        )

        # Get joint trajectory
        Q = builder.get_model_states(
            self.stretch_name
        )
        dQ = builder.get_model_states(
            self.stretch_name,
            time_deriv=1
        )
        X = builder.get_model_states(
            self.planar_mobile_base.name
        )
        dX = builder.get_model_states(
            self.planar_mobile_base_name, 
            time_deriv=1
        )

        ################# Differntial Drive Constraint #################
        # Constraint: Zero velocity in the direction orthogonal to base
        #             linear velocity
        for t in range(self.T - 1):
            x = builder.get_model_state(self.planar_mobile_base.name, t)
            dx = builder.get_model_state(self.planar_mobile_base.name, t, time_deriv=1)
            builder.add_equality_constraint(f"diff_drive{t}",-dx[0]*cs.sin(x[2]) + dx[1]*cs.cos(x[2]))
        
        ############ Integrate Joint velocities into positions ###########
        for t in range(self.T-1):
            x = builder.get_model_state(self.planar_mobile_base.name, t)
            dx = builder.get_model_state(self.planar_mobile_base.name, t, time_deriv=1)
            
            q_arm = Q[:, t]
            qd_arm = dQ[:, t]
            
            q_full = cs.vertcat(x, q_arm)
            
            # full_body_jac = self.stretch_full.get_global_link_linear_jacobian("link_grasp_center", q_full)
            # joint_rates = cs.vertcat(dx, qd_arm)
            # ee_vel = full_body_jac @ joint_rates

            p_ee = self.stretch_full.get_global_link_position("link_grasp_center", q_full)
            # p_ee_next = p_ee + self.dt * ee_vel

            builder.add_cost_term(f"ee_pos_error{t}", 1e4 * optas.sumsqr(ee_goal - p_ee))

        # Cost: minimize joint and base velocities
        builder.add_cost_term("min_base_vel", 1e-1 * optas.sumsqr(dX))
        builder.add_cost_term("min_joint_vel", 1e-1 * optas.sumsqr(dQ))

        # Prevent rotation in end-effector
        # quatc = self.stretch.get_global_link_quaternion(link_ee, qc)
        # quat = self.stretch.get_global_link_quaternion_function(link_ee, n=self.T)
        # builder.add_equality_constraint("no_eff_rot", quat(Q), quatc)

        obstacle_names = [f"obs{i}" for i in range(100)]
        link_names = ["link_grasp_center"]

        adapted_sphere_collision_avoidance_constraints(self.stretch_full, builder, cs.vertcat(X,Q),
                                                        obstacle_names, link_names=link_names)

        # Setup solver
        optimization = builder.build()
        solver = optas.CasADiSolver(optimization).setup("ipopt")

        return solver

    def is_ready(self):
        return True

    def reset(self, qc, qn, ee_goal, collision_params, mobile_base_init=[0, 0, 0]):
        
        params= {}
        params["qc"] = optas.DM(qc)
        params["qn"] = optas.DM(qn)
        params["ee_goal"] = optas.DM(ee_goal)
        params["mobile_base_init"] = optas.DM(mobile_base_init)

        params.update(collision_params)

        self.solver.reset_parameters(params)

        mobile_base_final = mobile_base_init + np.append(ee_goal[:2], 0)
        X0 = np.linspace(mobile_base_init, mobile_base_final, self.T).T

        # Set initial seed, note joint velocity will be set to zero
        Q0 = optas.diag(qc) @ optas.DM.ones(self.stretch.ndof, self.T)
        self.solver.reset_initial_seed({f"{self.stretch_name}/q/x": Q0})
        #self.solver.reset_initial_seed({f"{self.planar_mobile_base_name}/y/x": cs.DM(X0)})

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

    collision_params = {}

    num_obstacles = 100  # Parameterize the number of obstacles
    x_min, x_max = -1.0, -0.25
    y_min, y_max = -1.0, -0.5
    z_min, z_max =  0.0, 0.5

    # Generate obstacle names
    obstacle_names = [f"obs{i}" for i in range(num_obstacles)]
    link_names = ["link_grasp_center"]

    # Randomly generate obstacle positions
    obs = np.zeros((3, num_obstacles))
    obs[0, :] = np.random.uniform(x_min, x_max, num_obstacles)
    obs[1, :] = np.random.uniform(y_min, y_max, num_obstacles)
    obs[2, :] = np.random.uniform(z_min, z_max, num_obstacles)

    # Convert numpy array to optas.DM if necessary
    obs = optas.DM(obs)

    obsrad = 0.05 * optas.DM.ones(num_obstacles)
    linkrad = 0.2

    for link_name in link_names:
        collision_params[link_name + "_radii"] = linkrad

    for i, obstacle_name in enumerate(obstacle_names):
        collision_params[obstacle_name + "_position"] = obs[:, i]
        collision_params[obstacle_name + "_radii"] = obsrad[i]

    pn = cs.DM(planner.stretch.get_global_link_position("link_grasp_center", qc)).full()
    Rn = cs.DM(planner.stretch.get_global_link_rotation("link_grasp_center", qc)).full()
    t  = cs.DM(planner.t_).full()

    ee_goal = pn.flatten() + Rn @ np.array([0.5, -1.5, -0.2])

    planner.reset(qc, qn, ee_goal, collision_params)
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

    for i in range(num_obstacles):
        vis.sphere(radius=obsrad[i], position=obs[:, i], rgb=[0.0, 0.0, 1.0])

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

def get_one_point_path(T, x_end, y_end, z_end):
    path = np.zeros((3, T))

    for k in range(T):
        fraction = (k) / (T - 1)
        path[0, k] = 0 + fraction * (x_end)
        path[1, k] = 0 + fraction * (y_end)
        path[2, k] = 0 + fraction * (z_end)

    return path

def adapted_sphere_collision_avoidance_constraints(full_robot, builder, Q, 
                                                   obstacle_names, link_names=None):
    # Get model states
    n = Q.shape[1]

    if link_names is None:
        link_names = full_robot.link_names

    links = {}
    for name in link_names:
        links[name] = (
            full_robot.get_global_link_position_function(name),
            builder.add_parameter(name + "_radii"),
        )

    obstacles = {}
    for name in obstacle_names:
        obstacles[name] = (
            builder.add_parameter(name + "_position", 3),
            builder.add_parameter(name + "_radii"),
        )

    for t in range(n):
        q = Q[:, t]
        for link_name, (pos, linkrad) in links.items():
            p = pos(q)
            for obs_name, (obs, obsrad) in obstacles.items():
                name = f"sphere_col_avoid_{t}_{link_name}_{obs_name}"
                dist2 = cs.sumsqr(p - obs)
                bnd2 = (linkrad + obsrad) ** 2
                builder.add_leq_inequality_constraint(name, bnd2, dist2)

if __name__ == "__main__":
    main_arg = sys.argv[1] if len(sys.argv) > 1 else "figure_eight"
    sys.exit(main(main_arg))