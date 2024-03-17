# stretch_trajopt
Example implementation of a whole body kinematic trajectory optimization for the stretch mobile manipulator.

### Mobile Base and Arm joint optimization
The main idea is to include the diff drive dynamics as constraints in the optimization and to add the linear and angular velocities as decision variables. The mobile base is represented by its 3 degrees of freedom x, y and theta and integrated forward with the constraints of diff drive motion. The errors in the end effector are then calculated by transforming the stretch arm onto the simulated mobile base. In the end you get two solutions one for the mobile base and one for the stretch arm but since they are jointly optimized you get a whole body motion that respects the diff drive constraints.

### Comments on solve time and practical use
The examples are relatively large optimizations with 200 timesteps over 20 seconds and are just meant for to show types of motions possible with this whole body optimitaion. On my pc the figure 8 trajectory is generated in about 30 sec and the 3 point path is generated in about 200 sec. A more realistic use of this is on smaller timescales with 20 timesteps over 2 seconds where the optimization time is less than one second, so it could be implemented as an MPC.

### Potential improvements
This is not the most efficient way to solve this problem because the optimizer does not have access to the whole body jacobians due to the base and arm being modeled separately. In Modern Robotics by Kevin Lynch chapter 13 he describes how to get the whole body jacobians for non holonomic mobile manipulators which should make this a more efficient optimization. This would no longer fit into the design of optas which this example is build upon so there it would require some more effort. Nontheless this repo should serve as a way to get started with generating whole body motions on the stretch, and more generally provides a simple template diff drive mobile manipulators.

This example builds upon https://github.com/cmower/optas/blob/master/example/figure_eight_plan.py and the stretch description from https://github.com/hello-robot.
### install dependencies
pip install pyoptas
### Notes
This example runs in python 3.8 and last i checked the optas visualization is broken in python 3.10
### to run the examples from the videos:
```python3 stretch_traj_opt.py figure_eight```  
```python3 stretch_traj_opt.py three_point```


https://github.com/michal-cie/stretch_trajopt/assets/90311578/75120602-262f-4d08-91e3-6aca515f0d0a

https://github.com/michal-cie/stretch_trajopt/assets/90311578/cfe14bba-c59f-4b52-a415-a6beed4e7587
