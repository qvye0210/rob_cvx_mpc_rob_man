# Robust Convex Model Predictive Control with collision avoidance guarantees for robot manipulators

<p align="center">
<img src="demo.gif"/>
</p>

This is the project repository for the paper:

https://arxiv.org/abs/2508.21677


### Overview of repository:
```bash
.
├── controllers
├── corridor_simulators
├── examples
│   ├── planar_two_dof
│   └── six_dof
└── path_planner
```

### Requirements:
- Mosek license (optional)
- Python 3.10

### Installation and setup:
1. Install python packages:

`pip install -r requirements.txt`

2. Add project folder to python path

`export PYTHONPATH="$(pwd):$PYTHONPATH`

### Examples:
#### Planar 2 DOF manipulator

The following demonstrates how to run the offline pipeline for a planar 2 DOF manipulator with 10 % uncertainty link masses.

##### Offline pipeline (optional)
1. A Mosek license is required to run the offline pipeline.
2. Install cvxpy to support Mosek:

`pip install cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS]`

3. Run offline pipline:

`python examples/planar_two_dof/1_run_offline_pipline.py`

4. List offline results 

`ls examples/planar_two_dof/data/dof_2_ef_0.1`

##### Online Corridor Control
1. Run all methods:

`python examples/planar_two_dof/2_run_all.py`

2. Print results:

`python examples/planar_two_dof/3_print_results.py`

3. Run animation of selected method:

`python examples/planar_two_dof/4_visualize_mpcs.py --method {method_name}`

where method_name is one of {"nom_star", "rt", "ft"}, where "ft" is our method.

#### General 6 DOF manipulator

The following demonstrates how run the offline pipeline for a general 6 DOF manipulator with 2 % uncertainty in link masses. It 
also demonstrates how to run the controller online with our corridor planning approach.

The collisions geometry for the manipulator is made to over-approximates the wrist joints, i.e., the last three DOFs, reducing the configuration-space to 3 DOF.
The example includes a pretrained 3 DOF nSCDF which uses spheres as obstacle representation.  


##### Offline pipeline (optional)
1. A Mosek license is required to run the offline pipeline.
2. Install cvxpy to support Mosek:

`pip install cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS]`

3. Run offline pipline:

`python examples/six_dof/1_run_offline_pipline.py`


##### Online Corridor Control
1. Install pytorch with CPU version.

`
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
`

2. Install additional libraries for animations and collision-detection.

`
pip install -r requirements_6D.txt
`

3. Run all methods and print results to terminal:

`python examples/six_dof/2_run_all.py`

###### World space animation
The animation shows the following visualizations:
1) Obstacles (gray spheres)
2) Start configuration (transparent red) 
3) Goal configuration (transparent green) 
4) Configuration along closed loop trajectory (gray)

Run animation of our controller:

`python examples/six_dof/3_show_world_space_motion.py`

###### Configuration space animation
The animation is visualized in the 3 dimensional configuration space, which are the first 3 DOF of the manipulator. The animation includes the following: 
1) Obstacle region (gray mesh)
2) Start and goal (red and green spheres)
2) Safe corridor (transparent red mesh) 
3) Centerline of corridor (black curve) 
4) Optimized MPC path at time step (blue curve)

Run animation of our controller:

`python examples/six_dof/4_show_conf_space_motion.py`

