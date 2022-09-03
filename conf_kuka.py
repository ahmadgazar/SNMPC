
from robot_properties_kuka.config import IiwaConfig
from pinocchio import casadi as cpin
import numpy as np 

q0 = np.array([ 1.5,  0.5,  0. , -1.8,  0. ,  0. ,  0. ])
target = np.array([-0.3, 0.5, 0.2])
target2 = np.array([ 0.3, 0.5, 0.2])
# robot
robot = IiwaConfig.buildRobotWrapper()
rmodel = robot.model
cmodel = cpin.Model(rmodel)
cdata = cmodel.createData()
# timing
dt = 0.01
dt_ctrl = 0.001
N_traj = 100
N_mpc  = 20
# cost function weights
state_cost_weight = np.diag(
    [1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0,
     5e0, 5e0, 5e0, 5e0, 5e0, 5e0, 5e0]
    )
control_cost_weight = np.diag([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])
end_effector_position_weight = np.diag([5e2, 5e2, 5e2])
ee_frame_name = 'contact'
# box constraints
state_limits = np.array([
    2.97, 2.09, 2.97, 2.09, 2.97, 2.09, 3.05, # q
    1.48, 1.48, 1.75, 1.31, 2.27, 2.36, 2.36  # qdot
    ])
control_limits = np.array([320, 320, 176, 176, 110, 40, 40])
