from robot_properties_kuka.config import IiwaConfig
from pinocchio import casadi as cpin
import numpy as np
import pinocchio as pin 


q0 = np.array([0., 1.57,  -1.57 , -1.2,  1.57 ,  -1.57 ,  0.37 ])
ee_target_pos = np.array([0.1, 0.4, 0.45])
ee_target_pos2 = np.array([ 0.4, 0.5, 0.2])
# robot
ee_frame_name = 'contact'
robot = IiwaConfig.buildRobotWrapper()
rmodel = robot.model
rdata = rmodel.createData()
pin.framesForwardKinematics(rmodel, rdata, q0)
ee_init_pos = rdata.oMf[rmodel.getFrameId(ee_frame_name)].translation  
cmodel = cpin.Model(rmodel)
cdata = cmodel.createData()
# timing
dt = 0.01
dt_ctrl = 0.001
N_traj = 300
N_mpc  = 20
# QR control
nq = q0.shape[0]
Q = np.eye(2*nq)
R = 0.1*np.eye(nq)
# uncertainty parameters 
cov_w_dt = 0.05*dt*np.eye(2*nq)
# cost function weights
IK_cost_weight = 5e-1
IK_Kp_gains = np.diag([0.5, 0.5, 0.5])
state_cost_weight = np.diag(
    [1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3,
     5e1, 5e1, 5e1, 5e1, 5e1, 5e1, 5e1]
    )
control_cost_weight = np.diag([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])
end_effector_position_weight = np.diag([1e3, 1e3, 1e3])
# box constraints
state_limits = np.array([
    2.97, 2.09, 2.97, 2.09, 2.97, 2.09, 3.05, # q
    1.48, 1.48, 1.75, 1.31, 2.27, 2.36, 2.36  # qdot
    ])
control_limits = np.array([320, 320, 176, 176, 110, 40, 40])
# obstacle positions
x_obs = np.array(
                [[0.65-0.05, -0.05, 0.4+0.05],     #top left                          
                    [0.65+0.05, -0.05, 0.4+0.05],  #top right
                    [0.65-0.05, -0.05, 0.4-0.05],  #bottom left
                    [0.65+0.05, -0.05, 0.4-0.05]]  #bottom right
                )