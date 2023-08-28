import numpy as np
import pinocchio as pin 
import example_robot_data 
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from robot_properties_solo.solo12wrapper import Solo12Config
from contact_plan import create_contact_sequence, create_climbing_contact_sequence, create_hiking_contact_sequence

# walking parameters:
# -------------------
dt = 0.01
dt_ctrl = 0.001
gait ={'type': 'BOUND',
       'terrain': 'HIKE',
      'stepLength' : 0.1,  
      'stepWidth' : 0.1,
      'stepHeight' : 0.05,
      'stepKnots' : 15,
      'supportKnots' : 10,  
      'nbSteps': 3}
mu = 0.5 # linear friction coefficient

# robot model and parameters
# --------------------------
robot_name = 'solo12'
ee_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
solo12 = example_robot_data.ROBOTS[robot_name]()
rmodel = solo12.robot.model
rmodel.type = 'QUADRUPED'
rmodel.foot_type = 'POINT_FOOT'
rdata = rmodel.createData()
robot_mass = pin.computeTotalMass(rmodel)

gravity_constant = -9.81 
max_leg_length = 0.34
step_adjustment_bound = 0.07  
heuristic_bound = 0.03                       
foot_scaling  = 1.
lxp = 0.01  # foot length in positive x direction
lxn = 0.01  # foot length in negative x direction
lyp = 0.01  # foot length in positive y direction
lyn = 0.01  # foot length in negative y direction

# centroidal state and control dimensions
# ---------------------------------------
n_u_per_contact = 3
nb_contacts = 4
nq = nb_contacts*n_u_per_contact 
q0 = np.array(Solo12Config.initial_configuration.copy())
q0[0] = 0.0
urdf = open('solo12.urdf', 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)
joint_names = kindyn.joint_names()
if gait['terrain'] == 'FLAT':
      gait_templates, contact_sequence = create_contact_sequence(
            dt, gait, ee_frame_names, rmodel, rdata, q0
            )
elif gait['terrain'] == 'CLIMB':
      gait_templates, contact_sequence = create_climbing_contact_sequence(
            dt, gait, ee_frame_names, rmodel, rdata, q0
            )
elif gait['terrain'] == 'HIKE':
      gait_templates, contact_sequence = create_hiking_contact_sequence(
            dt, gait, ee_frame_names, rmodel, rdata, q0
            )      
# planning and control horizon lengths:   
# -------------------------------------
N = int(round(contact_sequence[-1][0].t_end/dt, 2))
N_mpc = (gait['stepKnots']*3 + (gait['supportKnots']))
N_mpc_wbd = int(round(N_mpc/2, 2))
N_ctrl = int((N-1)*(dt/dt_ctrl))    
# LQR gains (for stochastic control)      
# ----------------------------------
Q = 1*np.eye(45)
R = 1e-1*np.eye(30)
n_x = Q.shape[0]
n_u = R.shape[0]
# noise parameters:
# -----------------
n_w = nb_contacts*3  # no. of contact position parameters
# uncertainty parameters 
cov_w_dt = dt*np.diag(
      [
            0e-1, 0e-1, 0e-1, #com
            0e-1, 0e-1, 0e-1, #linear_momentum 
            0e-1, 0e-1, 0e-1, #angular_momentum 

            0.3**2, 0.3**2, 0.3**2, #base position 
            0.2**2, 0.2**2, 0.2**2, #drelative base orientation

            0.7**2, 0.7**2, 0.7**2, #q_FL 
            0.7**2, 0.7**2, 0.7**2, #q_FR
            0.7**2, 0.7**2, 0.7**2, #q_HL
            0.7**2, 0.7**2, 0.7**2, #q_HR

            0.8**2, 0.8**2, 0.8**2, #base linear velocity 
            0.1**2, 0.1**2, 0.1**2, #base angular velocity

            0.7**2, 0.7**2, 0.7**2, #qdot_FL 
            0.7**2, 0.7**2, 0.7**2, #qdot_FR
            0.7**2, 0.7**2, 0.7**2, #qdot_HL
            0.7**2, 0.7**2, 0.7**2, #qdot_HR
      ]
)     
beta_u = 0.01 # probability of constraint violation 

# centroidal cost objective weights MPC:
# -------------------------------------
state_cost_weights = 2*np.diag([1e3, 1e3, 1e3,    #com
                                1e2, 1e2, 1e2,    #linear_momentum 
                                1e4, 1e4, 1e4,    #angular_momentum 
                              
                               1e3, 1e3, 1e3,     #base position 
                               5e1, 5e1, 5e1,     #drelative base position
                              
                               1e3, 1e3, 1e3,     #q_FL 
                               1e3, 1e3, 1e3,     #q_FR
                               1e3, 1e3, 1e3,     #q_HL
                               1e3, 1e3, 1e3,     #q_HR

                               1e2, 1e2, 1e2,     #base linear velocity 
                               1e2, 1e2, 1e2,     #base angular velocity
                              
                               2e2, 2e2, 2e2,     #qdot_FL 
                               2e2, 2e2, 2e2,     #qdot_FR
                               2e2, 2e2, 2e2,     #qdot_HL
                               2e2, 2e2, 2e2,     #qdot_HR

                              ])     

control_cost_weights = 2*np.diag([5e1, 1e1, 1e0,   #FL_forces
                                  5e1, 1e1, 1e0,   #FR_forces
                                  5e1, 1e1, 1e0,   #HL_forces
                                  5e1, 1e1, 1e0,   #HR_forces
                  
                                1e1, 1e1, 1e1,      #base linear acceleration
                                1e2, 1e2, 1e2,      #base angular acceleration
                                
                                5e-3, 5e-3, 5e-3,   #qddot_FL
                                5e-3, 5e-3, 5e-3,   #qddot_FR
                                5e-3, 5e-3, 5e-3,   #qddot_HL
                                5e-3, 5e-3, 5e-3    #qddot_HR
                                ])

swing_foot_cost_weights = 2*np.diag([1e2, 1e2, 1e2, #FL 
                                   1e2, 1e2, 1e2,   #FR
                                   1e2, 1e2, 1e2,   #HL
                                   1e2, 1e2, 1e2])  #HR
# acados slack penalties MPC:
# --------------------------- 

# slack penalties on linear constraints
L2_pen_g = np.array([0e3, 0e3, 0e0,
                     0e3, 0e3, 0e0,
                     0e3, 0e3, 0e0,
                     0e3, 0e3, 0e0])

L1_pen_g = np.array([1e4, 1e4, 3e4,
                     1e4, 1e4, 3e4,
                     1e4, 1e4, 3e4,
                     1e4, 1e4, 3e4])                                                                                              

# slack penalties on nonlinear constraints
L2_contact_location_lateral = 8*[0e0]
L2_contact_location_vertical = 4*[0e0]
L2_friction_pyramid = 16*[0e-2]
L2_friction_cone = 4*[0e-1]
L2_pen_frame_vel = 12*[0e0]
L2_pen_lin_mom = 3*[1e1]
L2_pen_ang_mom = 3*[1e1]
L2_pen_com = 3*[5e1]
L2_pen_h = np.array(
      L2_friction_cone +
      # L2_friction_pyramid +
      L2_pen_frame_vel +
      # L2_contact_location_lateral + 
      # L2_contact_location_vertical +
      L2_pen_lin_mom +
      L2_pen_ang_mom +
      L2_pen_com
      )
L1_contact_location_lateral = 8*[1e1]
L1_contact_location_vertical = 4*[1e1]
L1_friction_pyramid = 16*[1e2]
L1_friction_cone = 4*[1e3]
L1_pen_frame_vel = 12*[1e4]
L1_pen_lin_mom = 3*[0e0]
L1_pen_ang_mom = 3*[0e0]
L1_pen_com = 3*[0e0]
L1_pen_h = np.array(
      L1_friction_cone +
      # L1_friction_pyramid +
      L1_pen_frame_vel +
      # L1_contact_location_lateral + 
      # L1_contact_location_vertical + 
      L1_pen_lin_mom +
      L1_pen_ang_mom +
      L1_pen_com
      )   
L2_pen = np.concatenate([L2_pen_g, L2_pen_h])
L1_pen = np.concatenate([L1_pen_g, L1_pen_h])
# whole-body cost objective weights:
# ---------------------------------- 
freeFlyerQWeight = [0.]*3 + [500.]*3
freeFlyerVWeight = [10.]*6
legsQWeight = [0.01]*(rmodel.nv - 6)
legsWWeights = [1.]*(rmodel.nv - 6)
wbd_state_reg_weights = np.array(
      freeFlyerQWeight + legsQWeight + freeFlyerVWeight + legsWWeights
      )         

whole_body_task_weights = {
                            'swingFoot':{'preImpact':{'position':1e7,'velocity':0e1}, 
                                            'impact':{'position':1e7,'velocity':5e5}
                                           }, 
                            'comTrack':1e5, 'stateBounds':1e3, 'centroidalTrack': 1e4, 
                            'stateReg':{'stance':1e-1, 'impact':1e0}, 'ctrlReg':{'stance':1e-3, 'impact':1e-2}, 
                            'frictionCone':20, 'contactForceTrack':100
                            }                                                                        
# Gepetto viewer:
cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
WITHDISPLAY = True
WITH_MESHCAT_DISPLAY = True
WITHPLOT = False
SAVEDAT = False