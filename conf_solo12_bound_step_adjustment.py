import numpy as np
import pinocchio as pin 
import example_robot_data 
from contact_plan import create_contact_sequence
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from robot_properties_solo.solo12wrapper import Solo12Config

# walking parameters:
# -------------------
dt = 0.01
dt_ctrl = 0.001
gait = {'type': 'BOUND',
      'stepLength' : 0.15,
      'stepHeight' : 0.1,
      'stepKnots' : 17,
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
step_adjustment_bound = 0.075                          
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
n_u = 2*nq
n_x = 9 + nq
q0 = np.array(Solo12Config.initial_configuration.copy())
q0[0] = 0.0
urdf = open('solo12.urdf', 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)
joint_names = kindyn.joint_names()
gait_templates, contact_sequence =\
     create_contact_sequence(dt, gait, ee_frame_names, rmodel, rdata, q0)
# planning and control horizon lengths:
# -------------------------------------
N = int(round(contact_sequence[-1][0].t_end/dt, 2))
N_mpc = (gait['stepKnots'] + (gait['supportKnots']))*4
N_mpc_wbd = int(round(N_mpc/2, 2))
N_ctrl = int((N-1)*(dt/dt_ctrl))    
# LQR gains (for stochastic control)      
# ----------------------------------
Q = np.eye(27)
R = 0.1*np.eye(30)

# noise parameters:
# -----------------
n_w = nb_contacts*3  # no. of contact position parameters
# uncertainty parameters 
cov_w_dt = (0.3**2)*dt*np.eye(27)

# discrete addtive noise
cov_white_noise = dt*np.diag(np.array([0.85**2, 0.4**2, 0.01**2,
                                       0.75**2, 0.4**2, 0.01**2,
                                       0.85**2, 0.4**2, 0.01**2]))
beta_u = 0.01 # probability of constraint violation 

# centroidal cost objective weights:
# ----------------------------------
state_cost_weights = 2*np.diag([1e2, 1e2, 1e2,    #com
                                1e1, 1e1, 1e1,    #linear_momentum 
                                1e3, 1e3, 1e3,    #angular_momentum 
                              
                               1e-1, 1e-1, 1e-1,   #base position 
                               1e2, 1e2, 1e2,      #drelative base position
                              
                              5e2, 5e2, 5e2,       #q_FL 
                              5e2, 5e2, 5e2,       #q_FR
                              5e2, 5e2, 5e2,       #q_HL
                              5e2, 5e2, 5e2])      #q_HR

control_cost_weights = 2*np.diag([1e1, 1e1, 1e1,   #FL_forces
                                1e1, 1e1, 1e1,     #FR_forces
                                1e1, 1e1, 1e1,     #HL_forces
                                1e1, 1e1, 1e1,     #HR_forces
                  
                                1e-1, 1e-1, 1e-1,  #base linear velocity
                                1e-1, 1e-1, 1e-1,  #base angular velocity  
                                
                                3e2, 3e2, 3e2,    #qdot_FL
                                3e2, 3e2, 3e2,    #qdot_FR
                                3e2, 3e2, 3e2,    #qdot_HL
                                3e2, 3e2, 3e2     #qdot_HR
                                ])

swing_foot_cost_weights = 2*np.diag([1e2, 1e2, 1e2, #FL 
                                   1e2, 1e2, 1e2,   #FR
                                   1e2, 1e2, 1e2,   #HL
                                   1e2, 1e2, 1e2])  #HR                                 
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
                            'swingFoot':{'preImpact':{'position':1e6,'velocity':0e1}, 
                                            'impact':{'position':1e6,'velocity':2e1}
                                           }, 
                            'comTrack':10, 'stateBounds':0e3, 'centroidalTrack': 1e3, 
                            'stateReg':{'stance':1e-1, 'impact':1e0}, 'ctrlReg':{'stance':1e0, 'impact':1e1}, 
                            'frictionCone':2, 'contactForceTrack':80
                            }                                                                        
# Gepetto viewer:
cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
WITHDISPLAY = True
WITH_MESHCAT_DISPLAY = True
WITHPLOT = False
SAVEDAT = False