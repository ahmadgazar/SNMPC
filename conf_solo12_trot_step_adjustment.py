import os
import numpy as np
import pinocchio as pin 
import example_robot_data 
from contact_plan import create_contact_sequence
from robot_properties_solo.solo12wrapper import Solo12Config
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from casadi import Function
# walking parameters:
# -------------------
dt = 0.01
dt_ctrl = 0.001
gait ={'type': 'TROT',
      'stepLength' : 0.,
      'stepHeight' : 0.1,
      'stepKnots' : 15,
      'supportKnots' : 5,
      'nbSteps': 10}
mu = 0.5 # linear friction coefficient

# robot model and parameters
# --------------------------
robot_name = 'solo12'
ee_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
solo12 = example_robot_data.ROBOTS[robot_name]()
rmodel = solo12.robot.model
# for i in rmodel.names:
#       print(i)
rmodel.type = 'QUADRUPED'
rmodel.foot_type = 'POINT_FOOT'
rdata = rmodel.createData()
robot_mass = pin.computeTotalMass(rmodel)

gravity_constant = -9.81 
max_leg_length = 0.34                          
foot_scaling  = 1.
lxp = 0.01  # foot length in positive x direction
lxn = 0.01  # foot length in negative x direction
lyp = 0.01  # foot length in positive y direction
lyn = 0.01  # foot length in negative y direction

# get urdf path
urdf_filename = solo12.urdf_filename
urdf_path = os.path.join(
      solo12.model_path, 
      os.path.join(solo12.path, solo12.urdf_subpath), 
      urdf_filename
      )

# centroidal state and control dimensions
# ---------------------------------------
n_u_per_contact = 3
nb_contacts = 4
nq = nb_contacts*n_u_per_contact 
n_u = 2*nq
n_x = 9 + nq
q0 = np.array(Solo12Config.initial_configuration.copy())
q0[0] = 0.0
print
# pin.framesForwardKinematics(rmodel, rdata, q0[7:])
urdf = open('/home/agazar/devel/workspace/src/SNMPC/solo12.urdf', 'r').read()
# pin.ccrba(rmodel, rdata, q0, np.zeros(rmodel.nv))
# print(rdata.Ag.shape)
# print(rmodel.nv)
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)
joint_names = kindyn.joint_names()
print(joint_names)
print(urdf_filename)
print(urdf_path)
# if 'universe' in joint_names: joint_names.remove('universe')
# if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')
fk_FL = Function.deserialize(kindyn.fk(ee_frame_names[0]))
fk_FR = Function.deserialize(kindyn.fk(ee_frame_names[1]))
fk_HL = Function.deserialize(kindyn.fk(ee_frame_names[2]))
fk_HR = Function.deserialize(kindyn.fk(ee_frame_names[3]))
print('flFootPos = ', fk_FL(q=q0)['ee_pos'])
print('frFootPos = ', fk_FR(q=q0)['ee_pos'])
print('hlFootPos = ', fk_HL(q=q0)['ee_pos'])
print('hrFootPos = ', fk_HR(q=q0)['ee_pos'])

# pin.framesForwardKinematics(rmodel, rdata, q0)
# hlFootPos = rdata.oMf[rmodel.getFrameId(ee_frame_names[2])].translation
# hrFootPos = rdata.oMf[rmodel.getFrameId(ee_frame_names[3])].translation
# flFootPos = rdata.oMf[rmodel.getFrameId(ee_frame_names[0])].translation
# frFootPos = rdata.oMf[rmodel.getFrameId(ee_frame_names[1])].translation
# print('flFootPos = ', flFootPos)
# print('frFootPos = ', frFootPos)
# print('hlFootPos = ', hlFootPos)
# print('hrFootPos = ', hrFootPos)

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
Q = np.diag([1e4, 1e4, 1e4, 
             1e3, 1e3, 1e3, 
             1e3, 1e3, 1e3])

R = np.diag([1e2,1e3,1e1,
             1e2,1e3,1e1,
             1e2,1e3,1e1,
             1e2,1e3,1e1])

# noise parameters:
# -----------------
n_w = nb_contacts*3  # no. of contact position parameters
# contact position noise
# discretizaton is done inside uncertainty propagation
cov_w = np.diag([0.4**2, 0.4**2, 0.1**2,
                 0.4**2, 0.4**2, 0.1**2,
                 0.4**2, 0.4**2, 0.1**2,
                 0.4**2, 0.4**2, 0.1**2])
# discrete addtive noise
cov_white_noise = dt*np.diag(np.array([0.85**2, 0.4**2, 0.01**2,
                                       0.75**2, 0.4**2, 0.01**2,
                                       0.85**2, 0.4**2, 0.01**2]))
beta_u = 0.01 # probability of constraint violation 

# centroidal cost objective weights:
# ----------------------------------
state_cost_weights = np.diag([5e1, 5e1, 5e1,  #com
                              1e2, 1e2, 1e2,  #linear_momentum 
                              1e2, 1e2, 1e2,  #angular_momentum 
                              1e3, 1e3, 1e3,  #q_FL 
                              1e3, 1e3, 1e3,  #q_FR
                              1e3, 1e3, 1e3,  #q_HL
                              1e3, 1e3, 1e3]) #q_HR

control_cost_weights = np.diag([5e0, 1e0, 1e0,    #FL_forces
                                5e0, 1e0, 1e0,    #FR_forces
                                5e0, 1e0, 1e0,    #HL_forces
                                5e0, 1e0, 1e0,    #HR_forces
                                1e1, 1e1, 1e1, #qdot_FL
                                1e1, 1e1, 1e1, #qdot_FR
                                1e1, 1e1, 1e1, #qdot_HL
                                1e1, 1e1, 1e1  #qdot_HR
                                ]) 
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
