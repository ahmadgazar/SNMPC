
# from contact_plan import create_contact_sequence
# import example_robot_data
# import numpy as np
# import pinocchio 

# robot = example_robot_data.load('talos')
# unactuatedJointNames =  [
#                      "arm_left_5_joint",
#                      "arm_left_6_joint",
#                      "arm_left_7_joint",
#                      "arm_right_5_joint",
#                      "arm_right_6_joint",
#                      "arm_right_7_joint",
#                      "gripper_left_joint",
#                      "gripper_right_joint",
#                      "head_1_joint",
#                      "head_2_joint"]

# actuatedJointNames = ['leg_left_1_joint',
#                        'leg_left_2_joint',
#                        'leg_left_3_joint',
#                        'leg_left_4_joint',
#                        'leg_left_5_joint',
#                        'leg_left_6_joint',
#                       'leg_right_1_joint',
#                       'leg_right_2_joint',
#                       'leg_right_3_joint',
#                       'leg_right_4_joint',
#                       'leg_right_5_joint',
#                       'leg_right_6_joint',
#                           'torso_1_joint',   
#                           'torso_2_joint', 
#                        'arm_left_1_joint', 
#                        'arm_left_2_joint',
#                        "arm_left_3_joint",
#                        "arm_left_4_joint", 
#                       'arm_right_1_joint', 
#                       'arm_right_2_joint',
#                       "arm_right_3_joint",
#                        "arm_right_4_joint"]   
# unactuatedJointIds = [i for (i, n) in enumerate(robot.model.names) if n in unactuatedJointNames]
# robot.model, [robot.collision_model, robot.visual_model] = pinocchio.buildReducedModel(
#         robot.model,
#         [robot.collision_model, robot.visual_model],
#         unactuatedJointIds,
#         robot.q0,
#     )
# robot.rebuildData() 
# rmodel = robot.model
# q0 = rmodel.referenceConfigurations['half_sitting'].copy()
# rmodel.name = 'talos_full'
# rmodel.type = 'HUMANOID'
# rmodel.foot_type = 'FLAT_FOOT'
# lxp = 0.10       # foot length in positive x direction
# lxn = -0.05      # foot length in negative x direction
# lyp = 0.05       # foot length in positive y direction
# lyn = -0.05      # foot length in negative y direction
# rdata = rmodel.createData()
# ee_frame_names = ['left_sole_link', 'right_sole_link']
# gravity_constant = -9.81 
# robot_mass = pinocchio.computeTotalMass(rmodel)
# mu = 0.5 # linear friction coefficient

# # centroidal state and control dimensions
# # ---------------------------------------
# n_u_per_contact = 6
# nb_contacts = 2
# n_u = nb_contacts*n_u_per_contact
# n_x = 9
# # noise parameters:
# # -----------------
# n_w = nb_contacts*3  # no. of contact position parameters
# # walking parameters:
# # -------------------
# dt = 0.01
# dt_ctrl = 0.001
# gait ={'type': 'PACE',
#         'stepLength': 0.,
#         'stepHeight': 0.03,
#         'stepKnots': 25,
#         'supportKnots': 9,
#         'nbSteps': 4}

# # planning and control horizon lengths:
# # -------------------------------------
# gait_templates, contact_sequence = create_contact_sequence(dt, gait, ee_frame_names, rmodel, rdata, q0)        
# N = int(round(contact_sequence[-1][0].t_end/dt, 2))
# N_mpc = (gait['stepKnots'] + (gait['supportKnots']))*4
# N_mpc_wbd = int(round(N_mpc/2, 2))
# N_ctrl = int((N-1)*(dt/dt_ctrl))

# # LQR gains (for stochastic control)      
# # ----------------------------------
# Q = np.diag([1e4, 1e4, 1e4, 
#              1e3, 1e3, 1e3, 
#              1e3, 1e3, 1e3])

# R = np.diag([1e2,1e3,1e1,
#              1e2,1e3,1e1,
#              1e2,1e3,1e1,
#              1e2,1e3,1e1])

# # contact position noise
# # discretizaton is done inside uncertainty propagation
# cov_w = np.diag([0.4**2, 0.4**2, 0.1**2,
#                  0.4**2, 0.4**2, 0.1**2,
#                  0.4**2, 0.4**2, 0.1**2,
#                  0.4**2, 0.4**2, 0.1**2])
# # discrete addtive noise
# cov_white_noise = dt*np.diag(np.array([0.85**2, 0.4**2, 0.01**2,
#                                        0.75**2, 0.4**2, 0.01**2,
#                                        0.85**2, 0.4**2, 0.01**2]))             
# beta_u = 0.01 # probability of constraint violation 

# # centroidal cost objective weights:
# # ----------------------------------
# state_cost_weights = np.diag([1e1, 1e1, 1e1, 1e2, 1e2, 1e2, 1e3, 1e3, 1e3])
# control_cost_weights = np.diag([1e-1, 1e-1, 1e1, 1e1, 1e0, 1e1,
#                                 1e-1, 1e-1, 1e1, 1e1, 1e0, 1e1])

# # whole-body cost objective weights:
# # ----------------------------------
# basisQWeight = [0, 0, 0, 80, 80, 80]
# legQWeight = [3, 3, 3, 3, 3, 3]
# torsoQWeight = [30, 30]
# armQWeight = [12, 12, 5, 5]
# basisVWeight = [0, 0, 0, 5, 5, 5] 
# legVWeight = [1] * 6
# torsoVWeight = [20] * 2
# armVWeight = [2, 2, 1, 1] 
# wbd_state_reg_weights = np.array( basisQWeight + legQWeight + legQWeight +
#                                   torsoQWeight + armQWeight + armQWeight +
#                                   basisVWeight + legVWeight + legVWeight +
#                                   torsoVWeight + armVWeight + armVWeight
#                                 )      
# whole_body_task_weights = {
#                             'swingFoot':{'preImpact':{'position':1e8,'velocity':1e4}, 
#                                             'impact':{'position':1e9, 'orientation':1e9,'velocity':1e11}
#                                            }, 
#                             'comTrack':1e7, 'stateBounds':1e3, 'centroidalTrack': 1e2, 
#                             'stateReg':{'stance':5e1, 'impact':1e2}, 'ctrlReg':{'stance':5e-1, 'impact':5e-1}, 
#                             'cop':1e6, 'frictionCone':20, 'contactForceTrack':600
#                             }    

# cameraTF = [3., 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]
# WITHDISPLAY = True 


from contact_plan import create_contact_sequence
import example_robot_data
import numpy as np
import pinocchio 

robot = example_robot_data.load('talos')
unactuatedJointNames =  [
                     "arm_left_5_joint",
                     "arm_left_6_joint",
                     "arm_left_7_joint",
                     "arm_right_5_joint",
                     "arm_right_6_joint",
                     "arm_right_7_joint",
                     "gripper_left_joint",
                     "gripper_right_joint",
                     "head_1_joint",
                     "head_2_joint"]

actuatedJointNames = ['leg_left_1_joint',
                       'leg_left_2_joint',
                       'leg_left_3_joint',
                       'leg_left_4_joint',
                       'leg_left_5_joint',
                       'leg_left_6_joint',
                      'leg_right_1_joint',
                      'leg_right_2_joint',
                      'leg_right_3_joint',
                      'leg_right_4_joint',
                      'leg_right_5_joint',
                      'leg_right_6_joint',
                          'torso_1_joint',   
                          'torso_2_joint', 
                       'arm_left_1_joint', 
                       'arm_left_2_joint',
                       "arm_left_3_joint",
                       "arm_left_4_joint", 
                      'arm_right_1_joint', 
                      'arm_right_2_joint',
                      "arm_right_3_joint",
                       "arm_right_4_joint"]   
unactuatedJointIds = [i for (i, n) in enumerate(robot.model.names) if n in unactuatedJointNames]
robot.model, [robot.collision_model, robot.visual_model] = pinocchio.buildReducedModel(
        robot.model,
        [robot.collision_model, robot.visual_model],
        unactuatedJointIds,
        robot.q0,
    )
robot.rebuildData() 
rmodel = robot.model
q0 = rmodel.referenceConfigurations['half_sitting'].copy()
rmodel.name = 'talos_full'
rmodel.type = 'HUMANOID'
rmodel.foot_type = 'FLAT_FOOT'
lxp = 0.10       # foot length in positive x direction
lxn = -0.05      # foot length in negative x direction
lyp = 0.05       # foot length in positive y direction
lyn = -0.05      # foot length in negative y direction
rdata = rmodel.createData()
ee_frame_names = ['left_sole_link', 'right_sole_link']
gravity_constant = -9.81 
robot_mass = pinocchio.computeTotalMass(rmodel)
mu = 0.5 # linear friction coefficient

# centroidal state and control dimensions
# ---------------------------------------
n_u_per_contact = 6
nb_contacts = 2
n_u = nb_contacts*n_u_per_contact
n_x = 9
# noise parameters:
# -----------------
n_w = nb_contacts*3  # no. of contact position parameters
# walking parameters:
# -------------------
dt = 0.01
dt_ctrl = 0.001
gait ={'type': 'PACE',
        'stepLength': 0.22,
        'stepHeight': 0.05,
        'stepKnots': 20,
        'supportKnots': 7,
        'nbSteps': 4}

# planning and control horizon lengths:
# -------------------------------------
gait_templates, contact_sequence = create_contact_sequence(dt, gait, ee_frame_names, rmodel, rdata, q0)        
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
state_cost_weights = np.diag([1e1, 1e1, 1e1, 1e2, 1e2, 1e2, 1e3, 1e3, 1e3])
control_cost_weights = np.diag([1e-1, 1e-1, 1e1, 1e1, 1e0, 1e1,
                                1e-1, 1e-1, 1e1, 1e1, 1e0, 1e1])

# whole-body cost objective weights:
# ----------------------------------
basisQWeight = [0, 0, 0, 80, 80, 80]
legQWeight = [3, 3, 3, 3, 3, 3]
torsoQWeight = [30, 30]
armQWeight = [12, 12, 5, 5]
basisVWeight = [0, 0, 0, 5, 5, 5] 
legVWeight = [1] * 6
torsoVWeight = [20] * 2
armVWeight = [2, 2, 1, 1] 
wbd_state_reg_weights = np.array( basisQWeight + legQWeight + legQWeight +
                                  torsoQWeight + armQWeight + armQWeight +
                                  basisVWeight + legVWeight + legVWeight +
                                  torsoVWeight + armVWeight + armVWeight
                                )      
whole_body_task_weights = {
                            'swingFoot':{'preImpact':{'position':1e8,'velocity':1e4}, 
                                            'impact':{'position':1e9, 'orientation':1e9,'velocity':1e11}
                                           }, 
                            'comTrack':1e7, 'stateBounds':1e3, 'centroidalTrack': 1e4, 
                            'stateReg':{'stance':5e1, 'impact':1e2}, 'ctrlReg':{'stance':5e-1, 'impact':5e-1}, 
                            'cop':1e6, 'frictionCone':20, 'contactForceTrack':500
                            }    

cameraTF = [3., 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]
WITHDISPLAY = True 