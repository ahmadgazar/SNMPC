
from contact_plan import create_contact_sequence
import example_robot_data
import numpy as np
import pinocchio 

robot = example_robot_data.load('talos_legs')
rmodel = robot.model
rmodel.name = 'talos_legs'
rmodel.type = 'HUMANOID'
rmodel.foot_type = 'FLAT_FOOT'
lxp = 0.05      # foot length in positive x direction
lxn = -0.05      # foot length in negative x direction
lyp = 0.05      # foot length in positive y direction
lyn = -0.05      # foot length in negative y direction
rdata = rmodel.createData()
q0 = rmodel.referenceConfigurations['half_sitting'].copy()
ee_frame_names = ['right_sole_link', 'left_sole_link']
gravity_constant = -9.81 
robot_mass = pinocchio.computeTotalMass(rmodel)

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
dt = 0.02
dt_ctrl = 0.001
gait ={'type': 'PACE',
        'stepLength': 0.1,
        'stepHeight': 0.05,
        'stepKnots': 100,
        'supportKnots': 10,
        'nbSteps': 4}

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
state_cost_weights = np.diag([1e2, 1e2, 1e2, 1e1, 1e1, 1e1, 1e2, 1e2, 1e2])
control_cost_weights = np.diag([1e-1, 1e-1, 1e1, 1e1, 1e0, 1e1,
                                1e-1, 1e-1, 1e1, 1e1, 1e0, 1e1])

# whole-body cost objective weights:
# ----------------------------------      
whole_body_task_weights = {'footTrack':{'swing':1e8, 'impact':1e8}, 'impulseVel':1e7, 'comTrack':1e6, 'stateBounds':0e3, 
                            'stateReg':{'stance':1e1, 'impact':1e0}, 'ctrlReg':{'stance':1e-1, 'impact':1e0}, 'cop':20, 'frictionCone':20,
                            'centroidalTrack': 1e4, 'contactForceTrack':100}    

mu = 0.5 # linear friction coefficient
gait_templates, contact_sequence = create_contact_sequence(dt, gait, ee_frame_names, rmodel, rdata, q0)        
# planning and control horizon lengths:
# -------------------------------------
N = int(round(contact_sequence[-1][0].t_end/dt, 2))
N_ctrl = int((N-1)*(dt/dt_ctrl))    
N_mpc = 90#int(round(contact_sequence[2][0].t_end/dt, 2))
N_mpc_wbd = 40
cameraTF = [3., 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]
WITHDISPLAY = True 