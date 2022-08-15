
from robot_properties_sarcos.config import HermesLowerConfig
from contact_plan import create_contact_sequence
import pinocchio as pin
import pybullet as p
import numpy as np
import yaml 
import sys

with open('bullet_default_hermes_lower.yaml') as config:
    config_file = yaml.safe_load(config)
robot_config = HermesLowerConfig()
pin_robot = robot_config.buildRobotWrapper()
rmodel = pin_robot.model
rmodel.name = 'hermes_lower'
rmodel.type = 'HUMANOID'
rmodel.foot_type = 'FLAT_FOOT'
rdata = rmodel.createData()
default_base_position = config_file['base_position']
default_base_orien = p.getQuaternionFromEuler(config_file['base_orientation'])
default_joint_config = config_file['joint_config']
num_joints = rmodel.nq # since its floating base
q0 = np.zeros(num_joints)
q0[0:3] = default_base_position
q0[3:7] = default_base_orien
for ji in range(num_joints-7):
    q0[ji+7] = config_file["default_pose"][robot_config.joint_names[ji]]
ee_frame_names = ['L_AAA', 'R_AAA']

# walking parameters:
# -------------------
dt = 0.02
dt_ctrl = 0.001
gait ={'type': 'PACE',
        'stepLength': 0.,
        'stepHeight': 0.1,
        'stepKnots': 15,
        'supportKnots': 5,
        'nbSteps': 4}
# whole-body cost objective weights:
# ----------------------------------      
whole_body_task_weights = {'footTrack':{'swing':1e8, 'impact':1e8}, 'impulseVel':1e6, 'comTrack':1e6, 'stateBounds':0e3, 
                            'stateReg':{'stance':1e1, 'impact':1e1}, 'ctrlReg':{'stance':1e-3, 'impact':1e-3}, 'frictionCone':10,
                            'centroidalTrack': 1e4, 'contactForceTrack':100}    

mu = 0.5 # linear friction coefficient
gait_templates, contact_sequence = create_contact_sequence(dt, gait, ee_frame_names, rmodel, rdata, q0)        
# planning and control horizon lengths:
# -------------------------------------
N = int(round(contact_sequence[-1][0].t_end/dt, 2))
N_ctrl = int((N-1)*(dt/dt_ctrl))    

cameraTF = [3., 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]
WITHDISPLAY = True 