import numpy as np
import pinocchio as pin
import jax.numpy as jnp 
import matplotlib.pyplot as plt
from collections import namedtuple
from utils import compute_5th_order_poly_traj

class Debris():
    def __init__(self, CONTACT, t_start=0.0, t_end=1.0, 
                     x=None, y=None, z=None, axis=None, 
                             angle=None, ACTIVE=False):
        """
        Minimal helper function: return the SE3 configuration of a stepstone, with some
        ad-hoc configuration.
        """
        if ACTIVE:
            STEP = 1.0
            axis = np.array(axis, np.float64)
            axis /= np.linalg.norm(axis)
            self.axis = axis
            self.pose = pin.SE3(pin.AngleAxis(angle, np.concatenate([axis, [0]])).matrix(),
                            np.array([x * STEP, y * STEP, z]))
        self.t_start = t_start 
        self.t_end = t_end
        self.CONTACT = CONTACT
        self.ACTIVE = ACTIVE 
        self.__fill_contact_idx()

    def __fill_contact_idx(self):
        if self.CONTACT == 'RF' or self.CONTACT == 'FR':
            self.idx = 0
        elif self.CONTACT == 'LF' or self.CONTACT == 'FL':
            self.idx = 1
        elif self.CONTACT == 'HR':
            self.idx = 2
        elif self.CONTACT == 'HL':
            self.idx = 3                                     
    
# given a contact plan, fill a contact trajectory    
def create_contact_trajectory(conf):
    contact_sequence = conf.contact_sequence
    contact_trajectory = dict([(foot.CONTACT, []) for foot in  contact_sequence[0]])
    for contacts in contact_sequence:
        for contact in contacts:
            contact_duration = int(round((contact.t_end-contact.t_start)/conf.dt))  
            for time in range(contact_duration):
                contact_trajectory[contact.CONTACT].append(contact)  
    return contact_trajectory                

def interpolate_contact_trajectory(conf, contact_trajectory):
    N_outer = len(contact_trajectory['FR'])
    N_inner = int(conf.dt/conf.dt_ctrl)
    N_interpol = (N_outer-1)*N_inner
    contact_trajectory_interpol = dict(FL = np.empty((N_interpol, 3)),
                                       FR = np.empty((N_interpol, 3)),
                                       HL = np.empty((N_interpol, 3)),
                                       HR = np.empty((N_interpol, 3)))
    for contact_name in contact_trajectory:
        contact = contact_trajectory[contact_name]
        for i in range(N_outer-1):
            if contact[i].ACTIVE:
                contact_pos = contact[i].pose.translation
            else:
                contact_pos = np.zeros(3)
            for j in range(N_inner):
                k =  i*N_inner + j
                contact_trajectory_interpol[contact_name][k] = contact_pos
    return contact_trajectory_interpol

def compute_foot_traj(conf):
    step_height = conf.step_height
    dt_ctrl = conf.dt_ctrl
    contact_sequence = conf.contact_sequence
    N_ctrl = conf.N_ctrl
    foot_traj_dict = dict([(foot.CONTACT, dict(x=np.zeros((3, N_ctrl)), x_dot=np.zeros((3, N_ctrl)), 
                                     x_ddot=np.zeros((3, N_ctrl)))) for foot in  contact_sequence[0]])
    previous_contact_sequence = contact_sequence[0]
    for i, contacts in enumerate(contact_sequence):
        if i < len(contact_sequence)-1:
            next_contact_sequence = contact_sequence[i+1]
        else:
           next_contact_sequence = contact_sequence[i]     
        for contact in contacts:
            t_start_idx = int(contact.t_start/dt_ctrl)
            t_end_idx = int(contact.t_end/dt_ctrl)
            N_contact = int((contact.t_end-contact.t_start)/dt_ctrl)
            # foot is in contact 
            if contact.ACTIVE:
                foot_traj_dict[contact.CONTACT]['x'][:, t_start_idx:t_end_idx] = np.tile(contact.pose.translation, (N_contact,1)).T 
            # foot is in the air
            elif not contact.ACTIVE:
                x0 = previous_contact_sequence[contact.idx].pose.translation 
                x1 = next_contact_sequence[contact.idx].pose.translation 
                # x and y directions
                x, xdot, xddot = compute_5th_order_poly_traj(x0[:2], x1[:2], (contact.t_end-contact.t_start), dt_ctrl)
                foot_traj_dict[contact.CONTACT]['x'][:2, t_start_idx:t_end_idx] = x
                foot_traj_dict[contact.CONTACT]['x_dot'][:2, t_start_idx:t_end_idx] = xdot
                foot_traj_dict[contact.CONTACT]['x_ddot'][:2, t_start_idx:t_end_idx] = xddot
                # z direction (interpolate half way from zero to a step height)
                x_up, xdot_up, xddot_up = compute_5th_order_poly_traj(np.array([0.]), np.array([step_height]), 0.5*(contact.t_end-contact.t_start), dt_ctrl)
                foot_traj_dict[contact.CONTACT]['x'][2, t_start_idx:t_start_idx+int(0.5*N_contact)] = x_up
                foot_traj_dict[contact.CONTACT]['x_dot'][2, t_start_idx:t_start_idx+int(0.5*N_contact)] = xdot_up
                foot_traj_dict[contact.CONTACT]['x_ddot'][2, t_start_idx:t_start_idx+int(0.5*N_contact)] = xddot_up
                # z direction (interpolate half way back from a step height to the ground)
                x_down, xdot_down, xddot_down = compute_5th_order_poly_traj(np.array([step_height]), np.array([0.]), 0.5*(contact.t_end-contact.t_start), dt_ctrl)
                foot_traj_dict[contact.CONTACT]['x'][2, t_start_idx+int(0.5*N_contact):t_end_idx] = x_down 
                foot_traj_dict[contact.CONTACT]['x_dot'][2, t_start_idx+int(0.5*N_contact):t_end_idx] = xdot_down 
                foot_traj_dict[contact.CONTACT]['x_ddot'][2, t_start_idx+int(0.5*N_contact):t_end_idx] = xddot_down 
        previous_contact_sequence = contact_sequence[i]        
    return foot_traj_dict 

def create_contact_sequence(dt, gait, ee_frame_names, rmodel, rdata, q0):
      gait_templates = []
      steps = gait['nbSteps']
      if gait['type'] == 'TROT':
            for step in range (steps):
                  if step < steps-1:
                        gait_templates += [['doubleSupport', 'rflhStep', 'doubleSupport', 'lfrhStep']]
                  else:
                        gait_templates += [['doubleSupport', 
                                            'rflhStep', 'doubleSupport', 
                                            'lfrhStep', 'doubleSupport']]
      elif gait['type'] =='PACE':
            if rmodel.name == 'solo':
                for step in range (steps):
                    if step < steps-1:
                        gait_templates += [['doubleSupport', 'rfrhStep', 'doubleSupport', 'lflhStep']]
                    else:
                        gait_templates += [['doubleSupport', 
                                            'rfrhStep', 'doubleSupport', 
                                            'lflhStep', 'doubleSupport']]
            # elif rmodel.name == 'bolt' or rmodel.name == 'bolt_humanoid':
            elif rmodel.name == 'talos':
                for step in range (steps):
                    if step < steps-1:
                        gait_templates += [['doubleSupport', 'rfStep', 'doubleSupport', 'lfStep']]
                    else:
                        gait_templates += [['doubleSupport', 
                                            'rfStep', 'doubleSupport', 
                                            'lfStep', 'doubleSupport']]                           
      elif gait['type'] == 'BOUND':
            for step in range (steps):
                  if step < steps-1:
                        gait_templates += [['doubleSupport', 'rflfStep', 'doubleSupport', 'rhlhStep']]
                  else:
                        gait_templates += [['doubleSupport', 
                                            'rflfStep', 'doubleSupport', 
                                            'rhlhStep', 'doubleSupport']]              
      pin.forwardKinematics(rmodel, rdata, q0)
      pin.updateFramePlacements(rmodel, rdata)
      if rmodel.name == 'solo':
        hlFootPos = rdata.oMf[rmodel.getFrameId(ee_frame_names[2])].translation
        hrFootPos = rdata.oMf[rmodel.getFrameId(ee_frame_names[3])].translation
      flFootPos = rdata.oMf[rmodel.getFrameId(ee_frame_names[0])].translation
      frFootPos = rdata.oMf[rmodel.getFrameId(ee_frame_names[1])].translation
      t_start = 0.0 
      contact_sequence = []
      stepKnots, supportKnots = gait['stepKnots'], gait['supportKnots']
      stepLength = gait['stepLength']
      for gait in gait_templates:
            for phase in gait:
                  contact_sequence_k = []
                  if rmodel.name == 'solo' and phase == 'doubleSupport':
                    t_end = t_start + supportKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0], 
                                         y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0], 
                                         y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, x=hrFootPos[0],
                                          y=hrFootPos[1], z=hrFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, x=hlFootPos[0],  
                                          y=hlFootPos[1], z=hlFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                #   elif rmodel.name == 'bolt' or rmodel.name == 'bolt_humanoid' and phase == 'doubleSupport':
                  elif rmodel.name == 'talos'  and phase == 'doubleSupport':
                    t_end = t_start + supportKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0], 
                                          y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0],  
                                          y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))        
                  elif phase == 'rflhStep':
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0],  
                                          y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, x=hrFootPos[0], 
                                          y=hrFootPos[1], z=hrFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    frFootPos[0] += stepLength
                    hlFootPos[0] += stepLength
                  elif phase == 'rfStep':
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0], 
                                          y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    frFootPos[0] += stepLength
                  elif phase == 'lfrhStep':
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0],
                                          y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, x=hlFootPos[0], 
                                          y=hlFootPos[1], z=hlFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    flFootPos[0] += stepLength
                    hrFootPos[0] += stepLength      
                  elif phase == 'lfStep':
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0],  
                                          y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    flFootPos[0] += stepLength
                  elif phase == 'rfrhStep':
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0],  
                                         y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, x=hlFootPos[0],  
                                         y=hlFootPos[1], z=hlFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    frFootPos[0] += stepLength
                    hrFootPos[0] += stepLength      
                  elif phase == 'lflhStep':
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0],  
                                          y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, x=hrFootPos[0], 
                                          y=hrFootPos[1], z=hrFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    flFootPos[0] += stepLength
                    hlFootPos[0] += stepLength
                  elif phase == 'rflfStep':
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, x=hrFootPos[0],  
                                          y=hrFootPos[1], z=hrFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, x=hlFootPos[0],  
                                          y=hlFootPos[1], z=hlFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    frFootPos[0] += stepLength
                    flFootPos[0] += stepLength      
                  elif phase == 'rhlhStep':
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0],  
                                          y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0], 
                                          y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))                        
                    contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    hrFootPos[0] += stepLength
                    hlFootPos[0] += stepLength
                  else:
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    frFootPos[0] += stepLength
                    flFootPos[0] += stepLength
                    hrFootPos[0] += stepLength      
                    hlFootPos[0] += stepLength 
                  t_start = t_end
                  contact_sequence += [contact_sequence_k] 
      return gait_templates, contact_sequence

def plot_swing_foot_traj(swing_foot_dict, conf):
    dt = conf.dt_ctrl
    for contact in swing_foot_dict:
        plt.rc('text', usetex = True)
        plt.rc('font', family ='serif')   
        fig, (p_x, v_x, a_x, p_y, v_y, a_y, p_z, v_z, a_z) = plt.subplots(9, 1, sharex=True) 
        px = swing_foot_dict[contact]['x'][0, :]
        py = swing_foot_dict[contact]['x'][1, :]
        pz = swing_foot_dict[contact]['x'][2, :]
        vx = swing_foot_dict[contact]['x_dot'][0, :]
        vy = swing_foot_dict[contact]['x_dot'][1, :]
        vz = swing_foot_dict[contact]['x_dot'][2, :]
        ax = swing_foot_dict[contact]['x_ddot'][0, :]
        ay = swing_foot_dict[contact]['x_ddot'][1, :]
        az = swing_foot_dict[contact]['x_ddot'][2, :]
        time = np.arange(0, np.round((px.shape[0])*dt, 2), dt)
        # end-effector positions
        p_x.plot(time, px)
        p_x.set_title('p$_x$')
        p_y.plot(time, py)
        p_y.set_title('p$_y$')
        p_z.plot(time, pz)
        p_z.set_title('p$_z$')
        # end-effector velocities
        v_x.plot(time, vx)
        v_x.set_title('v$_x$')
        v_y.plot(time, vy)
        v_y.set_title('v$_y$')
        v_z.plot(time, vz)
        v_z.set_title('v$_z$')
        # end-effector accelerations
        a_x.plot(time, ax)
        a_x.set_title('a$_x$')
        a_y.plot(time, ay)
        a_y.set_title('a$_y$')
        a_z.plot(time, az)
        a_z.set_title('a$_z$')
    plt.show()
 
def fill_debris_list(conf):
    Debri = namedtuple('Debris', 'LOGIC, R, p')  
    outer_tuple_list = []
    contact_trajectory = create_contact_trajectory(conf)
    for time_idx in range(conf.N):
        contacts_logic_k = []
        contacts_position_k = []
        contacts_orientation_k = [] 
        inner_tuple_list = []
        for contact in contact_trajectory:
            if contact_trajectory[contact][time_idx].ACTIVE:
                contact_logic = 1
                R = contact_trajectory[contact][time_idx].pose.rotation
                p = contact_trajectory[contact][time_idx].pose.translation
            else:
                contact_logic = 0
                R = jnp.zeros((3,3))
                p = jnp.zeros(3)
            contacts_logic_k.append(contact_logic)
            contacts_orientation_k.append(R)
            contacts_position_k.append(p) 
            inner_tuple_list.append(Debri(contacts_logic_k, contacts_orientation_k, contacts_position_k))                                
        outer_tuple_list.append(inner_tuple_list)
    return outer_tuple_list


