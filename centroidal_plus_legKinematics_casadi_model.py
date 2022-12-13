from utils import construct_friction_pyramid_constraint_matrix
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from contact_plan import create_contact_trajectory
from casadi import *
import numpy as np

# experimental
class LegImpedanceController(Callback):
    def __init__(self, name, kindyn, Kp, Kd, FRAME_NAME, leg_joint_indices, opts={}):
        Callback.__init__(self)   
        LOCAL_WORLD_ALIGNED = cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
        self.leg_name = str(FRAME_NAME[:2])
        self.J =  Function.deserialize(
                kindyn.jacobian(FRAME_NAME, LOCAL_WORLD_ALIGNED)
                )
        self.ee_pos = Function.deserialize(kindyn.fk(FRAME_NAME))
        self.ee_vel = Function.deserialize(
                kindyn.frameVelocity(FRAME_NAME, LOCAL_WORLD_ALIGNED)
                )         
        self.leg_joint_indices = leg_joint_indices
        self.Kp = diag(Kp)
        self.Kd = diag(Kd)       
        self.construct(name, opts)
    
    def get_n_in(self):
        return 5
    
    def get_n_out(self):
        return 1
    
    # Number of inputs and outputs
    def get_sparsity_in(self, i):
        if i == 0:
            return Sparsity.dense(19,1)  #generalized position vector
        elif i == 1:    
            return Sparsity.dense(18,1)  #generalized velocity vector
        elif i == 2:
            return Sparsity.dense(3, 1)  #desired forces
        elif i == 3:
            return Sparsity.dense(3, 1)  #desired ee position
        elif i == 4:
            return Sparsity.dense(3, 1)  #desired ee velocity    

    def get_sparsity_out(self, i): 
        return Sparsity.dense(3,1)  #desired leg joint torques

    # Initialize the object
    def init(self):
        print('initializing Impedance controller of', self.leg_name, 'leg ..')

    # Evaluate numerically  tau = -J.T @ (
    #           f_des + Kp @ (x - x_des) + Kp @ (xdot - xdot_des))
    def eval(self, arg):
        q, qdot = arg[0], arg[1]
        x = self.ee_pos(q=q)['ee_pos']
        xdot = self.ee_vel(q=q, qdot=qdot)['ee_vel_linear']
        J_leg = self.J(q=q)['J'][:3, self.leg_joint_indices] 
        return -J_leg.T @ (
            arg[2] + self.Kp @ (x - arg[3]) + self.Kd @ (xdot - arg[4]) 
        )
       
class CentroidalPlusLegKinematicsCasadiModel:
    # constructor
    def __init__(self, conf, STOCHASTIC_OCP=False):
        # protected members
        self._robot_type = conf.rmodel.type
        self._foot_type = conf.rmodel.foot_type 
        self._ee_frame_names = conf.ee_frame_names
        self._rmodel = conf.rmodel
        self._gait = conf.gait
        self._gait_templates = conf.gait_templates
        self._q0 = conf.q0
        # self._urdf_path = conf.urdf_path
        self._n_x = conf.n_x  
        self._n_u = conf.n_u  
        self._n_w = conf.n_w    
        self._N_mpc = conf.N_mpc
        self._N = conf.N 
        self._m = conf.robot_mass                                    # total robot mass
        self._g = conf.gravity_constant                              # gravity constant
        self._dt = conf.dt                                           # discretization time
        self._state_cost_weights = conf.state_cost_weights           # state weight
        self._control_cost_weights = conf.control_cost_weights       # control weight 
        self._swing_foot_cost_weights = conf.swing_foot_cost_weights # swing foot weight    
        self._linear_friction_coefficient = conf.mu
        self._step_bound = conf.step_adjustment_bound
        self._Q = conf.Q
        self._R = conf.R 
        if conf.rmodel.foot_type == 'FLAT_FOOT':
            self._robot_foot_range = {'x':np.array([conf.lxp, conf.lxn]),
                                      'y':np.array([conf.lyp, conf.lyn])}
        self._STOCHASTIC_OCP = STOCHASTIC_OCP
        self._beta_u = conf.beta_u
        self._Cov_w = conf.cov_w  
        self._Cov_eta = conf.cov_white_noise
        # private methods
        self.__fill_contact_data(conf)
        if conf.rmodel.type == 'QUADRUPED':
            urdf = open('solo12.urdf', 'r').read()
            kindyn = cas_kin_dyn.CasadiKinDyn(urdf)
            self.__setup_casadi_model_quadruped_with_leg_kinematics()
            self.FL_impedance_controller = LegImpedanceController(
                'tau_FL', kindyn, conf.Kp, conf.Kd, 'FL_FOOT', range(6, 9)
                )
            self.FR_impedance_controller = LegImpedanceController(
                'tau_FR', kindyn, conf.Kp, conf.Kd, 'FR_FOOT', range(9, 12)
                )
            self.HL_impedance_controller = LegImpedanceController(
                'tau_HL', kindyn, conf.Kp, conf.Kd, 'HL_FOOT', range(12, 15)
                )
            self.HR_impedance_controller = LegImpedanceController(
                'tau_HR', kindyn, conf.Kp, conf.Kd, 'HR_FOOT', range(15, 18)
                )

    def __fill_contact_data(self, conf):
        contact_trajectory = create_contact_trajectory(conf)
        contacts_logic = []
        contacts_orientation = []
        contacts_position = []
        for time_idx in range(self._N):
            contacts_logic_k = []
            contacts_position_k = []
            contacts_orientation_k = [] 
            for contact in contact_trajectory:
                if contact_trajectory[contact][time_idx].ACTIVE:
                    contact_logic = 1
                    R = contact_trajectory[contact][time_idx].pose.rotation
                    p = contact_trajectory[contact][time_idx].pose.translation
                else:
                    contact_logic = 0
                    R = np.zeros((3,3))
                    p = np.zeros(3)
                contacts_logic_k.append(contact_logic)
                contacts_orientation_k.append(R)
                contacts_position_k.append(p)
            contacts_logic.append(contacts_logic_k)
            contacts_orientation.append(contacts_orientation_k)
            contacts_position.append(
                np.array(contacts_position_k).reshape(len(contact_trajectory)*3)
                )
        contacts_logic = np.array(contacts_logic)
        contacts_orientation = np.array(contacts_orientation)
        contacts_position = np.array(contacts_position)
        self._contact_trajectory = contact_trajectory
        self._contact_data = dict(contacts_logic=contacts_logic, 
                           contacts_orient=contacts_orientation, 
                            contacts_position=contacts_position)            
    
    def __setup_casadi_model_quadruped_with_leg_kinematics(self):
        m, g = self._m, self._g       
        # setup states & controls symbols
        # state
        x = vertcat(
            MX.sym('com_x'), MX.sym('com_y'), MX.sym('com_z'),                    #com
            MX.sym('lin_mom_x'), MX.sym('lin_mom_y'), MX.sym('lin_mom_z'),        #linear momentum  
            MX.sym('ang_mom_x'), MX.sym('ang_mom_y'), MX.sym('ang_mom_z'),        #angular momentum
                                                                             
            MX.sym('FL_ee_pos_x'), MX.sym('FL_ee_pos_y'), MX.sym('FL_ee_pos_z'),  #FL end-effecor positions
            MX.sym('FR_ee_pos_x'), MX.sym('FR_ee_pos_y'), MX.sym('FR_ee_pos_z'),  #FR end-effecor positions
            MX.sym('HL_ee_pos_x'), MX.sym('HL_ee_pos_y'), MX.sym('HL_ee_pos_z'),  #HL end-effecor positions
            MX.sym('HR_ee_pos_x'), MX.sym('HR_ee_pos_y'), MX.sym('HR_ee_pos_z'),  #HR end-effecor positions
            
            MX.sym('FL_ee_vel_x'), MX.sym('FL_ee_vel_y'), MX.sym('FL_ee_vel_z'),  #FL end-effecor positions
            MX.sym('FR_ee_vel_x'), MX.sym('FR_ee_vel_y'), MX.sym('FR_ee_vel_z'),  #FR end-effecor positions
            MX.sym('HL_ee_vel_x'), MX.sym('HL_ee_vel_y'), MX.sym('HL_ee_vel_z'),  #HL end-effecor positions
            MX.sym('HR_ee_vel_x'), MX.sym('HR_ee_vel_y'), MX.sym('HR_ee_vel_z'),  #HR end-effecor positions
            )             
        # controls
        u = vertcat(
            MX.sym('fx_FL'), MX.sym('fy_FL'), MX.sym('fz_FL'),                    #FL contact forces 
            MX.sym('fx_FR'), MX.sym('fy_FR'), MX.sym('fz_FR'),                    #FR contact forces
            MX.sym('fx_HL'), MX.sym('fy_HL'), MX.sym('fz_HL'),                    #HL contact forces
            MX.sym('fx_HR'), MX.sym('fy_HR'), MX.sym('fz_HR'),                    #HR contact forces
            
            MX.sym('FL_ee_acc_x'), MX.sym('FL_ee_acc_y'), MX.sym('FL_ee_acc_z'),  #FL end-effecor accelerations
            MX.sym('FR_ee_acc_x'), MX.sym('FR_ee_acc_y'), MX.sym('FR_ee_acc_z'),  #FR end-effecor accelerations
            MX.sym('HL_ee_acc_x'), MX.sym('HL_ee_acc_y'), MX.sym('HL_ee_acc_z'),  #HL end-effecor accelerations
            MX.sym('HR_ee_acc_x'), MX.sym('HR_ee_acc_y'), MX.sym('HR_ee_acc_z'),  #HR end-effecor accelerations
            )
        xdot = vertcat(
            MX.sym('com_xdot'), MX.sym('com_ydot'), MX.sym('com_zdot'),                  
            MX.sym('lin_mom_xdot'), MX.sym('lin_mom_ydot'), MX.sym('lin_mom_zdot'),      
            MX.sym('ang_mom_xdot'), MX.sym('ang_mom_ydot'), MX.sym('ang_mom_zdot'),     

            MX.sym('FL_ee_pos_xdot'), MX.sym('FL_ee_pos_ydot'), MX.sym('FL_ee_pos_zdot'),   
            MX.sym('FR_ee_pos_xdot'), MX.sym('FR_ee_pos_ydot'), MX.sym('FR_ee_pos_zdot'),  
            MX.sym('HL_ee_pos_xdot'), MX.sym('HL_ee_pos_ydot'), MX.sym('HL_ee_pos_zdot'),   
            MX.sym('HR_ee_pos_xdot'), MX.sym('HR_ee_pos_ydot'), MX.sym('HR_ee_pos_zdot'),

            MX.sym('FL_ee_vel_xdot'), MX.sym('FL_ee_vel_ydot'), MX.sym('FL_ee_vel_zdot'),   
            MX.sym('FR_ee_vel_xdot'), MX.sym('FR_ee_vel_ydot'), MX.sym('FR_ee_vel_zdot'),  
            MX.sym('HL_ee_vel_xdot'), MX.sym('HL_ee_vel_ydot'), MX.sym('HL_ee_vel_zdot'),   
            MX.sym('HR_ee_vel_xdot'), MX.sym('HR_ee_vel_ydot'), MX.sym('HR_ee_vel_zdot')
            )
        # setup parametric symbols
        # contact surface centroid position where the robot feet should stay within
        P_FR = MX.sym('p_FR',3)
        P_FL = MX.sym('p_FL',3)
        P_HR = MX.sym('p_HR',3)
        P_HL = MX.sym('p_HL',3)
        # contact surface orientation
        R_FR = vertcat(MX.sym('Rx_FR', 3),
                       MX.sym('Ry_FR', 3),
                       MX.sym('Rz_FR', 3))
        R_FL = vertcat(MX.sym('Rx_FL', 3),
                       MX.sym('Ry_FL', 3),
                       MX.sym('Rz_FL', 3))
        R_HR = vertcat(MX.sym('Rx_HR', 3),
                       MX.sym('Ry_HR', 3),
                       MX.sym('Rz_HR', 3))
        R_HL = vertcat(MX.sym('Rx_HL', 3),
                       MX.sym('Ry_HL', 3),
                       MX.sym('Rz_HL', 3))  
        # time-based gait specific parameters (determines which contact is active)
        CONTACT_ACTIVATION_FR = MX.sym('ACTIVE_FR')
        CONTACT_ACTIVATION_FL = MX.sym('ACTIVE_FL')
        CONTACT_ACTIVATION_HR = MX.sym('ACTIVE_HR')
        CONTACT_ACTIVATION_HL = MX.sym('ACTIVE_HL')
        contacts_logic = vertcat(
            CONTACT_ACTIVATION_FR, CONTACT_ACTIVATION_FL,
            CONTACT_ACTIVATION_HR, CONTACT_ACTIVATION_HL
            )
        # contact data    
        contacts_position = vertcat(P_FR, P_FL, P_HR, P_HL)
        contacts_norms = vertcat(R_FR, R_FL, R_HR, R_HL)
        contact_data = vertcat(
            contacts_logic, contacts_position, contacts_norms
            )
        # get end-effector positions
        contact_position_FL = x[9:12]
        contact_position_FR = x[12:15]
        contact_position_HL = x[15:18]
        contact_position_HR = x[18:21]
        # algebraic variables
        z = []   
        # centroidal dynamics 
        com = x[:3]
        contact_force_FR = u[0:3] 
        contact_force_FL = u[3:6] 
        contact_force_HR = u[6:9] 
        contact_force_HL = u[9:12] 
        lin_mom = CONTACT_ACTIVATION_FR*contact_force_FR +\
                  CONTACT_ACTIVATION_FL*contact_force_FL +\
                  CONTACT_ACTIVATION_HR*contact_force_HR +\
                  CONTACT_ACTIVATION_HL*contact_force_HL
        ang_mom = CONTACT_ACTIVATION_FR*cross((contact_position_FR-com),contact_force_FR)+\
                  CONTACT_ACTIVATION_FL*cross((contact_position_FL-com),contact_force_FL)+\
                  CONTACT_ACTIVATION_HR*cross((contact_position_HR-com),contact_force_HR)+\
                  CONTACT_ACTIVATION_HL*cross((contact_position_HL-com),contact_force_HL)       
        mg_vector = np.array([0., 0., m*g])
        # dynamics (centroidal + double integrator of end-effectors accelerations)
        f = vertcat(
                    (1./m)*x[3], (1./m)*x[4], (1./m)*x[5], 
                    lin_mom + mg_vector,
                    ang_mom,
                    x[21:],
                    u[12:]
                    )
        # construct             
        # friction pyramid constraints
        friction_pyramid_mat = construct_friction_pyramid_constraint_matrix(self)
        cone_constraints_fr = \
            CONTACT_ACTIVATION_FR*(friction_pyramid_mat @ (reshape(R_FR,(3,3)).T))
        cone_constraints_fl = \
            CONTACT_ACTIVATION_FL*(friction_pyramid_mat @ (reshape(R_FL,(3,3)).T))
        cone_constraints_hr = \
            CONTACT_ACTIVATION_HR*(friction_pyramid_mat @ (reshape(R_HR,(3,3)).T))
        cone_constraints_hl = \
            CONTACT_ACTIVATION_HL*(friction_pyramid_mat @ (reshape(R_HL,(3,3)).T)) 
        A_friction_pyramid = vertcat(
                horzcat(cone_constraints_fl, MX.zeros(5, 9)),
                horzcat(MX.zeros(5, 3), cone_constraints_fr, MX.zeros(5, 6)),
                horzcat(MX.zeros(5, 6), cone_constraints_hl, MX.zeros(5,3)),
                horzcat(MX.zeros(5, 9), cone_constraints_hr) 
                ) @ u[:12]
        lb_friction_pyramid = -1e15*np.ones(A_friction_pyramid.shape[0])
        ub_friction_pyramid = np.zeros(A_friction_pyramid.shape[0])
        # box constraints on lateral direction of the contact location
        A_contact_location_lateral = vertcat(
            CONTACT_ACTIVATION_FR*(contact_position_FR[:2]-P_FR[:2]),
            CONTACT_ACTIVATION_FL*(contact_position_FL[:2]-P_FL[:2]),
            CONTACT_ACTIVATION_HR*(contact_position_HR[:2]-P_HR[:2]),
            CONTACT_ACTIVATION_HL*(contact_position_HL[:2]-P_HL[:2])
            )
        ub_contact_location_lateral =  self._step_bound*np.ones(\
            A_contact_location_lateral.shape[0]
            )
        lb_contact_location_lateral = -ub_contact_location_lateral

        # zero height on the vertical direction of the contact location 
        A_contact_location_vertical = vertcat(
            CONTACT_ACTIVATION_FR*(contact_position_FR[2]-P_FR[2]),
            CONTACT_ACTIVATION_FL*(contact_position_FL[2]-P_FL[2]),
            CONTACT_ACTIVATION_HR*(contact_position_HR[2]-P_HR[2]),
            CONTACT_ACTIVATION_HL*(contact_position_HL[2]-P_HL[2])
            )
        lb_contact_location_vertical = np.zeros(A_contact_location_vertical.shape[0])
        ub_contact_location_vertical = lb_contact_location_vertical    
        # end-effector zero acceleration constraint when feet 
        # are in contact with the ground (i.e a_ee = 0)
        A_frame_acceleration = vertcat(
                                CONTACT_ACTIVATION_FL*u[12:15],
                                CONTACT_ACTIVATION_FR*u[15:18],
                                CONTACT_ACTIVATION_HL*u[18:21],
                                CONTACT_ACTIVATION_HR*u[21:]
                                )         
        lb_frame_acceleration = np.zeros(A_frame_acceleration.shape[0])
        ub_frame_acceleration = lb_frame_acceleration
        ## CasADi Model
        # define structs
        model = types.SimpleNamespace()
        contacts_params = types.SimpleNamespace()
        constraints = types.SimpleNamespace()
        ## fill casadi model
        model.model_name = "quadruped_centroidal_momentum_plus_leg_kinematics"
        model.f_impl_expr = xdot - f
        model.f_expl_expr = f
        model.x = x
        model.xdot = xdot
        model.u = u
        model.p = vertcat(contact_data)
        model.z = z 
        # fill contact parameters
        contacts_params.contacts_position = contacts_position 
        contacts_params.contacts_logic = contacts_logic
        contacts_params.contacts_norms = contacts_norms
        model.contacts_params = contacts_params
        # concatenate constraints    
        constraints.expr = vertcat(
            A_friction_pyramid,
            A_contact_location_vertical,
            A_contact_location_lateral,
            A_frame_acceleration,
 
            )
        constraints.lb = np.hstack([
            lb_friction_pyramid,
            lb_contact_location_vertical,
            lb_contact_location_lateral,
            lb_frame_acceleration,
        ])
        constraints.ub = np.hstack([
            ub_friction_pyramid,
            ub_contact_location_vertical,
            ub_contact_location_lateral,
            ub_frame_acceleration,
        ])
        model.constraints = constraints
        self.casadi_model = model

if __name__ == "__main__":
    from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
    import conf_solo12_trot_step_adjustment as conf
    model = CentroidalPlusLegKinematicsCasadiModel(conf)