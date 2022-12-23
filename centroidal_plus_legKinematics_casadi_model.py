from utils import construct_friction_pyramid_constraint_matrix, quaternion_plus_casadi_fun
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from contact_plan import create_contact_trajectory
from casadi import *
import numpy as np

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
        self._Q = conf.Q
        self._R = conf.R
        self._linear_friction_coefficient = conf.mu
        self._step_bound = conf.step_adjustment_bound
        self._Q = conf.Q
        self._R = conf.R 
        if conf.rmodel.foot_type == 'FLAT_FOOT':
            self._robot_foot_range = {'x':np.array([conf.lxp, conf.lxn]),
                                      'y':np.array([conf.lyp, conf.lyn])}
        self._STOCHASTIC_OCP = STOCHASTIC_OCP
        self._beta_u = conf.beta_u
        self._Cov_w = conf.cov_w_dt  
        self._Cov_eta = conf.cov_white_noise

        # private methods
        self.__fill_contact_data(conf)
        if conf.rmodel.type == 'QUADRUPED':
            self.__setup_casadi_model_quadruped_with_leg_kinematics()    

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
    
    def __create_leg_kinematics_and_jacobians_casadi_funs(self):
        urdf = open('solo12.urdf', 'r').read()
        kindyn = cas_kin_dyn.CasadiKinDyn(urdf)
        LOCAL_WORLD_ALIGNED = cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
        self.hg = Function.deserialize(kindyn.computeCentroidalDynamics())
        self.com = Function.deserialize(kindyn.centerOfMass())
        if self._robot_type == 'QUADRUPED':
            # get frame names
            FL_frame_name = self._ee_frame_names[0]
            FR_frame_name = self._ee_frame_names[1]
            HL_frame_name = self._ee_frame_names[2]
            HR_frame_name = self._ee_frame_names[3]
            # end-effectors kinematics casadi functions
            self.fk_FL = Function.deserialize(kindyn.fk(FL_frame_name))
            self.fk_FR = Function.deserialize(kindyn.fk(FR_frame_name))
            self.fk_HL = Function.deserialize(kindyn.fk(HL_frame_name))
            self.fk_HR = Function.deserialize(kindyn.fk(HR_frame_name))
            # end-effectors jacobians
            self.J_FL = Function.deserialize(
                kindyn.jacobian(FL_frame_name, LOCAL_WORLD_ALIGNED)
                )
            self.J_FR = Function.deserialize(
                kindyn.jacobian(FR_frame_name, LOCAL_WORLD_ALIGNED)
                )
            self.J_HL = Function.deserialize(
                kindyn.jacobian(HL_frame_name, LOCAL_WORLD_ALIGNED)
                )
            self.J_HR = Function.deserialize(
                kindyn.jacobian(HR_frame_name, LOCAL_WORLD_ALIGNED)
                )
            # end-effectors frame velocities
            self.FL_frame_vel = Function.deserialize(
                kindyn.frameVelocity(FL_frame_name, LOCAL_WORLD_ALIGNED)
                )
            self.FR_frame_vel = Function.deserialize(
                kindyn.frameVelocity(FR_frame_name, LOCAL_WORLD_ALIGNED)
                )
            self.HL_frame_vel = Function.deserialize(
                kindyn.frameVelocity(HL_frame_name, LOCAL_WORLD_ALIGNED)
                )
            self.HR_frame_vel = Function.deserialize(
                kindyn.frameVelocity(HR_frame_name, LOCAL_WORLD_ALIGNED)
                )          

    def __setup_casadi_model_quadruped_with_leg_kinematics(self):
        m, g = self._m, self._g       
        # setup states & controls symbols
        # state
        lamda = MX.sym('lambda', 3, 1) 
        omega = MX.sym('omega', 3, 1)
        x = vertcat(
            MX.sym('com_x'), MX.sym('com_y'), MX.sym('com_z'),                   #com
            MX.sym('lin_mom_x'), MX.sym('lin_mom_y'), MX.sym('lin_mom_z'),       #lin. momentum  
            MX.sym('ang_mom_x'), MX.sym('ang_mom_y'), MX.sym('ang_mom_z'),       #ang. momentum
            
            MX.sym('base_pos_x'), MX.sym('base_pos_y'), MX.sym('base_pos_z'),    #base pos.
            lamda,                                                               #dq base

            MX.sym('FL_HAA_pos'), MX.sym('FL_HFE_pos'), MX.sym('FL_KFE_pos'),    #FL joint pos.
            MX.sym('FR_HAA_pos'), MX.sym('FR_HFE_pos'), MX.sym('FR_KFE_pos'),    #FR joint pos.
            MX.sym('HL_HAA_pos'), MX.sym('HL_HFE_pos'), MX.sym('HL_KFE_pos'),    #HL joint pos.
            MX.sym('HR_HAA_pos'), MX.sym('HR_HFE_pos'), MX.sym('HR_KFE_pos')     #HR joint pos.
                    )             
        # controls
        u = vertcat(
            MX.sym('fx_FL'), MX.sym('fy_FL'), MX.sym('fz_FL'),                   #FL contact forces 
            MX.sym('fx_FR'), MX.sym('fy_FR'), MX.sym('fz_FR'),                   #FR contact forces
            MX.sym('fx_HL'), MX.sym('fy_HL'), MX.sym('fz_HL'),                   #HL contact forces
            MX.sym('fx_HR'), MX.sym('fy_HR'), MX.sym('fz_HR'),                   #HR contact forces
            
            MX.sym('base_vel_x'), MX.sym('base_vel_y'), MX.sym('base_vel_z'),    #base lin. vel.
            omega,                                                               #base ang. vel.
            
            MX.sym('FL_HAA_vel'), MX.sym('FL_HFE_vel'), MX.sym('FL_KFE_vel'),    #FL joint vel.
            MX.sym('FR_HAA_vel'), MX.sym('FR_HFE_vel'), MX.sym('FR_KFE_vel'),    #FR joint vel.
            MX.sym('HL_HAA_vel'), MX.sym('HL_HFE_vel'), MX.sym('HL_KFE_vel'),    #HL joint vel.
            MX.sym('HR_HAA_vel'), MX.sym('HR_HFE_vel'), MX.sym('HR_KFE_vel'),    #HR joint vel.
                    )
        xdot = vertcat(
            MX.sym('dcom_x')    , MX.sym('dcom_y')    , MX.sym('dcom_z'),        #dCoM linear vel.
            MX.sym('dlin_mom_x'), MX.sym('dlin_mom_y'), MX.sym('dlin_mom_z'),    #dCoM linear acc.
            MX.sym('dang_mom_x'), MX.sym('dang_mom_y'), MX.sym('dang_mom_z'),    #dCoM ang. acc.

            MX.sym('dbase_pos_x'), MX.sym('dbase_pos_y'), MX.sym('dbase_pos_z'), #dbase lin. acc.
            MX.sym('dlambda_x')  , MX.sym('dlambda_y')  , MX.sym('dlambda_z'),   #ddq base

            MX.sym('dFL_HAA_pos'), MX.sym('dFL_HFE_pos'), MX.sym('dFL_KFE_pos'), #dFL joint vel.
            MX.sym('dFR_HAA_pos'), MX.sym('dFR_HFE_pos'), MX.sym('dFR_KFE_pos'), #dFR joint vel.
            MX.sym('dHL_HAA_pos'), MX.sym('dHL_HFE_pos'), MX.sym('dHL_KFE_pos'), #dHL joint vel.
            MX.sym('dHR_HAA_pos'), MX.sym('dHR_HFE_pos'), MX.sym('dHR_KFE_pos')  #dHR joint vel.
                       )
        # setup parametric symbols
        # contact surface centroid position 
        # where the robot feet should stay within
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
        qref_base = MX.sym('qref_base', 4, 1)
        # get box plus and box minus functions
        q_plus = quaternion_plus_casadi_fun() 
        # q_minus = quaternion_minus_casadi_fun()
        qbase_relative = q_plus(qref_base, lamda)
        # this is now the generalized position coordinates vector 
        q_bar = vertcat(
            x[9:12],          # base position
            qbase_relative,   # relative base orientation
            x[15:]            # joint positions
            )
        # get end-effectors forward kinematics
        self.__create_leg_kinematics_and_jacobians_casadi_funs()
        contact_position_FL = self.fk_FL(q=q_bar)['ee_pos']
        contact_position_FR = self.fk_FR(q=q_bar)['ee_pos']
        contact_position_HL = self.fk_HL(q=q_bar)['ee_pos']
        contact_position_HR = self.fk_HR(q=q_bar)['ee_pos']
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
        # qref [+] lambda_next =  (qref_curr [+] lambda_curr) [+] omega
        # which is equivalent to 
        # lambda_next = lambda_curr + omega.dt
        f = vertcat(
                    (1./m)*x[3], (1./m)*x[4], (1./m)*x[5], 
                    lin_mom + mg_vector,
                    ang_mom,      
                    u[12:15],     # base linear vel.
                    omega,        # base angular vel.
                    u[18:30]      # joint vel.
                    )        
        # full-kinematics com constraint
        A_com = self.com(q=q_bar)['com'] - com 
        lb_com = np.zeros(A_com.shape[0])
        ub_com = lb_com
        # full-kinematics linear momentum constraint 
        qdot = u[12:30]
        nv = qdot.shape[0] 
        A_dh_linmom = self.hg(q=q_bar, v=qdot, a=MX.zeros(nv))['dh_lin'] - x[3:6]
        lb_dh_linmom = np.zeros(A_dh_linmom.shape[0])
        ub_dh_linmom = lb_dh_linmom
        # full-kinematics angular momentum constraint 
        A_dh_angmom = self.hg(q=q_bar, v=qdot, a=MX.zeros(nv))['dh_ang'] - x[6:9]
        lb_dh_angmom = np.zeros(A_dh_angmom.shape[0])
        ub_dh_angmom = lb_dh_angmom
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
                horzcat(MX.zeros(5, 6), cone_constraints_hl, MX.zeros(5, 3)),
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
        # end-effector frame velocity constraint when feet 
        # are in contact with the ground (i.e J(q).qdot = 0)
        A_frame_velocity = vertcat(
                                CONTACT_ACTIVATION_FR*(
                                    self.FR_frame_vel(q=q_bar, qdot=qdot)['ee_vel_linear']
                                    ),
                                CONTACT_ACTIVATION_FL*(
                                    self.FL_frame_vel(q=q_bar, qdot=qdot)['ee_vel_linear']
                                    ),
                                CONTACT_ACTIVATION_HR*(
                                    self.HR_frame_vel(q=q_bar, qdot=qdot)['ee_vel_linear']
                                    ),
                                CONTACT_ACTIVATION_HL*(
                                    self.HL_frame_vel(q=q_bar, qdot=qdot)['ee_vel_linear']
                                    )
                                )         
        lb_frame_velocity = np.zeros(A_frame_velocity.shape[0])
        ub_frame_velocity = lb_frame_velocity
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
        model.p = vertcat(contact_data, qref_base)
        model.z = z 
        # fill contact parameters
        contacts_params.contacts_position = contacts_position 
        contacts_params.contacts_logic = contacts_logic
        contacts_params.contacts_norms = contacts_norms
        model.contacts_params = contacts_params
        # concatenate constraints 
        constraints.expr = vertcat(
            # A_friction_pyramid, 
            A_frame_velocity,
            # A_contact_location_lateral,
            # A_contact_location_vertical,
            # A_dh_linmom,
            A_dh_angmom,
            A_com
            )
        constraints.lb = np.hstack([
            # lb_friction_pyramid,
            lb_frame_velocity,
            # lb_contact_location_lateral,
            # lb_contact_location_vertical,
            # lb_dh_linmom,
            lb_dh_angmom,
            lb_com
        ])
        constraints.ub = np.hstack([
            # ub_friction_pyramid,
            ub_frame_velocity,
            # ub_contact_location_lateral,
            # ub_contact_location_vertical,
            # ub_dh_linmom,
            ub_dh_angmom,
            ub_com
        ])
        model.constraints = constraints
        model.q_bar = q_bar
        model.q_plus = quaternion_plus_casadi_fun() 
        model.fk_q_bar_pos = vertcat(
            self.fk_FL(q=q_bar)['ee_pos'], 
            self.fk_FR(q=q_bar)['ee_pos'],
            self.fk_HL(q=q_bar)['ee_pos'],
            self.fk_HR(q=q_bar)['ee_pos']
        )
        # model.A_friction_pyramid = friction_pyramid_mat
        model.ee_frame_vel = A_frame_velocity
        # dynamics jacobians
        discrete_dynamics = x + f*self._dt
        model.Jx_fun = Function("A",
                    [x, u, model.p], [jacobian(discrete_dynamics, x)],
                    ['x', 'u', 'p'], ['J_x']
                    )
        model.Ju_fun = Function("B",
                    [x, u, model.p], [jacobian(discrete_dynamics, u)],
                    ['x', 'u', 'p'], ['J_u']
                    )
        self.casadi_model = model
    
  
if __name__ == "__main__":
    from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
    import conf_solo12_trot_step_adjustment as conf
    model = CentroidalPlusLegKinematicsCasadiModel(conf)