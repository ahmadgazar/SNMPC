from utils import construct_friction_pyramid_constraint_matrix
from contact_plan import create_contact_trajectory 
from warnings import warn
from casadi import *
import numpy as np

class CentroidalModelCasadi:
    # constructor
    def __init__(self, conf, STOCHASTIC_OCP=False):
        # protected members
        self._DYNAMICS_FIRST = conf.DYNAMICS_FIRST
        self._robot = conf.robot_name 
        self._n_x = conf.n_x  
        self._n_u_per_contact = conf.n_u_per_contact
        self._n_u = conf.n_u  
        self._n_w = conf.n_w   
        self._n_t = conf.n_t
        self._N_mpc = conf.N_mpc
        self._N = conf.N 
        self._total_nb_optimizers = self._n_x*(self._N+1) + self._n_u*self._N + \
                                    self._n_t*(self._N+1) + self._n_t*self._N
        self._max_leg_length = conf.max_leg_length 
        self._m = conf.robot_mass        # total robot mass
        self._g = conf.gravity_constant  # gravity constant
        self._dt = conf.dt               # discretization time
        self._state_cost_weights = conf.state_cost_weights     # state weight
        self._control_cost_weights = conf.control_cost_weights # control weight    
        self._linear_friction_coefficient = conf.mu
        self._Q = conf.Q
        self._R = conf.R 
        if self._robot == 'TALOS':
            self._robot_foot_range = {'x':np.array([conf.lxp, conf.lxn]),
                                      'y':np.array([conf.lyp, conf.lyn])}
        self._STOCHASTIC_OCP = STOCHASTIC_OCP
        self._beta_u = conf.beta_u
        self._Cov_w = conf.cov_w  
        self._Cov_eta = conf.cov_white_noise                                                           
        # private methods
        self.__set_init_and_final_states(conf)
        self.__fill_contact_data(conf)
        self.__fill_initial_trajectory()
        self.__setup_casadi_dynamics_model()
    
    # private methods
    def __set_init_and_final_states(self, conf):
        self._DYNAMICS_FIRST = conf.DYNAMICS_FIRST
        if conf.DYNAMICS_FIRST:
            self._com_z = conf.com_z
            self._x_init = conf.x_init
            self._x_final = conf.x_final   
        else:
            IK_to_Dyn = np.load('wholeBody_to_centroidal_traj.npz')['X']                             
            self._x_init = IK_to_Dyn[0]
            self._x_final = IK_to_Dyn[-1]
   
    def __fill_contact_data(self, conf):
        contact_trajectory = create_contact_trajectory(conf)
        contacts_logic = []
        contacts_orientation = []
        contacts_position = []
        #TODO Try to figure out how to do nested loops in JAX
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
            contacts_position.append(np.array(contacts_position_k).reshape(len(contact_trajectory)*3))
        contacts_logic = np.array(contacts_logic)
        contacts_orientation = np.array(contacts_orientation)
        contacts_position = np.array(contacts_position)
        self._contact_trajectory = contact_trajectory
        self._contact_data = dict(contacts_logic=contacts_logic, 
                           contacts_orient=contacts_orientation, 
                            contacts_position=contacts_position)            
    
    def __fill_initial_trajectory(self):
        N = self._N
        init_trajectories = {'state':np.zeros((self._n_x, N+1)), 'control':np.zeros((self._n_u, N))}
        if self._DYNAMICS_FIRST:
            warn('currently kinematics first is only supported!')    
            pass 
        else:   
            # warm-start SCP states using whole-body DDP
            init_trajectories['state'] = np.array(np.load('wholeBody_to_centroidal_traj.npz')['X'])
            # warm-start stochastic SCP controls 
            robot_weight =  -self._m*self._g
            contacts_logic_total = self._contact_data['contacts_logic']
            for time_idx, contacts_logic_k in enumerate(contacts_logic_total):
                robot_weight_per_contact = robot_weight/np.sum(contacts_logic_k)
                for contact_idx, contact_logic in enumerate(contacts_logic_k):
                    if contact_logic:
                        fx_idx = contact_idx*3
                        init_trajectories['control'][fx_idx:fx_idx+3, time_idx] = \
                                   np.array([1e-3, 1e-3, robot_weight_per_contact])
        self._init_trajectories = init_trajectories

    def __setup_casadi_dynamics_model(self):
        m, g = self._m, self._g       
        # setup states & controls symbols
        x = vertcat(MX.sym('com_x'), MX.sym('com_y'), MX.sym('com_z'),\
                    MX.sym('lin_mom_x'), MX.sym('lin_mom_y'), MX.sym('lin_mom_z'),\
                    MX.sym('ang_mom_x'), MX.sym('ang_mom_y'), MX.sym('ang_mom_z'))
     
        u = vertcat(MX.sym('fx_FR'), MX.sym('fy_FR'), MX.sym('fz_FR'),\
                    MX.sym('fx_FL'), MX.sym('fy_FL'), MX.sym('fz_FL'),\
                    MX.sym('fx_HR'), MX.sym('fy_HR'), MX.sym('fz_HR'),\
                    MX.sym('fx_HL'), MX.sym('fy_HL'), MX.sym('fz_HL,'))
        # xdot
        xdot = vertcat(MX.sym('com_xdot'), MX.sym('com_ydot'), MX.sym('com_zdot'),\
                       MX.sym('lin_mom_xdot'), MX.sym('lin_mom_ydot'), MX.sym('lin_mom_zdot'),\
                       MX.sym('ang_mom_xdot'), MX.sym('ang_mom_ydot'), MX.sym('ang_mom_zdot'))
        # setup parametric symbols
        contact_position_FR = MX.sym('p_FR',3)
        contact_position_FL = MX.sym('p_FL',3)
        contact_position_HR = MX.sym('p_HR',3)
        contact_position_HL = MX.sym('p_HL',3)
        CONTACT_ACTIVATION_FR = MX.sym('ACTIVE_FR')
        CONTACT_ACTIVATION_FL = MX.sym('ACTIVE_FL')
        CONTACT_ACTIVATION_HR = MX.sym('ACTIVE_HR')
        CONTACT_ACTIVATION_HL = MX.sym('ACTIVE_HL')
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
        contacts_logic = vertcat(CONTACT_ACTIVATION_FR, CONTACT_ACTIVATION_FL,
                                 CONTACT_ACTIVATION_HR, CONTACT_ACTIVATION_HL)               
        contacts_position = vertcat(contact_position_FR, contact_position_FL,
                                    contact_position_HR, contact_position_HL)
        contacts_norms = vertcat(R_FR, R_FL, R_HR, R_HL)
        contact_data = vertcat(contacts_logic, contacts_position, contacts_norms)
        # algebraic variables
        z = vertcat([])
        # dynamics
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
        f = vertcat((1./m)*x[3], (1./m)*x[4], (1./m)*x[5], lin_mom + mg_vector , ang_mom)
        # friction pyramid constraints
        friction_pyramid_mat = construct_friction_pyramid_constraint_matrix(self)
        cone_constraints_fr = CONTACT_ACTIVATION_FR*(friction_pyramid_mat @ (reshape(R_FR,(3,3)).T))
        cone_constraints_fl = CONTACT_ACTIVATION_FL*(friction_pyramid_mat @ (reshape(R_FL,(3,3)).T))
        cone_constraints_hr = CONTACT_ACTIVATION_HR*(friction_pyramid_mat @ (reshape(R_HR,(3,3)).T))
        cone_constraints_hl = CONTACT_ACTIVATION_HL*(friction_pyramid_mat @ (reshape(R_HL,(3,3)).T)) 
        A = vertcat(
                horzcat(cone_constraints_fr, MX.zeros(5, 9)),
                horzcat(MX.zeros(5, 3), cone_constraints_fl, MX.zeros(5, 6)),
                horzcat(MX.zeros(5, 6), cone_constraints_hr, MX.zeros(5,3)),
                horzcat(MX.zeros(5, 9), cone_constraints_hl) 
                )
        lb = -1e15*np.ones(contacts_logic.shape[0]*friction_pyramid_mat.shape[0])
        ub = np.zeros(contacts_logic.shape[0]*friction_pyramid_mat.shape[0])
        ## CasADi Model
        # define structs
        model = types.SimpleNamespace()
        contacts_params = types.SimpleNamespace()
        friction_pyramid_constraints = types.SimpleNamespace()
        ## fill casadi model
        model.model_name = "centroidal_momentum_model"
        model.f_impl_expr = xdot - f
        model.f_expl_expr = f
        model.x = x
        model.xdot = xdot
        model.u = u
        model.p = contact_data
        model.z = z 
        # fill parameters
        contacts_params.contacts_position = contacts_position 
        contacts_params.contacts_logic = contacts_logic
        contacts_params.contacts_norms = contacts_norms
        model.contacts_params = contacts_params    
        # fill constraints
        friction_pyramid_constraints.expr = A @ u
        friction_pyramid_constraints.lb = lb 
        friction_pyramid_constraints.ub = ub
        model.friction_pyramid_constraints = friction_pyramid_constraints
        self.casadi_model = model
        self.casadi_model_fcn = Function('f', [x, u, contact_data], [x + self._dt*f])
        self.casadi_model_const = Function('bla', [u, contact_data], [friction_pyramid_constraints.expr, lb, ub])

#     # def test_casadi_dynamics(self)
# if __name__ == "__main__":
#     import conf_solo12_trot as conf
#     from centroidal_model import Centroidal_model
#     from constraints import construct_friction_pyramid_constraints
#     import matplotlib.pylab as plt
#     model_casadi = CentroidalModelCasadi(conf)
#     model_jax = Centroidal_model(conf)
#     contacts_logic_N = model_casadi._contact_data['contacts_logic']
#     contacts_position_N = model_casadi._contact_data['contacts_position'] 
#     contacts_norms_N = model_casadi._contact_data['contacts_orient']
#     N = model_jax._N
#     constraint_jax = construct_friction_pyramid_constraints(model_jax)
#     for t_idx in range(N):
#         contacts_logic_k = contacts_logic_N[t_idx]
#         contacts_position_k = contacts_position_N[t_idx]
#         contacts_norms_k = contacts_norms_N[t_idx].flatten()
#         contact_params_k = np.concatenate([contacts_logic_k, contacts_position_k, contacts_norms_k])
#         x = model_casadi._init_trajectories['state'][t_idx, :]
#         u = model_casadi._init_trajectories['control'][:, t_idx]
#         # casadi stuff
#         f_casadi = model_casadi.casadi_model_fcn(x, u, contact_params_k)
#         A_casadi, lb_casadi, ub_casadi = model_casadi.casadi_model_const(u, contact_params_k)
#         # jax stuff
#         f_jax = model_jax.integrate_model_one_step(x, u, contacts_position_k, 
#                                        contacts_logic_k, contacts_norms_N[0])
#         nx, nu = model_jax._n_x, model_jax._n_u
#         A_FR = constraint_jax.mat[t_idx*5:(t_idx*5)+5, (N+1)*nx + nu*t_idx:((N+1)*nx + nu*t_idx)+nu]
#         A_FL = constraint_jax.mat[N*5+t_idx*5:(N*5+t_idx*5)+5, (N+1)*nx + nu*t_idx:((N+1)*nx + nu*t_idx)+nu]
#         A_HR = constraint_jax.mat[2*N*5 +t_idx*5:(2*N*5+t_idx*5)+5, (N+1)*nx + nu*t_idx:((N+1)*nx + nu*t_idx)+nu]
#         A_HL = constraint_jax.mat[3*N*5+ t_idx*5:(3*N*5+t_idx*5)+5, (N+1)*nx + nu*t_idx:((N+1)*nx + nu*t_idx)+nu]
#         A_jax = np.concatenate([A_FR, A_FL, A_HR, A_HL], axis=0)
#         for i in range(f_jax.shape[0]):
#             if(np.allclose(f_jax[i], np.array(f_casadi)[i])):
#                 continue
#             else:
#                 print('ffffffffuuuuack dynamically wrong!')
#         # print('A_jax = ', A_jax)
#         # print('A_casadi = ', A_casadi)             
#         if np.allclose(np.array(A_casadi), A_jax):
#             continue
#         else:
#             print('ffffffffuuuuack constraints are wrong!')

