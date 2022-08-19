from utils import construct_friction_pyramid_constraint_matrix
from contact_plan import create_contact_trajectory 
from casadi import *
import numpy as np

class CentroidalModelCasadi:
    # constructor
    def __init__(self, conf, STOCHASTIC_OCP=False):
        # protected members
        # self._DYNAMICS_FIRST = conf.DYNAMICS_FIRST
        # self._robot = conf.robot_name
        self._robot_type = conf.rmodel.type
        self._foot_type = conf.rmodel.foot_type 
        self._n_x = conf.n_x  
        self._n_u = conf.n_u  
        self._n_w = conf.n_w   
        self._N_mpc = conf.N_mpc
        self._N = conf.N 
        # self._max_leg_length = conf.max_leg_length 
        self._m = conf.robot_mass        # total robot mass
        self._g = conf.gravity_constant  # gravity constant
        self._dt = conf.dt               # discretization time
        self._state_cost_weights = conf.state_cost_weights     # state weight
        self._control_cost_weights = conf.control_cost_weights # control weight    
        self._linear_friction_coefficient = conf.mu
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
            self.__setup_casadi_model_quadruped()
        if conf.rmodel.type == 'HUMANOID' and conf.rmodel.foot_type == 'FLAT_FOOT':
            self.__setup_casadi_model_flat_foot_humanoid()

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
    
    def __setup_casadi_model_quadruped(self):
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
        model.model_name = "centroidal_momentum_quadruped"
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

    def __setup_casadi_model_flat_foot_humanoid(self):
        m, g = self._m, self._g       
        # setup states & controls symbols
        x = vertcat(MX.sym('com_x'), MX.sym('com_y'), MX.sym('com_z'),\
                    MX.sym('lin_mom_x'), MX.sym('lin_mom_y'), MX.sym('lin_mom_z'),\
                    MX.sym('ang_mom_x'), MX.sym('ang_mom_y'), MX.sym('ang_mom_z'))
        
        u = vertcat(MX.sym('cop_x_FR'), MX.sym('cop_y_FR'), MX.sym('fx_FR'), 
                    MX.sym('fy_FR')   , MX.sym('fz_FR')   , MX.sym('tau_z_FR'),\
                    MX.sym('cop_x_FL'), MX.sym('cop_y_FL'), MX.sym('fx_FL'), 
                    MX.sym('fy_FL')   , MX.sym('fz_FL,')  , MX.sym('tau_z_FL'))
        # xdot
        xdot = vertcat(MX.sym('com_xdot'), MX.sym('com_ydot')    , MX.sym('com_zdot'),\
                   MX.sym('lin_mom_xdot'), MX.sym('lin_mom_ydot'), MX.sym('lin_mom_zdot'),\
                   MX.sym('ang_mom_xdot'), MX.sym('ang_mom_ydot'), MX.sym('ang_mom_zdot'))
        # setup parametric symbols
        contact_position_FR = MX.sym('p_FR',3)
        contact_position_FL = MX.sym('p_FL',3)
        CONTACT_ACTIVATION_FR = MX.sym('ACTIVE_FR')
        CONTACT_ACTIVATION_FL = MX.sym('ACTIVE_FL')
        R_FR = vertcat(MX.sym('Rx_FR', 3),
                       MX.sym('Ry_FR', 3),
                       MX.sym('Rz_FR', 3))
        R_FL = vertcat(MX.sym('Rx_FL', 3),
                       MX.sym('Ry_FL', 3),
                       MX.sym('Rz_FL', 3))
        contacts_logic = vertcat(CONTACT_ACTIVATION_FR, CONTACT_ACTIVATION_FL)               
        contacts_position = vertcat(contact_position_FR, contact_position_FL)
        contacts_norms = vertcat(R_FR, R_FL)
        contact_data = vertcat(contacts_logic, contacts_position, contacts_norms)
        # algebraic variables
        z = vertcat([])
        # dynamics
        com = x[:3]
        cop_FR = u[0:2] 
        contact_force_FR = u[2:5]
        tau_z_FR = u[5] 
        cop_FL = u[6:8] 
        contact_force_FL = u[8:11]
        tau_z_FL = u[11]
        lin_mom = CONTACT_ACTIVATION_FR*contact_force_FR +\
                  CONTACT_ACTIVATION_FL*contact_force_FL            
        ang_mom_rf = CONTACT_ACTIVATION_FR*(
                       cross((contact_position_FR-com), contact_force_FR) + \
                       cross((reshape(R_FR[:6], (3, 2)) @ cop_FR), contact_force_FR) + R_FR[6:9] @ tau_z_FR
                       ) 
        ang_mom_lf = CONTACT_ACTIVATION_FL*(
                       cross((contact_position_FL-com), contact_force_FL) + \
                       cross((reshape(R_FL[:6], (3, 2)) @ cop_FL), contact_force_FL) + R_FL[6:9] @ tau_z_FL
                       )
        ang_mom = ang_mom_rf + ang_mom_lf
        mg_vector = np.array([0., 0., m*g])
        f = vertcat((1./m)*x[3], (1./m)*x[4], (1./m)*x[5], lin_mom + mg_vector , ang_mom)
        # friction pyramid constraints
        friction_pyramid_mat = construct_friction_pyramid_constraint_matrix(self)
        cone_constraints_fr = CONTACT_ACTIVATION_FR*(friction_pyramid_mat @ (reshape(R_FR,(3,3)).T))
        cone_constraints_fl = CONTACT_ACTIVATION_FL*(friction_pyramid_mat @ (reshape(R_FL,(3,3)).T))
        A_fc = vertcat(
                horzcat(MX.zeros(5, 2), cone_constraints_fr, MX.zeros(5, 7)),
                horzcat(MX.zeros(5, 8), cone_constraints_fl, MX.zeros(5, 1))
                )
        lb_fc = -1e15*np.ones(contacts_logic.shape[0]*friction_pyramid_mat.shape[0])
        ub_fc = np.zeros(contacts_logic.shape[0]*friction_pyramid_mat.shape[0])
        # CoP box bounds
        cop_constraint_mat = np.array([[1, 0], [0, 1]])
        A_cop = vertcat(
                horzcat(CONTACT_ACTIVATION_FR*cop_constraint_mat, MX.zeros(2, 10)),
                horzcat(MX.zeros(2, 6), CONTACT_ACTIVATION_FL*cop_constraint_mat, MX.zeros(2, 4))
                )
        lb_cop = np.array([self._robot_foot_range['x'][1], self._robot_foot_range['y'][1], 
                           self._robot_foot_range['x'][1], self._robot_foot_range['y'][1]])
        ub_cop = np.array([self._robot_foot_range['x'][0], self._robot_foot_range['y'][0],
                           self._robot_foot_range['x'][0], self._robot_foot_range['y'][0]])        
        ## CasADi Model
        # define structs
        model = types.SimpleNamespace()
        contacts_params = types.SimpleNamespace()
        cop_constraints = types.SimpleNamespace()
        friction_pyramid_constraints = types.SimpleNamespace()
        ## fill casadi model
        model.model_name = "centroidal_momentum_flat_foot_humanoid"
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
        friction_pyramid_constraints.expr = A_fc @ u
        friction_pyramid_constraints.lb = lb_fc 
        friction_pyramid_constraints.ub = ub_fc
        cop_constraints.expr = A_cop @ u
        cop_constraints.lb = lb_cop
        cop_constraints.ub = ub_cop
        model.friction_pyramid_constraints = friction_pyramid_constraints
        model.cop_constraints = cop_constraints
        self.casadi_model = model

if __name__ == "__main__":
    from wholebody_croccodyl_model import WholeBodyModel
    from wholebody_croccodyl_solver import WholeBodyDDPSolver
    from centroidal_acados_solver import CentroidalSolverAcados
    import matplotlib.pylab as plt
    import conf_talos_full_with_locked_joints as conf
    wbd_model = WholeBodyModel(conf)
    ddp_planner = WholeBodyDDPSolver(wbd_model)
    ddp_planner.solve()
    sol_hg = np.asarray(ddp_planner.get_solution_trajectories()['centroidal'])
    centroidal_model = CentroidalModelCasadi(conf)
    centroidal_planner = CentroidalSolverAcados(centroidal_model, sol_hg)
    X, U = centroidal_planner.solve()
    fig, (comx, comy, comz, lx, ly, lz, kappax, kappay, kappaz) = plt.subplots(9, 1, sharex=True)
    time = np.arange(0, np.round((X.shape[0])*conf.dt, 2), conf.dt)
    comx.plot(time, X[:, 0])
    comx.plot(time, sol_hg[:, 0])
    comy.plot(time, X[:, 1])
    comy.plot(time, sol_hg[:, 1])
    comz.plot(time, X[:, 2])
    comz.plot(time, sol_hg[:, 2])
    lx.plot(time, X[:, 3])
    lx.plot(time, sol_hg[:, 3])
    ly.plot(time, X[:, 4])
    ly.plot(time, sol_hg[:, 4])
    lz.plot(time, X[:, 5])
    lz.plot(time, sol_hg[:, 5])
    kappax.plot(time, X[:, 6])
    kappax.plot(time, sol_hg[:, 6])
    kappay.plot(time, X[:, 7])
    kappay.plot(time, sol_hg[:, 7])
    kappaz.plot(time, X[:, 8])
    kappaz.plot(time, sol_hg[:, 8])
    plt.show()


    


