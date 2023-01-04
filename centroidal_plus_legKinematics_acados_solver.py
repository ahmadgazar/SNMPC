from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from scipy.stats import norm
import scipy.linalg as la
from casadi import *
import numpy as np
import pinocchio
import time 

class CentroidalPlusLegKinematicsAcadosSolver:
    # constructor
    def __init__(self, model, x_ref, u_ref, MPC=False, WARM_START=True):
        self.RECEEDING_HORIZON = MPC
        # timing
        self.N_mpc = model._N_mpc
        self.N_traj = model._N
        self.dt = model._dt
        # tracking reference 
        self.x_init = x_ref
        self.u_ref = u_ref
        self.contact_data = model._contact_data
        # cost function weights
        self.state_cost_weights = model._state_cost_weights
        self.control_cost_weights = model._control_cost_weights
        self.swing_cost_weights = model._swing_foot_cost_weights
        # QR weights for stochastic ocp 
        self.Q = model._Q
        self.R = model._R
        self.W = model._Cov_w 
        # casadi model
        self.model = model
        self.casadi_model = model.casadi_model
        # acados model
        self.__fill_acados_model()      
        # dimensions
        self.nx = self.acados_model.x.size()[0]
        self.nu = self.acados_model.u.size()[0]
        self.ny = self.nx + self.nu + self.swing_cost_weights.shape[0]
        # contact location bound
        self.step_bound = model._step_bound
        if model._robot_type == 'QUADRUPED':
            self.nb_contacts = 4
        elif model._robot_type == 'HUMANOID':
            self.nb_contacts = 2    
        # create optimal control problem
        self.ocp = AcadosOcp()
        self.ocp.model = self.acados_model  
        if MPC:
            self.ocp.dims.N = self.N_mpc
        else:
            self.ocp.dims.N = self.N_traj    
        # initialize stuff
        self.__fill_init_params()
        self.__init_pinocchio_robot()
        self.__create_swing_foot_cost_ref()
        # set ocp costs
        self.__fill_ocp_cost()
        # set ocp constraints
        self.__fill_ocp_constraints()
        self.__fill_ocp_solver_settings()
        # create solution struct
        if MPC:
            self.__generate_mpc_refs()
        # create Acados solver
        self.acados_solver = AcadosOcpSolver(
            self.ocp, json_file="acados_ocp.json", build=True, generate=True
            )
        # if WARM_START:
        #     self.__warm_start(x_ref, u_ref)

    def __fill_acados_model(self):
        acados_model = AcadosModel()
        acados_model.f_impl_expr = self.casadi_model.f_impl_expr
        acados_model.f_expl_expr = self.casadi_model.f_expl_expr
        acados_model.name = self.casadi_model.model_name
        acados_model.xdot = self.casadi_model.xdot
        acados_model.x = self.casadi_model.x
        acados_model.u = self.casadi_model.u
        acados_model.z = self.casadi_model.z
        acados_model.p = self.casadi_model.p
        acados_model.z = self.casadi_model.z
        self.acados_model = acados_model

    def __fill_init_params(self): 
        self.ocp.parameter_values = np.zeros(self.casadi_model.p.shape[0])
        self.ocp.solver_options.__initialize_t_slacks = 0
    
    def __fill_ocp_cost(self):
        ny, nx = self.ny, self.nx
        x, u = self.casadi_model.x, self.casadi_model.u
        cost = self.ocp.cost
        # coefficient matrices
        Vx_e = np.eye(nx)
        # cost function weights
        Q = self.state_cost_weights
        cost.W = la.block_diag(
            Q, self.control_cost_weights, self.swing_cost_weights
            )
        cost.W_e = Q       
        # cost type
        cost.cost_type = 'NONLINEAR_LS'
        cost.cost_type_e = 'LINEAR_LS'
        # cost expressions
        ee_fk_pos = self.model.casadi_model.fk_q_bar_pos
        ee_frame_vel = self.model.casadi_model.ee_frame_vel
        self.ocp.model.cost_y_expr = vertcat(x, u, ee_frame_vel)
        self.ocp.model.cost_y_expr_e = x
        # initial state tracking reference
        cost.yref = np.zeros(ny)
        cost.yref_e = np.zeros(nx)
        cost.Vx_e = Vx_e

    def __fill_ocp_constraints(self):
        ocp = self.ocp
        x_init = np.concatenate(
            [self.x_init[0][:12], np.zeros(3), self.x_init[0][16:]]
        )
        # initial constraints
        ocp.constraints.x0 = x_init 
        if not self.RECEEDING_HORIZON:
            # initial constraints
            ocp.constraints.x0 = x_init 
            # terminal constraints
            # x_goal = self.x_init[-1]
            # self.ocp.constraints.idxbx_e = np.array(range(self.nx))
            # self.ocp.constraints.lbx_e = x_goal 
            # self.ocp.constraints.ubx_e = x_goal        
        if self.casadi_model.model_name == 'quadruped_centroidal_momentum_plus_leg_kinematics':
            nh = self.casadi_model.constraints.lb.shape[0] 
            ocp.model.con_h_expr = self.casadi_model.constraints.expr
            ocp.constraints.lh = self.casadi_model.constraints.lb
            ocp.constraints.uh = self.casadi_model.constraints.ub
            # linearized contact-location end-effector
            ng = 12
            ocp.constraints.C = np.zeros((ng, self.nx))
            ocp.constraints.D = np.zeros((ng, self.nu))
            ocp.constraints.lg = np.zeros(ng)
            ocp.constraints.ug = np.zeros(ng)
            # slacks on general linear constraints
            ocp.constraints.idxsg = np.array(range(ng))
            ocp.constraints.lsg = np.zeros(ng)
            ocp.constraints.usg = np.zeros(ng)
            # slacks on nonlinear constraints
            ocp.constraints.idxsh = np.array(range(nh))
            ocp.constraints.lsh = np.zeros(nh)
            ocp.constraints.ush = np.zeros(nh)
            # slack penalties
            L2_pen = 1e3
            L1_pen = 1e2 #1e0
            ocp.cost.Zl = L2_pen * np.ones(nh+ng)
            ocp.cost.Zu = L2_pen * np.ones(nh+ng)
            ocp.cost.zl = L1_pen * np.ones(nh+ng)
            ocp.cost.zu = L1_pen * np.ones(nh+ng)

        elif self.casadi_model.model_name == 'flat_foot_humanoid_centroidal_momentum_plus_leg_kinematics':
            self.ocp.model.con_h_expr = vertcat(
                self.casadi_model.friction_pyramid_constraints.expr, self.casadi_model.cop_constraints.expr
                )
            self.ocp.constraints.lh = np.concatenate(
                [self.casadi_model.friction_pyramid_constraints.lb, self.casadi_model.cop_constraints.lb]
                )    
            self.ocp.constraints.uh = np.concatenate(
                [self.casadi_model.friction_pyramid_constraints.ub, self.casadi_model.cop_constraints.ub]
                ) 

    def __fill_ocp_solver_settings(self):
        if self.RECEEDING_HORIZON:
            N = self.N_mpc
        else:
            N = self.N_traj    
        self.ocp.solver_options.tf = N*self.dt
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        # self.ocp.solver_options.ext_cost_num_hess = 1
        # self.ocp.solver_options.hpipm_mode = "SPEED"
        self.ocp.solver_options.integrator_type = "ERK"
        self.ocp.solver_options.sim_method_num_stages = 1
        self.ocp.solver_options.sim_method_num_steps = 1
        # self.ocp.solver_options.sim_method_newton_iter = 1
        self.ocp.solver_options.print_level = 0
        # self.ocp.solver_options.qp_solver_cond_N = N
        # ocp.solver_options.sim_method_newton_iter = 10
        ## ---------------------
        ##  NLP solver settings
        ## ---------------------
        # self.ocp.solver_options.nlp_solver_type = "SQP"
        self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
        # self.ocp.solver_options.nlp_solver_tol_stat = 1e-3
        # self.ocp.solver_options.nlp_solver_tol_eq = 1e-3
        # self.ocp.solver_options.nlp_solver_tol_ineq = 1e-3
        # self.ocp.solver_options.nlp_solver_tol_comp = 1e-1
        # self.ocp.solver_options.nlp_solver_max_iter=0
        # self.ocp.solver_options.nlp_solver_step_length=1e-20
        # self.ocp.solver_options.globalization = ['FIXED_STEP', 'MERIT_BACKTRACKING']
        # self.ocp.solver_options.alpha_min = 0.01
        # --------------------
        # - QP solver settings
        # --------------------
        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        # self.ocp.solver_options.levenberg_marquardt = 1e-6
        self.ocp.solver_options.qp_solver_warm_start = True
        # self.ocp.solver_options.print_level = 1

    def __generate_mpc_refs(self):
        nb_contacts, N_mpc = self.nb_contacts, self.N_mpc
        self.x_ref_mpc = self.x_init[:self.N_traj]
        contacts_logic_final = \
            self.contact_data['contacts_logic'][-1].reshape(1, nb_contacts)
        contacts_position_final = \
            self.contact_data['contacts_position'][-1].reshape(1, nb_contacts*3)
        contacts_orient_final = \
            self.contact_data['contacts_orient'][-1].reshape(1, nb_contacts, 3, 3)
        x_ref_mpc_final = self.x_ref_mpc[-1].reshape(1, self.nx)
        for _ in range(N_mpc):
            self.x_ref_mpc = np.concatenate([self.x_ref_mpc, x_ref_mpc_final], axis=0)
            self.contact_data['contacts_logic'] = np.concatenate(
                [self.contact_data['contacts_logic'], contacts_logic_final], axis=0
                )
            self.contact_data['contacts_position'] = np.concatenate(
                [self.contact_data['contacts_position'], contacts_position_final], axis=0
                )
            self.contact_data['contacts_orient'] = np.concatenate(
                [self.contact_data['contacts_orient'], contacts_orient_final], axis=0
                )
    
    def __warm_start(self, x_ref_N, u_ref_N):
        N_traj, N_mpc = self.N_traj, self.N_mpc
        solver = self.acados_solver
        if self.RECEEDING_HORIZON:
            N = N_mpc
        else:
            N = N_traj 
        for time_idx in range(N):
            solver.set(time_idx, 'x', x_ref_N[time_idx])
            solver.set(time_idx, 'u', u_ref_N[time_idx])
        solver.set(N, 'x', x_ref_N[N]) 

    def __init_pinocchio_robot(self):
        self.rmodel = self.model._rmodel
        self.rdata = self.rmodel.createData()
        frame_names = self.model._ee_frame_names
        self.flFootId = self.rmodel.getFrameId(frame_names[0])
        self.frFootId = self.rmodel.getFrameId(frame_names[1])
        self.hlFootId = self.rmodel.getFrameId(frame_names[2])
        self.hrFootId = self.rmodel.getFrameId(frame_names[3])

    def __create_swing_foot_cost_ref(self):
        q0 = self.model._q0
        rmodel, rdata = self.rmodel, self.rdata
        pinocchio.forwardKinematics(rmodel, rdata, q0)
        pinocchio.updateFramePlacements(rmodel, rdata)
        flFootId = self.flFootId
        frFootId = self.frFootId
        hlFootId = self.hlFootId
        hrFootId = self.hrFootId
        flFootPos0 = rdata.oMf[flFootId].translation
        frFootPos0 = rdata.oMf[frFootId].translation
        hlFootPos0 = rdata.oMf[hlFootId].translation
        hrFootPos0 = rdata.oMf[hrFootId].translation 
        com_ref = (flFootPos0 + frFootPos0 + hlFootPos0 + hrFootPos0)/4
        feet_tasks = []
        com_tasks = []
        # stance feet positions are not to be tracked since there 
        # is a zero frame velocity constraint to take care of that 
        # once a foot is in contact with the ground
        for gait in self.model._gait_templates:
            for phase in gait:
                if phase == 'doubleSupport':
                    com_phase_task, phase_feet_tasks = self.__create_foot_refs(
                        [flFootPos0, frFootPos0, hlFootPos0, hrFootPos0],
                        ['STANCE','STANCE','STANCE','STANCE'], 
                        nb_swing_feet=0, com_ref=com_ref
                        )
                elif phase == 'rflhStep':
                    com_phase_task, phase_feet_tasks = self.__create_foot_refs(
                        [flFootPos0, frFootPos0, hlFootPos0, hrFootPos0],
                        ['STANCE'  ,'SWING'    ,'SWING'    ,'STANCE'], 
                        nb_swing_feet=2, com_ref=com_ref
                        )
                elif phase == 'lfrhStep':
                    com_phase_task, phase_feet_tasks = self.__create_foot_refs(
                        [flFootPos0, frFootPos0, hlFootPos0, hrFootPos0],
                        ['SWING'   ,'STANCE'   ,'STANCE'   ,'SWING'], 
                        nb_swing_feet=2, com_ref=com_ref
                        )
                feet_tasks += phase_feet_tasks
                com_tasks  += com_phase_task    
        self.swing_feet_tasks = feet_tasks
        self.com_tasks = com_tasks

    def __create_foot_refs(self, feet_pos, feet_status, nb_swing_feet, com_ref):
        stepLength = self.model._gait['stepLength']
        stepHeight = self.model._gait['stepHeight']
        numKnots = self.model._gait['stepKnots'] 
        comPercentage = nb_swing_feet/4
        foot_tasks_total = []
        com_task_total = []
        for k in range(numKnots):
            foot_tasks = []
            for STATUS, pos in zip(feet_status, feet_pos):
                if STATUS == 'SWING':
                    # Defining a foot swing task given the step length
                    # resKnot = numKnots % 2
                    phKnots = numKnots / 2
                    if k < phKnots:
                        dp = np.array(
                            [stepLength*(k+1)/numKnots, 0., stepHeight*k/phKnots]
                            )
                    elif k == phKnots:
                        dp = np.array(
                            [stepLength*(k+1)/numKnots, 0., stepHeight]
                            )
                    else:
                        dp = np.array(
                            [stepLength*(k+1)/numKnots, 0., stepHeight*(1-float(k-phKnots)/phKnots)]
                            )
                    tref = pos + dp
                    foot_tasks += [pinocchio.SE3(np.eye(3), tref)]
                elif STATUS == 'STANCE':
                    foot_tasks += [pinocchio.SE3(np.eye(3), pos)]
            foot_tasks_total += [foot_tasks]
            if nb_swing_feet == 0 :
                com_task_total += [np.zeros(3)]
            else:
                com_task_total += [
                    np.array([stepLength*(k+1)/numKnots, 0., 0.])*comPercentage + com_ref
                    ]
        # update com and contact positions initial conditions    
        com_ref += [stepLength*comPercentage, 0., 0.]
        for STATUS, pos in zip(feet_status, feet_pos):
            if STATUS == 'SWING':
                pos += [stepLength, 0., 0.] 
        return com_task_total, foot_tasks_total

    def run_mpc(self):
        # trajectory references
        N_traj, N_mpc = self.N_traj, self.N_mpc
        contacts_logic = self.contact_data['contacts_logic']
        contacts_position = self.contact_data['contacts_position']
        contacts_norms = self.contact_data['contacts_orient']
        x_ref_mpc = self.x_ref_mpc
        # get acados solver object
        solver = self.acados_solver
        # create open-loop tuples
        self.X_sim = np.zeros((N_traj, self.N_mpc+1, self.nx))
        self.U_sim = np.zeros((N_traj, self.N_mpc, self.nu))
        # create closed-loop tuples
        X_sol = np.zeros((N_traj+1, self.nx))
        U_sol = np.zeros((N_traj+1, self.nu))
        x0 = self.x_init[0]
        X_sol[0] = x0
        # moving horizon loop
        for traj_time_idx in range(N_traj):
            # get horizon references 
            horizon_range = range(traj_time_idx, traj_time_idx+N_mpc)
            x_ref_N = x_ref_mpc[horizon_range] 
            contacts_logic_N = contacts_logic[horizon_range]
            contacts_norms_N = contacts_norms[horizon_range]
            contacts_position_N = contacts_position[horizon_range] 
            # OCP loop
            for mpc_time_idx in range(N_mpc):
                x_ref_k = x_ref_N[mpc_time_idx]
                contacts_logic_k = contacts_logic_N[mpc_time_idx]
                contacts_position_k = contacts_position_N[mpc_time_idx]
                contacts_norms_k = contacts_norms_N[mpc_time_idx].flatten()
                contact_params_k = np.concatenate(
                    [contacts_logic_k, contacts_position_k, contacts_norms_k, ]
                    )            
                # update paramters and tracking cost
                y_ref_k = np.concatenate([x_ref_k, np.zeros(self.nu)])
                solver.set(mpc_time_idx, 'p', contact_params_k)
                solver.cost_set(mpc_time_idx,'yref', y_ref_k)    
            # terminal constraints
            # x_ref_terminal = x_ref_mpc[traj_time_idx+N_mpc]
            # self.ocp.constraints.idxbx_e = np.array(range(self.nx))
            # self.ocp.constraints.lbx_e = x_ref_terminal 
            # self.ocp.constraints.ubx_e = x_ref_terminal     
            # update terminal cost
            solver.cost_set(N_mpc,'yref', x_ref_k)
            # solve OCP
            print('\n'+'=' * 50)
            print('MPC Iteration ' + str(traj_time_idx))
            print('-' * 50)
            # update initial conditon
            solver.set(0, "lbx", x0)
            solver.set(0, "ubx", x0)
            if self.ocp.solver_options.nlp_solver_type == 'SQP_RTI':
                # QP preparation rti_phase:
                print('starting RTI preparation phase ' + '...')
                solver.options_set('rti_phase', 1)
                t_prep = time.time()
                status = solver.solve()
                elapsed_prep = time.time() - t_prep
                print('RTI preparation phase took ' + str(elapsed_prep) + " seconds")
                # feedback rti_phase
                print('starting RTI feedback phase ' + '...')
                solver.options_set('rti_phase', 2)
                t_feedback = time.time()
                status = solver.solve()
                elapsed_feedback = time.time() - t_feedback
                print('RTI feedback phase took ' + str(elapsed_feedback) + " seconds")
                solver.print_statistics()
                if status == 0:
                    print("HOORAY ! found a solution after :", 
                    elapsed_prep+elapsed_feedback, " seconds")
                else:
                    raise Exception(f'acados returned status {status}.')
            else:
                t = time.time()
                status = solver.solve()
                elapsed_time= time.time() - t
                solver.print_statistics()
                if status == 0:
                    print("HOORAY ! found a solution after :", 
                    elapsed_time, " seconds")
                else:
                    raise Exception(f'acados returned status {status}.')
            # save open-loop trajectories
            x_sol = np.array([solver.get(i,"x") for i in range(N_mpc+1)])
            u_sol = np.array([solver.get(i,"u") for i in range(N_mpc)])
            self.X_sim[traj_time_idx] = x_sol
            self.U_sim[traj_time_idx] = u_sol
            # save closed-loop solution
            X_sol[traj_time_idx+1] = x_sol[0]
            U_sol[traj_time_idx+1] = u_sol[0]
            # warm-start solver from the previous solution 
            x0 = x_sol[1]
        return X_sol, U_sol        

    def solve(self):
        model = self.model
        STOCHASTIC = model._STOCHASTIC_OCP
        if STOCHASTIC:
            self.eta = norm.ppf(1-self.model._beta_u)
        if self.RECEEDING_HORIZON:
            X_sim, U_sim = self.run_mpc()
        else:
            nx, N = self.nx, self.N_traj
            # reference trajectories 
            x_ref_N = self.x_init
            u_ref_N = self.u_ref
            # initial warm-start 
            x_warm_start_N = np.copy(x_ref_N)
            u_warm_start_N = np.copy(u_ref_N)
            # nominal contact data
            contacts_logic_N = self.contact_data['contacts_logic']
            contacts_position_N = self.contact_data['contacts_position'] 
            contacts_norms_N = self.contact_data['contacts_orient']
            solver = self.acados_solver
            nb_lateral_contact_loc_constr = self.nb_contacts*2
            nb_vertical_contact_loc_constr = self.nb_contacts
            # get dynamics jacobians
            A = self.model.casadi_model.Jx_fun
            B = self.model.casadi_model.Ju_fun
            # get forward kinematics and jacobians expressions
            ee_fk = [model.fk_FR, model.fk_FL, model.fk_HR, model.fk_HL]
            ee_jacobians = [model.J_FR, model.J_FL, model.J_HR, model.J_HL]
            step_bound = self.step_bound
            swing_feet_tasks = self.swing_feet_tasks
            com_tasks = self.com_tasks
            qref_base_k =  x_ref_N[0][12:16]
            # solver main loop
            for SQP_iter in range(100):
                Sigma_k = np.zeros((nx, nx))
                for time_idx in range(N):
                    # get stage and terminal references
                    x_ref_k = np.concatenate(
                            [x_ref_N[time_idx][:12],
                            np.zeros(3),
                             x_ref_N[time_idx][16:]]
                        )
                    x_ref_goal = np.concatenate(
                            [x_ref_N[-1][:12],
                            np.zeros(3), 
                             x_ref_N[-1][16:]]
                        )     
                    if SQP_iter == 0:
                        x_warm_start_k = x_ref_k
                        x_warm_start_goal = x_ref_goal
                    else:    
                        x_warm_start_k = x_warm_start_N[time_idx]
                        x_warm_start_goal = x_warm_start_N[-1]

                    u_warm_start_k = u_warm_start_N[time_idx]
                    y_ref_k = np.concatenate(
                        [x_ref_k,
                        u_warm_start_k, 
                        np.zeros(12)]
                        )
                    # get contact parameters and base orientation references
                    contacts_logic_k = contacts_logic_N[time_idx]
                    contacts_position_k = contacts_position_N[time_idx]
                    contacts_norms_k = contacts_norms_N[time_idx].flatten()
                    qref_base_k = x_ref_N[time_idx][12:16]
                    params_k = np.concatenate(
                        [contacts_logic_k,
                         contacts_position_k, 
                         contacts_norms_k,  
                         qref_base_k]
                    )      
                    # set parameters and cost references
                    solver.set(time_idx, 'p', params_k)
                    solver.cost_set(time_idx,'yref', y_ref_k)
                    # propagate uncertainties ()
                    if STOCHASTIC:
                        A_k = A(x_warm_start_k, u_warm_start_k, params_k)
                        B_k = B(x_warm_start_k, u_warm_start_k,params_k)  
                        K_k = self.compute_riccatti_gains(A_k, B_k)
                        Sigma_next = self.propagate_covariance(A_k, B_k, K_k, Sigma_k) 
                    # set warm-start 
                    solver.set(time_idx, 'x', x_warm_start_k)
                    solver.set(time_idx, 'u', u_warm_start_k)
                    # get the generalized position vector at the jth SQP iteration
                    base_posj_k = np.array(x_warm_start_k[9:12])
                    lambdaj_k = x_warm_start_k[12:15]
                    joint_posj_k = np.array(x_warm_start_k[15:])
                    qj_k = np.concatenate(
                        [base_posj_k, 
                        np.array(
                            model.casadi_model.q_plus(qref_base_k, lambdaj_k)
                            ).squeeze(),
                        joint_posj_k]
                    )
                    dqj_k = np.concatenate(
                        [base_posj_k, 
                        lambdaj_k,
                        joint_posj_k]
                        )
                    # intialize linearized contact location constraint matrices
                    C_lateral = np.zeros((nb_lateral_contact_loc_constr, nx))
                    lg_lateral = np.zeros(nb_lateral_contact_loc_constr)
                    ug_lateral = np.zeros(nb_lateral_contact_loc_constr)
                    C_vertical = np.zeros((nb_vertical_contact_loc_constr, nx))
                    lg_vertical = np.zeros(nb_vertical_contact_loc_constr)
                    ug_vertical = np.zeros(nb_vertical_contact_loc_constr)
                    # fill constraints for each end-effector
                    for contact_idx in range(self.nb_contacts):
                        contact_position_param = \
                            contacts_position_k[contact_idx*3:(contact_idx*3)+3] 
                        ee_linear_jac = ee_jacobians[contact_idx](q=qj_k)['J'][:3, :]
                        contact_logic = contacts_logic_k[contact_idx]               
                        ee_Jlin_times_qj = ee_linear_jac @ dqj_k
                        contact_fk = ee_fk[contact_idx](q=qj_k)['ee_pos']
                        # lateral part (x-y direction)
                        for lateral_constraint_idx in range(2):
                            idx = contact_idx*2 + lateral_constraint_idx
                            constraint_row = ee_linear_jac[lateral_constraint_idx, :]
                            C_lateral[idx, 9:] = contact_logic*(constraint_row)
                            temp = contact_position_param[lateral_constraint_idx] \
                                   + ee_Jlin_times_qj[lateral_constraint_idx] \
                                   - contact_fk[lateral_constraint_idx, :]
                            # compute back-off mangnitude (only for lateral part)
                            if STOCHASTIC:
                                backoff = self.eta*np.sqrt(
                                    (constraint_row @ Sigma_next[9:, 9:]) @ constraint_row.T 
                                    )
                            else:
                                backoff = 0.
                            lg_lateral[idx] = contact_logic*(
                                temp - step_bound + backoff
                                )
                            ug_lateral[idx] = contact_logic*(
                                temp + step_bound - backoff
                                )
                        # vertical part (z-direction)
                        C_vertical[contact_idx, 9:] = contact_logic*(ee_linear_jac[2, :])
                        temp = ee_Jlin_times_qj[2] - contact_fk[2] + contact_position_param[2] 
                        lg_vertical[contact_idx] = contact_logic*temp 
                        ug_vertical[contact_idx] = contact_logic*temp
                    # add contatenated constraints 
                    C_total = np.concatenate([C_lateral, C_vertical], axis=0)
                    ub_total = np.concatenate([lg_lateral, lg_vertical], axis=0)
                    lb_total = np.concatenate([ug_lateral, ug_vertical], axis=0)                   
                    solver.constraints_set(time_idx, 'C', C_total, api='new')
                    solver.constraints_set(time_idx, 'lg', lb_total)
                    solver.constraints_set(time_idx, 'ug', ub_total)
                # set terminal references
                solver.cost_set(N,'yref', x_ref_goal)
                solver.set(N, 'x', x_warm_start_goal)
                # terminal constraints
                # self.ocp.constraints.idxbx_e = np.array(range(self.nx))
                # self.ocp.constraints.lbx_e = x_ref_terminal 
                # self.ocp.constraints.ubx_e = x_ref_terminal      
                # solve ocp
                t = time.time()
                status = solver.solve()
                # solver.print_statistics()
                if status == 0:
                    elapsed = time.time() - t
                    print("HOORAY found a solution after :", elapsed, "seconds")
                else:
                    print('Acados solver failed with error status = ', status)
                # save open-loop trajectories
                X_sim = np.array([solver.get(i,"x") for i in range(N+1)])
                U_sim = np.array([solver.get(i,"u") for i in range(N)])
                residuals = solver.get_residuals()
                if SQP_iter > 0:
                    print(
                        "difference between two SQP iterations = ",
                         np.linalg.norm(X_sim - x_warm_start_N)
                         )
                    print("residuals after ", SQP_iter, "SQP_RTI iterations:\n", residuals)
                    if np.linalg.norm(residuals[2]) < 5e-4:
                        print(
                            '[SUCCESS] .. ', " breaking at SQP iteration number: ",
                             SQP_iter
                            )
                        break
                x_warm_start_N = X_sim
                u_warm_start_N = U_sim 
                # if STOCHASTIC:
                #     Sigma_k = Sigma_next  

        return X_sim, U_sim    
 
    def compute_riccatti_gains(self, A, B):
        Q, R  = self.Q, self.R
        P = np.copy(Q)
        At_P  = A.T @ P
        At_P_B = At_P @ B
        P = (Q + (A.T @ P)) - (
            At_P_B @ la.solve((R + B.T @ P @ B), At_P_B.T)
            )
        return -la.solve((R + (B.T @ P @ B)), (B.T @ P @ A))

    def propagate_covariance(self, A, B, K, Sigma):
        AB = np.hstack([A, B])
        Sigma_Kt = Sigma @ K.T 
        Sigma_xu = np.vstack(
            [np.hstack([Sigma    , Sigma_Kt]),
            np.hstack([Sigma_Kt.T, K@Sigma_Kt])]
            )
        return AB @ Sigma_xu @ AB.T + self.W

if __name__ == "__main__":
    from centroidal_plus_legKinematics_casadi_model import CentroidalPlusLegKinematicsCasadiModel
    from wholebody_croccodyl_solver import WholeBodyDDPSolver
    from wholebody_croccodyl_model import WholeBodyModel
    import conf_solo12_trot_step_adjustment as conf
    import pinocchio as pin
    import numpy as np
    import utils

    # DDP warm-start
    wbd_model = WholeBodyModel(conf)
    ddp_planner = WholeBodyDDPSolver(wbd_model, MPC=False, WARM_START=False)
    ddp_planner.solve()
    ddp_sol = ddp_planner.get_solution_trajectories()
    centroidal_warmstart = ddp_sol['centroidal']
    q_warmstart = ddp_sol['jointPos']
    qdot_warmstart = ddp_sol['jointVel']
    x_warmstart = []
    u_warmstart = []
    rmodel, rdata = conf.rmodel, conf.rdata
    for k in range(len(centroidal_warmstart)):
        x_warmstart.append(np.concatenate([centroidal_warmstart[k], q_warmstart[k]]))
        u_warmstart.append(np.concatenate([np.zeros(12), qdot_warmstart[k]]))
    # nominal traj-opt
    model_nom = CentroidalPlusLegKinematicsCasadiModel(conf, STOCHASTIC_OCP=False)
    solver_nom = CentroidalPlusLegKinematicsAcadosSolver(
        model_nom, x_warmstart, u_warmstart, MPC=False)
    x_nom, u_nom = solver_nom.solve()
    # stochastic traj-opt
    model_stoch = CentroidalPlusLegKinematicsCasadiModel(conf, STOCHASTIC_OCP=True)
    solver_stoch = CentroidalPlusLegKinematicsAcadosSolver(
        model_stoch, x_warmstart, u_warmstart, MPC=False)
    x_stoch, u_stoch = solver_stoch.solve()
    robot = conf.solo12.robot
    dt = conf.dt
    dt_ctrl = 0.01
    N_ctrl =  int(dt/dt_ctrl)
    # initialize end-effector trajectories
    FL_nom = np.zeros((3, x_nom.shape[0]-1)).astype(np.float32)
    FR_nom = np.zeros((3, x_nom.shape[0]-1)).astype(np.float32)
    HL_nom = np.zeros((3, x_nom.shape[0]-1)).astype(np.float32)
    HR_nom = np.zeros((3, x_nom.shape[0]-1)).astype(np.float32)
    FL_stoch = np.zeros((3, x_stoch.shape[0]-1)).astype(np.float32)
    FR_stoch = np.zeros((3, x_stoch.shape[0]-1)).astype(np.float32)
    HL_stoch = np.zeros((3, x_stoch.shape[0]-1)).astype(np.float32)
    HR_stoch = np.zeros((3, x_stoch.shape[0]-1)).astype(np.float32)
    # visualize in meshcat
    if conf.WITH_MESHCAT_DISPLAY:
        viz = pin.visualize.MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model)
        try:
            viz.initViewer(open=True)
        except ImportError as err:
            print(err)
            sys.exit(0)
        viz.loadViewerModel()
        # visualize nominal motion
        for k in range(conf.N-1):
            q_base_next = np.array(
                    model_nom.casadi_model.q_plus(
                        x_warmstart[k][3:7], x_nom[k, 12:15]                        )
                    ).squeeze()
            q = np.concatenate(
                [x_nom[k, 9:12], 
                q_base_next,
                    x_nom[k, 15:]]
                )
            pin.framesForwardKinematics(rmodel, rdata, q)
            FL_nom[:, k] = rdata.oMf[rmodel.getFrameId('FL_FOOT')].translation
            FR_nom[:, k] = rdata.oMf[rmodel.getFrameId('FR_FOOT')].translation
            HL_nom[:, k] = rdata.oMf[rmodel.getFrameId('HL_FOOT')].translation
            HR_nom[:, k] = rdata.oMf[rmodel.getFrameId('HR_FOOT')].translation            
            for j in range(20): 
                viz.display(q)
        # visualize stochastic motion
        for k in range(conf.N-1):
            q_base_next = np.array(
                    model_stoch.casadi_model.q_plus(
                        x_warmstart[k][3:7], x_stoch[k, 12:15]
                        )
                    ).squeeze()
            q = np.concatenate(
                    [x_stoch[k, 9:12], 
                    q_base_next,
                     x_stoch[k, 15:]]
                    )
            pin.framesForwardKinematics(rmodel, rdata, q)
            FL_stoch[:, k] = rdata.oMf[rmodel.getFrameId('FL_FOOT')].translation
            FR_stoch[:, k] = rdata.oMf[rmodel.getFrameId('FR_FOOT')].translation
            HL_stoch[:, k] = rdata.oMf[rmodel.getFrameId('HL_FOOT')].translation
            HR_stoch[:, k] = rdata.oMf[rmodel.getFrameId('HR_FOOT')].translation            
            for j in range(20):
                viz.display(q)        
    # display nominal end-effector trajectories
    utils.addLineSegment(viz, 'FL_trajectory_nom', FL_nom, [1,0,0,1])
    utils.addLineSegment(viz, 'FR_trajectory_nom', FR_nom, [1,0,0,1])
    # utils.addLineSegment(viz, 'HL_trajectory_nom', HL_nom, [1,0,0,1])
    # utils.addLineSegment(viz, 'HR_trajectory_nom', HR_nom, [1,0,0,1])
    # display stochastic end-effector trajectories
    utils.addLineSegment(viz, 'FL_stoch', FL_stoch, [0,1,0,1])
    utils.addLineSegment(viz, 'FR_stoch', FR_stoch, [0,1,0,1])
    # utils.addLineSegment(viz, 'HL_stoch', HL_stoch, [0,1,0,1])
    # utils.addLineSegment(viz, 'HR_stoch', HR_stoch, [0,1,0,1])




