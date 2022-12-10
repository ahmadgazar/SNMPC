from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from utils import normalize_quaternion, log_map_casadi, rotToQuat_casadi
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
        self.Q = model._state_cost_weights
        self.R = model._control_cost_weights
        self.S = model._swing_foot_cost_weights
        # casadi model
        self.model = model
        self.casadi_model = model.casadi_model
        # acados model
        self.__fill_acados_model()      
        # dimensions
        self.nx = self.acados_model.x.size()[0]
        self.nu = self.acados_model.u.size()[0]
        self.ny = self.nx + self.nu + self.S.shape[0]
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
        ny, nx, nu = self.ny, self.nx, self.nu
        x, u = self.casadi_model.x, self.casadi_model.u
        p = self.casadi_model.p
        cost = self.ocp.cost
        # coefficient matrices
        # Vx = np.zeros((ny, nx))
        # Vx[:nx, :] = np.eye(nx)
        # Vu = np.zeros((ny, nu))
        # Vu[nx:, :] = np.eye(nu)
        Vx_e = np.eye(nx)
        # cost function weights
        Q, R, S = self.Q, self.R, self.S
        # cost.W = la.block_diag(Q, R, S)
        cost_W = la.block_diag(Q, R)
        cost.W_e = Q       
        # initial state tracking reference
        ee_fk_pos = self.model.casadi_model.fk_q_bar_pos
        ee_fk_rot = self.model.casadi_model.fk_q_bar_rot
        q_identity = MX([0.,0.,0., 1.0])
        cost.cost_type = 'EXTERNAL'
        cost.cost_type_e = 'EXTERNAL'
        # com tracking cost
        com_error = x[:3] - p[-15:-12]
        com_cost = (.5*com_error.T @ Q[:3,:3]) @ com_error
        # swing foot frame orientation cost
        ee_orient_cost = []
        for contact_idx in range(self.nb_contacts):
            idx = contact_idx*3
            ee_orient_cost = vertcat(
                ee_orient_cost,
                log_map_casadi(
                    rotToQuat_casadi(ee_fk_rot[idx:idx+3, :]), q_identity)
                )
        # swing foot position tracking cost
        ee_pos_error = ee_fk_pos - p[-12::]
        ee_pos_cost = (.5*ee_pos_error.T @ S) @ ee_pos_error
        # state regularization cost
        state_reg_cost = (.5*x[3:].T @ Q[3:, 3:]) @ x[3:]   
        # control regularization cost 
        control_reg_cost = (.5*u.T @ R) @ u
        self.ocp.model.cost_y_expr = vertcat(
            com_cost, 
            state_reg_cost, 
            control_reg_cost,
            ee_pos_cost,
            ee_orient_cost
        )
        self.ocp.model.cost_y_expr_e = vertcat(com_cost, state_reg_cost)  
        # cost.cost_type = "LINEAR_LS"
        # cost.cost_type_e = "LINEAR_LS"
        # initial state tracking reference
        # cost.yref_e = np.zeros(nx)
        # cost.yref = np.zeros(ny)
        # cost.yref_e = np.zeros(nx)
        # cost.yref = np.zeros(ny)
        # cost.Vx_e = Vx_e
        # cost.Vx = Vx
        # cost.Vu = Vu

    def __fill_ocp_constraints(self):
        ocp = self.ocp
        x_init = np.concatenate(
            [self.x_init[0][:12], self.x_init[0][16:]]
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
            # ng = 12
            # ocp.constraints.C = np.zeros((ng, self.nx))
            # ocp.constraints.D = np.zeros((ng, self.nu))
            # ocp.constraints.lg = np.zeros(ng)
            # ocp.constraints.ug = np.zeros(ng)
            # slack on general linear constraints
            # ocp.constraints.idxsg = np.array(range(ng))
            # ocp.constraints.lsg = np.zeros(ng)
            # ocp.constraints.usg = np.zeros(ng)
            # slacks on nonlinear constraints
            L2_pen = 1e6
            L1_pen = 1e0 #1e0
            ocp.constraints.idxsh = np.array(range(12, nh))
            ocp.constraints.lsh = np.zeros(nh-12)
            ocp.constraints.ush = np.zeros(nh-12)
            ocp.cost.Zl = L2_pen * np.ones(nh-12)
            ocp.cost.Zu = L2_pen * np.ones(nh-12)
            ocp.cost.zl = L1_pen * np.ones(nh-12)
            ocp.cost.zu = L1_pen * np.ones(nh-12)

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
        self.ocp.solver_options.nlp_solver_tol_stat = 1e-3
        self.ocp.solver_options.nlp_solver_tol_eq = 1e-3
        self.ocp.solver_options.nlp_solver_tol_ineq = 1e-3
        self.ocp.solver_options.nlp_solver_tol_comp = 1e-1
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
                    [contacts_logic_k, contacts_position_k, contacts_norms_k]
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
                    print("HOORAY ! found a solution after :", elapsed_prep+elapsed_feedback, " seconds")
                else:
                    raise Exception(f'acados returned status {status}.')
            else:
                t = time.time()
                status = solver.solve()
                elapsed_time= time.time() - t
                solver.print_statistics()
                if status == 0:
                    print("HOORAY ! found a solution after :", elapsed_time, " seconds")
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
        if self.RECEEDING_HORIZON:
            X_sim, U_sim = self.run_mpc()
        else:
            dt, N = self.dt, self.N_traj
            nx, nu = self.nx, self.nu
            x_ref_N = self.x_init
            u_ref_N = self.u_ref
            x_warm_start_N = np.copy(x_ref_N)
            u_warm_start_N = np.copy(u_ref_N)
            contacts_logic_N = self.contact_data['contacts_logic']
            contacts_position_N = self.contact_data['contacts_position'] 
            contacts_norms_N = self.contact_data['contacts_orient']
            solver = self.acados_solver
            nb_lateral_contact_loc_constr = self.nb_contacts*2
            nb_vertical_contact_loc_constr = self.nb_contacts
            model = self.model
            # get forward kinematics and jacobians expressions
            ee_fk = [model.fk_FR, model.fk_FL, model.fk_HR, model.fk_HL]
            ee_jacobians = [model.J_FR, model.J_FL, model.J_HR, model.J_HL]
            step_bound = self.step_bound
            swing_feet_tasks = self.swing_feet_tasks
            com_tasks = self.com_tasks
            q0_base_N = []
            # solver main loop
            for SQP_iter in range(100):
                for time_idx in range(N):
                    # get stage and terminal references
                    x_ref_k = np.concatenate(
                            [x_ref_N[time_idx][:12],
                             x_ref_N[time_idx][16:]]
                        )
                    x_ref_goal = np.concatenate(
                            [x_ref_N[-1][:12], 
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
                        [com_tasks[time_idx],
                        np.zeros(9),
                        x_ref_k[12:],
                        np.zeros(12),
                        np.zeros(6), 
                        u_warm_start_N[time_idx][18:], 
                        swing_feet_tasks[time_idx][0].translation,
                        swing_feet_tasks[time_idx][1].translation,
                        swing_feet_tasks[time_idx][2].translation,
                        swing_feet_tasks[time_idx][3].translation]
                        )
                    # get contact parameters
                    contacts_logic_k = contacts_logic_N[time_idx]
                    contacts_position_k = contacts_position_N[time_idx]
                    contacts_norms_k = contacts_norms_N[time_idx].flatten()
                    # get base orientation reference 
                    if SQP_iter == 0:
                        q0_base_k =  x_ref_N[time_idx][12:16]
                    else:
                        q0_base_k = np.array(q0_base_N[time_idx]).squeeze() 
                    params_k = np.concatenate(
                        [contacts_logic_k, contacts_position_k, contacts_norms_k, q0_base_k]
                    )      
                    solver.set(time_idx, 'p', params_k)
                    solver.cost_set(time_idx,'yref', y_ref_k)
                    # warm-start 
                    solver.set(time_idx, 'x', x_warm_start_k)
                    solver.set(time_idx, 'u', u_warm_start_k)
                    # get the generalized position vector at the jth SQP iteration
                    # base_posj_k = np.array(x_warm_start_k[:3])
                    # joint_posj_k = np.array(x_warm_start_k[12:])
                    # qj_k = np.concatenate(
                    #     [base_posj_k, 
                    #     np.array(
                    #         model.casadi_model.integrate_base(q0_base_k, u_warm_start_k[15:18])
                    #         ).squeeze(),
                    #     joint_posj_k]
                    # )
                    # # intialize linearized contact location constraint matrices
                    # C_lateral = np.zeros((nb_lateral_contact_loc_constr, nx))
                    # D_lateral = np.zeros((nb_lateral_contact_loc_constr, nu))
                    # lg_lateral = np.zeros(nb_lateral_contact_loc_constr)
                    # ug_lateral = np.zeros(nb_lateral_contact_loc_constr)
                    # C_vertical = np.zeros((nb_vertical_contact_loc_constr, nx))
                    # D_vertical = np.zeros((nb_vertical_contact_loc_constr, nu))
                    # lg_vertical = np.zeros(nb_vertical_contact_loc_constr)
                    # ug_vertical = np.zeros(nb_vertical_contact_loc_constr)
                    # # fill constraints for each end-effector
                    # for contact_idx in range(self.nb_contacts):
                    #     contact_position_param = \
                    #         contacts_position_k[contact_idx*3:(contact_idx*3)+3] 
                    #     ee_jacobian = ee_jacobians[contact_idx](q=qj_k)['J']
                    #     contact_logic = contacts_logic_k[contact_idx]
                    #     ee_jacobian_base_pos = ee_jacobian[:, :3]
                    #     ee_jacobian_joints_pos = ee_jacobian[:, 6:]
                    #     ee_Jlin_times_base_pos = ee_jacobian_base_pos[:3, :] @ base_posj_k
                    #     ee_Jlin_times_joint_pos = ee_jacobian_joints_pos[:3, :] @ joint_posj_k
                    #     Jj_k_state = np.concatenate(
                    #         [ee_jacobian_base_pos, ee_jacobian_joints_pos],
                    #          axis=1
                    #         )
                    #     Jj_k_control = (ee_jacobian[:, 3:6]*dt)    
                    #     contact_fk = ee_fk[contact_idx](q=qj_k)['ee_pos']
                    #     # lateral part (x-y direction)
                    #     for lateral_constraint_idx in range(2):
                    #         idx = contact_idx*2 + lateral_constraint_idx
                    #         C_lateral[idx, 9:] = contact_logic*(
                    #             Jj_k_state[lateral_constraint_idx]
                    #             )
                    #         D_lateral[idx, 15:18] = contact_logic*(
                    #             Jj_k_control[lateral_constraint_idx]
                    #             )    
                    #         temp = - contact_position_param[lateral_constraint_idx] \
                    #                - contact_fk[lateral_constraint_idx] \
                    #                + ee_Jlin_times_base_pos[lateral_constraint_idx] \
                    #                + ee_Jlin_times_joint_pos[lateral_constraint_idx] 
                    #         lg_lateral[idx] = contact_logic*(temp - step_bound)
                    #         ug_lateral[idx] = contact_logic*(temp + step_bound)
                    #     # vertical part (z-direction)
                    #     C_vertical[contact_idx, 9:] = contact_logic*(Jj_k_state[2])
                    #     D_vertical[contact_idx, 15:18] = contact_logic*(Jj_k_control[2])
                    #     temp = contact_position_param[2] \
                    #            - contact_fk[2] \
                    #            + ee_Jlin_times_base_pos[2] \
                    #            + ee_Jlin_times_joint_pos[2]
                    #     lg_vertical[contact_idx] = contact_logic*temp 
                    #     ug_vertical[contact_idx] = contact_logic*temp
                    # # add contatenated constraints 
                    # C_total = np.concatenate([C_lateral, C_vertical], axis=0)
                    # D_total = np.concatenate([D_lateral, D_vertical], axis=0)
                    # ub_total = np.concatenate([lg_lateral, lg_vertical], axis=0)
                    # lb_total = np.concatenate([ug_lateral, ug_vertical], axis=0)                   
                    # solver.constraints_set(time_idx, 'C', C_total, api='new')
                    # solver.constraints_set(time_idx, 'D', D_total, api='new')
                    # solver.constraints_set(time_idx, 'lg', lb_total)
                    # solver.constraints_set(time_idx, 'ug', ub_total)
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
                    print("difference between two SQP iterations = ", np.linalg.norm(X_sim - x_warm_start_N))
                    print("residuals after ", SQP_iter, "SQP_RTI iterations:\n", residuals)
                    if np.linalg.norm(residuals[2]) < 1e-3:
                        print("YESSSSSSSSSSSSSSSSSSSSSSSSS !! .. breaking at SQP iteration number: ", SQP_iter)
                        break
                # integrate base based on current solution of omega and q0_base
                for j in range(N):
                    if SQP_iter == 0:
                        q0_base_N += [x_ref_N[time_idx][12:16]]
                    else:     
                        q0_plus = self.casadi_model.integrate_base(
                            normalize_quaternion(q0_base_N[j]), U_sim[j, 15:18]
                        )
                        q0_base_N[j] = q0_plus   
                x_warm_start_N = X_sim
                u_warm_start_N = U_sim    
        return X_sim, U_sim    
 
 
if __name__ == "__main__":
    from centroidal_plus_legKinematics_casadi_model import CentroidalPlusLegKinematicsCasadiModel
    from wholebody_croccodyl_solver import WholeBodyDDPSolver
    from wholebody_croccodyl_model import WholeBodyModel
    import conf_solo12_trot_step_adjustment as conf
    import pinocchio as pin
    import numpy as np

    wbd_model = WholeBodyModel(conf)
    ddp_planner = WholeBodyDDPSolver(wbd_model, MPC=False, WARM_START=False)
    ddp_planner.solve()
    ddp_sol = ddp_planner.get_solution_trajectories()
    centroidal_warmstart = ddp_sol['centroidal']
    q_warmstart = ddp_sol['jointPos']
    qdot_warmstart = ddp_sol['jointVel']
    x_warmstart = []
    u_warmstart = []
    for k in range(len(centroidal_warmstart)):
        x_warmstart.append(np.concatenate([centroidal_warmstart[k], q_warmstart[k]]))
        u_warmstart.append(np.concatenate([np.zeros(12), qdot_warmstart[k]]))
    model = CentroidalPlusLegKinematicsCasadiModel(conf)
    solver = CentroidalPlusLegKinematicsAcadosSolver(model, x_warmstart, u_warmstart, MPC=False)
    x, u = solver.solve()
    robot = conf.solo12.robot
    if conf.WITH_MESHCAT_DISPLAY:
        viz = pin.visualize.MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model)
        try:
            viz.initViewer(open=True)
        except ImportError as err:
            print(
                "Error while initializing the viewer. Make sure you installed Python meshcat"
            )
            print(err)
            sys.exit(0)
        viz.loadViewerModel()
        for k in range(u.shape[0]):
            q_base_next = np.array(
                        model.casadi_model.integrate_base(x_warmstart[k][3:7],u[k, 15:18])).squeeze()
            for j in range(10):
                q = np.concatenate(
                    [x[k, 9:12], 
                    q_base_next,
                     x[k, 12:]]
                    )        
                viz.display(q)



