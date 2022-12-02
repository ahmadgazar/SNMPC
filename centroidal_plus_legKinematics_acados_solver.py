from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import scipy.linalg as la
from casadi import *
import numpy as np
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
        # casadi model
        self.model = model
        self.casadi_model = model.casadi_model
        # acados model
        self.__fill_acados_model()      
        # dimensions
        self.nx = self.acados_model.x.size()[0]
        self.nu = self.acados_model.u.size()[0]
        self.ny = self.nx + self.nu 
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
        # set ocp costs
        self.__fill_init_params()
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
        # acados_model.f_expl_expr = self.casadi_model.f_expl_expr
        acados_model.name = self.casadi_model.model_name
        acados_model.xdot = self.casadi_model.xdot
        acados_model.x = self.casadi_model.x
        acados_model.u = self.casadi_model.u
        acados_model.z = self.casadi_model.z
        acados_model.p = self.casadi_model.p
        self.acados_model = acados_model

    def __fill_init_params(self): 
        self.ocp.parameter_values = np.zeros(self.casadi_model.p.shape[0])
        self.ocp.solver_options.__initialize_t_slacks = 0
    
    def __fill_ocp_cost(self):
        ny, nx, nu = self.ny, self.nx, self.nu
        cost = self.ocp.cost
        # coefficient matrices
        Vx = np.zeros((ny, nx))
        Vx[:nx, :] = np.eye(nx)
        Vu = np.zeros((ny, nu))
        Vu[nx:, :] = np.eye(nu)
        Vx_e = np.eye(nx)
        # cost function weights
        Q, R = self.Q, self.R
        self.ocp.cost.W = la.block_diag(Q, R)
        self.ocp.cost.W_e = la.block_diag(Q)       
        # initial state tracking reference
        cost.cost_type = "LINEAR_LS"
        cost.cost_type_e = "LINEAR_LS"
        # initial state tracking reference
        cost.yref_e = np.zeros(nx)
        cost.yref = np.zeros(ny)
        cost.Vx_e = Vx_e
        cost.Vx = Vx
        cost.Vu = Vu

    def __fill_ocp_constraints(self):
        ocp = self.ocp
        # initial constraints
        ocp.constraints.x0 = self.x_init[0] 
        if not self.RECEEDING_HORIZON:
            # initial constraints
            ocp.constraints.x0 = self.x_init[0] 
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
            # slack on general linear constraints
            ocp.constraints.idxsg = np.array(range(ng))
            ocp.constraints.lsg = np.zeros(ng)
            ocp.constraints.usg = np.zeros(ng)
            # slacks on nonlinear constraints
            L2_pen = 1e4
            L1_pen = 1e0 #1e0
            ocp.constraints.idxsh = np.array(range(nh))
            ocp.constraints.lsh = np.zeros(nh)
            ocp.constraints.ush = np.zeros(nh)
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
        # self.ocp.solver_options.hessian_approx = "EXACT"
        self.ocp.solver_options.integrator_type = "IRK"
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
            nx = self.nx
            N = self.N_traj
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
            # solver main loop
            for SQP_iter in range(100):
                x_goal_ref = x_ref_N[-1]
                x_goal_warm_start = x_warm_start_N[-1]
                # set stage references
                for time_idx in range(N):
                    x_ref_k = x_ref_N[time_idx]
                    if SQP_iter == 0:
                        x_warm_start_k = np.concatenate(
                            [x_warm_start_N[time_idx][0:3], x_warm_start_N[time_idx][7:]]
                        )
                    else:    
                        x_warm_start_k = x_warm_start_N[time_idx]
                    u_warm_start_k = u_warm_start_N[time_idx]
                    y_ref_k = np.concatenate(
                        [x_warm_start_k, u_ref_N[time_idx]]
                        )
                    contacts_logic_k = contacts_logic_N[time_idx]
                    contacts_position_k = contacts_position_N[time_idx]
                    contacts_norms_k = contacts_norms_N[time_idx].flatten()
                    params_k = np.concatenate(
                        [contacts_logic_k, contacts_position_k, contacts_norms_k, x_ref_k[3:7]]
                        )            
                    solver.set(time_idx, 'p', params_k)
                    solver.cost_set(time_idx,'yref', y_ref_k)
                    # warm-start 
                    solver.set(time_idx, 'x', x_warm_start_k)
                    solver.set(time_idx, 'u', u_warm_start_k)
                    # linearize contact location constraints
                    qk = x_warm_start_k[:3]
                    

                    C_lateral = np.zeros((nb_lateral_contact_loc_constr, nx))
                    lg_lateral = np.zeros(nb_lateral_contact_loc_constr)
                    ug_lateral = np.zeros(nb_lateral_contact_loc_constr)
                    C_vertical = np.zeros((nb_vertical_contact_loc_constr, nx))
                    lg_vertical = np.zeros(nb_vertical_contact_loc_constr)
                    ug_vertical = np.zeros(nb_vertical_contact_loc_constr)
                    for contact_idx in range(self.nb_contacts):
                        # lateral part (x-y direction)
                        contact_position_param = contacts_position_k[contact_idx*3:(contact_idx*3)+3] 
                        contact_jacobian = ee_jacobians[contact_idx](q=qk)['J']
                        contact_logic = contacts_logic_k[contact_idx]
                        Jk_times_qk = contact_jacobian @ qdot_k
                        contact_fk = ee_fk[contact_idx](q=qk)['ee_pos']
                        for lateral_constraint_idx in range(2):
                            idx = contact_idx*2 + lateral_constraint_idx
                            C_lateral[idx, 9:] = contact_logic*(
                                contact_jacobian[lateral_constraint_idx]
                                )
                            temp = -contact_position_param[lateral_constraint_idx] \
                                   -contact_fk[lateral_constraint_idx] \
                                   +Jk_times_qk[lateral_constraint_idx]  
                            lg_lateral[idx] = contact_logic*(temp - step_bound)
                            ug_lateral[idx] = contact_logic*(temp + step_bound)
                        # vertical part (z-direction)
                        C_vertical[contact_idx, 9:] = contact_logic*(contact_jacobian[2])
                        temp = contact_position_param[2] - contact_fk[2] + Jk_times_qk[2]
                        lg_vertical[contact_idx] = contact_logic*temp 
                        ug_vertical[contact_idx] = contact_logic*temp
                    C_total = np.concatenate([C_lateral, C_vertical], axis=0)
                    ub_total = np.concatenate([lg_lateral, lg_vertical], axis=0)
                    lb_total = np.concatenate([ug_lateral, ug_vertical], axis=0)                   
                    solver.constraints_set(time_idx, 'C', C_total, api='new')
                    solver.constraints_set(time_idx, 'lg', lb_total)
                    solver.constraints_set(time_idx, 'ug', ub_total)
                # set terminal references
                solver.cost_set(N,'yref', x_goal_ref)
                solver.set(N, 'x', x_goal_warm_start)
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
                print("difference between two SQP iterations = ", np.linalg.norm(X_sim - x_warm_start_N))
                print("residuals after ", SQP_iter, "SQP_RTI iterations:\n", solver.get_residuals())
                if np.linalg.norm(X_sim-x_warm_start_N) < 5e-4:
                    print("YESSSSSSSSSSSSSSSSSSSSSSSSS !! .. breaking at SQP iteration number: ", SQP_iter)
                    break
                else:
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
        u_warmstart.append(np.concatenate([np.zeros(12), np.zeros(18), np.zeros(3)]))
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
        for k in range(x.shape[0]):
            for j in range(10):
                q = np.concatenate([x[k, 9:]])        
                viz.display(q)