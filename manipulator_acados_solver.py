from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from scipy.sparse import csc_matrix 
from qpsolvers import solve_qp
from scipy.stats import norm
import scipy.linalg as la
import pinocchio as pin 
import numpy as np
import time 

class ManipulatorSolverAcados:
    # constructor
    def __init__(self, model, ee_ref, MPC=False, WARM_START=True):
        self.model = model
        self.RECEEDING_HORIZON = MPC
        # timing
        self.dt = model.dt
        self.dt_ctrl = model.dt_ctrl
        self.N_mpc = model.N_mpc
        self.N_traj = model.N_traj
        self.N_interpol = int(model.dt/model.dt_ctrl)
        # # tracking reference 
        self.x_init = ee_ref
        self.ee_frame_name = model.ee_frame_name
        # SQP cost function weights
        self.alpha = model.alpha
        self.beta  = model.beta
        self.gamma = model.gamma
        # casadi model
        self.casadi_fwdDyn_model = model.casadi_fwdDyn_model
        self.forwardKinematics = model.forwardKinematics
        # casadi pinocchio 
        self.rmodel = model.rmodel
        self.rdata = model.rdata
        self.cmodel = model.cmodel 
        self.cdata = model.cdata
        # acados model
        self.__fill_acados_model()      
        # dimensions
        self.nx = self.acados_model.x.size()[0]
        self.nu = self.acados_model.u.size()[0]
        self.ny = self.nx + self.nu + 3
        # joint limits
        self.x_limits = model.x_limits
        self.u_limits = model.u_limits
        # create optimal control problem
        self.ocp = AcadosOcp()
        self.ocp.model = self.acados_model
        if MPC:
            self.ocp.dims.N = self.N_mpc
        else:
            self.ocp.dims.N = self.N_traj
        # solve an IK QP to guide the SQP
        if WARM_START:
            self.__IK_warm_start(ee_ref)
        # initialize paramters
        self.__fill_init_params()

        # set ocp costs
        self.__fill_ocp_cost()
        # set ocp constraints
        self.__fill_ocp_constraints()
        self.__fill_ocp_solver_settings()
        # create solution struct
        if MPC:
            self.__generate_mpc_refs()
        # create Acados solver
        self.acados_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")
       
    def __fill_init_params(self): self.ocp.parameter_values = np.zeros(self.casadi_fwdDyn_model.p.shape[0])

    def __fill_acados_model(self):
        acados_model = AcadosModel()
        acados_model.f_impl_expr = self.casadi_fwdDyn_model.f_impl_expr
        acados_model.f_expl_expr = self.casadi_fwdDyn_model.f_expl_expr
        acados_model.name = self.casadi_fwdDyn_model.model_name
        acados_model.xdot = self.casadi_fwdDyn_model.xdot
        acados_model.x = self.casadi_fwdDyn_model.x
        acados_model.u = self.casadi_fwdDyn_model.u
        acados_model.z = self.casadi_fwdDyn_model.z
        acados_model.p = self.casadi_fwdDyn_model.p
        self.acados_model = acados_model    

    def __IK_warm_start(self, ee_des):
        # Ik cost function weights
        w_vel = self.model.IK_cost_weight
        Kp    = self.model.IK_Kp_gains
        w_vel = 1e-1
        dt = self.dt
        N = ee_des.shape[1]
        qk = self.casadi_fwdDyn_model.initial_state[:7]
        x_opt = np.zeros((14, N))
        x_opt[:7, 0] = qk
        rmodel, rdata = self.rmodel, self.rdata
        ee_idx = rmodel.getFrameId(self.ee_frame_name)
        for time_idx in range (N):
            pin.framesForwardKinematics(rmodel, rdata, qk)
            J = pin.computeFrameJacobian(
                rmodel, rdata, qk, ee_idx, 
                )        
            # Hessian
            Q = w_vel*(J.T @ J) + 1e-6*np.eye(7)
            # gradient
            frame_vel_residual = ee_des[3:6, time_idx] +\
                 Kp@(ee_des[:3, time_idx] - rdata.oMf[ee_idx].translation)
            v = -w_vel*frame_vel_residual.T @ J[:3]
            # kinematic constraints
            G = np.concatenate([-np.eye(7), np.eye(7), -np.eye(7), np.eye(7)], axis=0)
            lqdot = self.x_limits[7:]
            uqdot = self.x_limits[7:]
            lq = (self.x_limits[:7] + qk)/dt 
            uq = (self.x_limits[:7] - qk)/dt
            h = np.concatenate([lq, uq, lqdot, uqdot])
            # solve QP
            qdot_opt_k = solve_qp(
                P=csc_matrix(Q), q=v, G=csc_matrix(G), h=h, solver="osqp"
                ) 
            x_opt[7:, time_idx] = np.copy(qdot_opt_k)
            qk = qk + (qdot_opt_k*dt)
            if time_idx < N-1:
                x_opt[:7, time_idx+1] = np.copy(qk)
        self.x_warm_start = x_opt 

    def __fill_ocp_cost(self):
        nx, nu = self.nx, self.nu
        ny = nx + nu
        # coefficient matrices
        Vx = np.zeros((ny, nx))
        Vx[:nx, :] = np.eye(nx)
        Vu = np.zeros((ny, nu))
        Vu[nx:, :] = np.eye(nu)
        Vx_e = np.eye(nx)
        # cost function weights
        self.ocp.cost.W = la.block_diag(self.alpha, self.beta)
        self.ocp.cost.W_e = la.block_diag(self.alpha)       
        # cost type
        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"
        self.ocp.cost.yref_e = np.zeros(nx)
        self.ocp.cost.yref = np.zeros(ny)
        self.ocp.cost.Vx_e = Vx_e
        self.ocp.cost.Vx = Vx
        self.ocp.cost.Vu = Vu

    def __fill_ocp_solver_settings(self):
        if self.RECEEDING_HORIZON:
            N = self.N_mpc
        else:
            N = self.N_traj    
        self.ocp.solver_options.tf = N*self.dt
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        # self.ocp.solver_options.hessian_approx = "EXACT"
        self.ocp.solver_options.integrator_type = "ERK"
        self.ocp.solver_options.sim_method_num_stages = 1
        self.ocp.solver_options.sim_method_num_steps = 1
        self.ocp.solver_options.print_level = 1
        ## ---------------------
        ##  NLP solver settings
        ## ---------------------
        # self.ocp.solver_options.nlp_solver_type = "SQP"
        self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
        self.ocp.solver_options.nlp_solver_tol_stat = 1e-6
        self.ocp.solver_options.nlp_solver_tol_eq = 1e-6
        self.ocp.solver_options.nlp_solver_tol_ineq = 1e-6
        self.ocp.solver_options.nlp_solver_tol_comp = 1e-6
        # --------------------
        # - QP solver settings
        # --------------------
        # self.ocp.solver_options.levenberg_marquardt = 1e-6
        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        # self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_OSQP"
        self.ocp.solver_options.qp_solver_warm_start = True

    def __fill_ocp_constraints(self):
        q = self.model.q 
        nx, nu = self.nx, self.nu
        x_limits, u_limits = self.x_limits, self.u_limits
        # P = ca.reshape(self.casadi_fwdDyn_model.p, (nx, nx))
        # ----------------------------
        # Fwd dynamics NLP constraints
        # ----------------------------
        # initial constraints
        self.ocp.constraints.x0 = self.casadi_fwdDyn_model.initial_state
        if not self.RECEEDING_HORIZON:
            # terminal constraints
            x_goal = self.x_warm_start[:, -1]
            self.ocp.constraints.idxbx_e = np.array(range(self.nx))
            self.ocp.constraints.lbx_e = x_goal 
            self.ocp.constraints.ubx_e = x_goal       
        # state box constraints
        self.ocp.constraints.idxbx = np.array(range(nx))
        self.ocp.constraints.ubx = x_limits
        self.ocp.constraints.lbx = -x_limits
        # control box constraints
        self.ocp.constraints.idxbu = np.array(range(nu))
        self.ocp.constraints.ubu = u_limits
        self.ocp.constraints.lbu = -u_limits
        # end-effector constraint
        self.ocp.constraints.C = np.zeros((4, nx))
        self.ocp.constraints.D = np.zeros((4, nu))
        self.ocp.constraints.lg = np.zeros(4)
        self.ocp.constraints.ug = np.zeros(4)

    def __generate_mpc_refs(self): self.x_ref_mpc = np.repeat(
        np.copy(self.x_init[-1]), self.N_mpc, axis=0
        )
    
    def compute_riccatti_gains(self, A, B, niter=2):
        Q, R  = self.model.Q, self.model.R
        P = np.copy(Q)
        for _ in range(niter):
            At_P  = A.T @ P
            At_P_B = At_P @ B
            P = (Q + (A.T @ P)) - (
                At_P_B @ la.solve((R + B.T @ P @ B), At_P_B.T)
                )
        return -la.solve((R + (B.T @ P @ B)), (B.T @ P @ A))

    def propagate_covariance(self, A, B, K, Sigma):
        AB = np.hstack([A, B])
        Sigma_Kt = Sigma @ K.T 
        Sigma_xu = np.vstack([np.hstack([Sigma      , Sigma_Kt]),
                              np.hstack([Sigma_Kt.T , K@Sigma_Kt])])
        return AB @ Sigma_xu @ AB.T + self.model.cov_w_dt

    def solve(self):
        x_ref_N, N = self.x_warm_start, self.N_traj
        x_ref_N = x_ref_N.T
        x_warm_start_N = np.concatenate([x_ref_N, x_ref_N[-1].reshape(1,14)], axis=0)
        u_warm_start_N = np.zeros((N, 7))
        nx, nu = self.nx, self.nu
        sens_x = self.casadi_fwdDyn_model.A
        sens_u = self.casadi_fwdDyn_model.B
        x_obs_total = self.model.x_obs
        nb_obs = x_obs_total.shape[0]
        delta = 0.25
        if self.model.STOCHASTIC_OCP:
            eta = norm.ppf(1-self.model.epsilon)
        # solver main loop
        for SQP_iter in range(100):
            if self.RECEEDING_HORIZON:
                X_sim, U_sim = self.run_mpc()
            else:
                nx, nu = self.nx, self.nu
                solver = self.acados_solver
                x_goal_ref = x_ref_N[-1]
                x_goal_warm_start = x_warm_start_N[-1]
                Sigma_k = np.zeros((nx, nx))
                # set stage references
                for time_idx in range(N):
                    x_ref_k = x_ref_N[time_idx]
                    x_warm_start_k = x_warm_start_N[time_idx]
                    u_warm_start_k = u_warm_start_N[time_idx]
                    solver.set(time_idx, 'x', x_warm_start_k)
                    solver.set(time_idx, 'u', u_warm_start_k)
                    solver.cost_set(
                        time_idx,'yref', np.concatenate([x_ref_k, np.zeros(nu)])
                        )    
                    if self.model.STOCHASTIC_OCP:
                        # compute jacobians at the current SQP solution iterate
                        # and propagate covariances 
                        A_k = sens_x(x_warm_start_k, u_warm_start_k)
                        B_k = sens_u(x_warm_start_k, u_warm_start_k)
                        K_k = self.compute_riccatti_gains(A_k, B_k)
                        Sigma_next = self.propagate_covariance(A_k, B_k, K_k, Sigma_k) 
                    qk = x_warm_start_k[:7]
                    x_ee = self.model.forwardKinematics(qk)[1]
                    J = self.model.jacobian(qk)
                    cons_expr = np.zeros((nb_obs, nx))
                    lg = np.zeros(nb_obs)
                    ug = 1e8*np.ones(nb_obs)
                    for x_obs_idx, x_obs in enumerate(x_obs_total):
                        distance_fun_norm = np.linalg.norm(x_ee - x_obs)
                        distance_fun_normal = \
                            (J[0,:]@(x_ee[0]-x_obs[0])) + (J[1,:]@(x_ee[1]-x_obs[1])) + (J[2,:]@(x_ee[2]-x_obs[2]))/distance_fun_norm   
                        if self.model.STOCHASTIC_OCP:
                            backoff_magintude = \
                                eta*np.sqrt((distance_fun_normal @ (Sigma_next[:7, :7]) @ distance_fun_normal.T))
                        else:
                            backoff_magintude = 0.
                        cons_expr[x_obs_idx, :7] = distance_fun_normal
                        lg[x_obs_idx] = \
                            delta + backoff_magintude - distance_fun_norm + (distance_fun_normal @ qk)
                    solver.constraints_set(time_idx, 'C', cons_expr, api='new')
                    solver.constraints_set(time_idx, 'lg', lg)
                    solver.constraints_set(time_idx, 'ug', ug) 
                # set terminal references
                solver.set(N, 'x', x_goal_warm_start)
                solver.cost_set(
                    N,'yref', np.concatenate([x_goal_ref])
                    ) 
                # terminal constraints
                self.ocp.constraints.idxbx_e = np.array(range(self.nx))
                self.ocp.constraints.lbx_e = x_goal_ref 
                self.ocp.constraints.ubx_e = x_goal_ref  
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
                x_sim = np.array([solver.get(i,"x") for i in range(N+1)])
                u_sim = np.array([solver.get(i,"u") for i in range(N)])
                print("difference between two SQP iterations = ", np.linalg.norm(x_sim - x_warm_start_N))
                if np.linalg.norm(x_sim - x_warm_start_N) <= 1e-6:
                    print("YESSSSSSSSSSSSSSSSSSSSSSSSS !! .. breaking at SQP iteration number: ", SQP_iter)
                    break
                else:
                    x_warm_start_N = x_sim
                    u_warm_start_N = u_sim    
        return x_sim, u_sim    
    
    def interpolate_one_step(self, q, q_next, qdot, qdot_next, tau, tau_next):
        nq, nv = len(q), len(qdot)
        N_interpol, rmodel = self.N_interpol, self.rmodel
        x_interpol = np.zeros((N_interpol, nq+nv))
        tau_interpol = np.zeros((N_interpol, len(tau)))
        dtau = (tau_next - tau)/float(N_interpol)
        dqdot = (qdot_next - qdot)/float(N_interpol)
        dt = self.dt_ctrl/self.dt
        for interpol_idx in range(N_interpol):
            tau_interpol[interpol_idx] = tau + interpol_idx*dtau 
            x_interpol[interpol_idx, :nq] = pin.interpolate(rmodel, q, q_next, interpol_idx*dt)
            x_interpol[interpol_idx, nq:] = qdot + interpol_idx*dqdot        
        return x_interpol, tau_interpol

    # # plot optimized joint positions vs IK warm-start
    # q_ref = solver.x_warm_start.T
    # fig, (q1, q2, q3, q4, q5, q6, q7) = plt.subplots(7, 1, sharex=True)
    # time = np.arange(0, np.round(conf.N_traj*conf.dt, 2),conf.dt)
    # q1.plot(time, X_sim[:-1, 0])
    # q1.plot(time, q_ref[:, 0])
    # q2.plot(time, X_sim[:-1, 1])
    # q2.plot(time, q_ref[:, 1])
    # q3.plot(time, X_sim[:-1, 2])
    # q3.plot(time, q_ref[:, 2])
    # q4.plot(time, X_sim[:-1, 3])
    # q4.plot(time, q_ref[:, 3])
    # q5.plot(time, X_sim[:-1, 4])
    # q5.plot(time, q_ref[:, 4])
    # q6.plot(time, X_sim[:-1, 5])
    # q6.plot(time, q_ref[:, 5])
    # q7.plot(time, X_sim[:-1, 6])
    # q7.plot(time, q_ref[:, 6])
    # plt.show()
