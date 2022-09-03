from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import scipy.linalg as la
import pinocchio as pin 
import casadi as ca
import numpy as np
import time 

class ManipulatorSolverAcados:
    # constructor
    def __init__(self, model, x_ref, MPC=False, WARM_START=True):
        self.model = model
        self.RECEEDING_HORIZON = MPC
        # timing
        self.dt = model.dt
        self.dt_ctrl = model.dt_ctrl
        self.N_mpc = model.N_mpc
        self.N_traj = model.N_traj
        self.N_interpol = int(model.dt/model.dt_ctrl)
        # # tracking reference 
        self.x_init = x_ref
        self.ee_frame_name = model.ee_frame_name
        # cost function weights
        self.alpha = model.alpha
        self.beta  = model.beta
        self.gamma = model.gamma
        # casadi model
        self.casadi_model = model.casadi_model
        # casadi pinocchio 
        self.rmodel = model.rmodel
        self.cmodel = model.cmodel 
        self.cdata = model.cdata
        # acados model
        self.__fill_acados_model()      
        # dimensions
        self.nx = self.acados_model.x.size()[0]
        self.nu = self.acados_model.u.size()[0]
        self.ny = self.nx + self.nu + 3
        # create optimal control problem
        self.ocp = AcadosOcp()
        self.ocp.model = self.acados_model
        if MPC:
            self.ocp.dims.N = self.N_mpc
        else:
            self.ocp.dims.N = self.N_traj    
        # set ocp costs
        self.__fill_ocp_cost()
        # set ocp constraints
        self.x_limits = model.x_limits
        self.u_limits = model.u_limits
        self.__fill_ocp_constraints()
        self.__fill_ocp_solver_settings()
        # create solution struct
        if MPC:
            self.__generate_mpc_refs()
        # create Acados solver
        self.acados_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")
        # if WARM_START:
        #     self.__warm_start(x_ref)
    
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
        self.acados_model = acados_model    

    def __fill_ocp_cost(self):
        x, u = self.casadi_model.x, self.casadi_model.u
        nx, nu, ny = self.nx, self.nu, self.ny
        # coefficient matrices
        Vx = np.eye(nx) @ x
        Vu = np.eye(nu) @ u
        # cost function weights
        self.ocp.cost.W = la.block_diag(self.alpha, self.beta, self.gamma)
        self.ocp.cost.W_e = la.block_diag(self.alpha, self.gamma)       
        # cost type
        self.ocp.cost.cost_type = "NONLINEAR_LS"
        self.ocp.cost.cost_type_e = "NONLINEAR_LS"
        # Nonlinear least square end-effector cost
        ee_position = self.model.forward_kinematics(self.ee_frame_name)           
        self.ocp.model.cost_y_expr = ca.vertcat(Vx, Vu, ee_position)
        self.ocp.model.cost_y_expr_e = ca.vertcat(Vx, ee_position)                                          
        # initial state tracking reference
        self.ocp.cost.yref = np.zeros(ny)
        self.ocp.cost.yref_e = np.zeros(ny-nu)
    
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
        self.ocp.solver_options.nlp_solver_type = "SQP"
        # self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
        self.ocp.solver_options.nlp_solver_tol_stat = 1e-3
        self.ocp.solver_options.nlp_solver_tol_eq = 1e-3
        self.ocp.solver_options.nlp_solver_tol_ineq = 1e-3
        self.ocp.solver_options.nlp_solver_tol_comp = 1e-3
        # --------------------
        # - QP solver settings
        # --------------------
        # self.ocp.solver_options.levenberg_marquardt = 1e-6
        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        # self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_OSQP"
        self.ocp.solver_options.qp_solver_warm_start = True

    def __fill_ocp_constraints(self):
        nx, nu = self.nx, self.nu
        x_limits, u_limits = self.x_limits, self.u_limits
        # initial constraints
        self.ocp.constraints.x0 = self.casadi_model.initial_state
        if not self.RECEEDING_HORIZON:
            # terminal constraints
            x_goal = self.x_init[-1]
            self.ocp.model.con_h_expr_e = self.model.forward_kinematics(self.ee_frame_name)   
            self.ocp.constraints.lh_e = x_goal 
            self.ocp.constraints.uh_e = x_goal        
        # state box constraints
        self.ocp.constraints.idxbx = np.array(range(nx))
        self.ocp.constraints.ubx = x_limits
        self.ocp.constraints.lbx = -x_limits
        # control box constraints
        self.ocp.constraints.idxbu = np.array(range(nu))
        self.ocp.constraints.ubu = u_limits
        self.ocp.constraints.lbu = -u_limits
     
    def __generate_mpc_refs(self): self.x_ref_mpc = np.repeat(
        np.copy(self.x_init[-1]), self.N_mpc, axis=0
        )
    
    def solve(self):
        if self.RECEEDING_HORIZON:
            X_sim, U_sim = self.run_mpc()
        else:
            nx, nu = self.nx, self.nu
            ee_pos_N, N = self.x_init, self.N_traj
            solver = self.acados_solver
            # set stage references
            for time_idx in range(N):
                ee_pos_k = ee_pos_N[time_idx]
                solver.cost_set(
                    time_idx,'yref', np.concatenate([np.zeros(nx+nu), ee_pos_k])
                    ) 
            # set terminal references
            solver.cost_set(
                N,'yref', np.concatenate([np.zeros(nx), ee_pos_k])
                ) 
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
        return X_sim, U_sim    
    
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

if __name__ == "__main__":
    from manipulator_casadi_model import fixedBaseManipulatorCasadiModel
    import conf_kuka as conf
    import pinocchio as pin
    import casadi as ca
    x_ref = np.reshape(conf.target, (1, 3))
    x_ref_N = np.repeat(x_ref, conf.N_traj, axis=0)
    solver = ManipulatorSolverAcados(
        fixedBaseManipulatorCasadiModel(conf), x_ref_N, MPC=False
        )    
    X_sim, U_sim = solver.solve()
    robot = conf.robot
    viz = pin.visualize.MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model
        )
    viz.initViewer(open=True)
    viz.loadViewerModel()
    for i in range(conf.N_traj-1):
        x_des, tau_des = solver.interpolate_one_step(
            X_sim[i, :7], X_sim[i+1, :7], 
            X_sim[i, 7:14], X_sim[i+1, 7:14],
            U_sim[i], U_sim[i+1] 
        )
        for t in range(10):
            viz.display(x_des[t, :7])
    rmodel = conf.rmodel 
    rdata = rmodel.createData()
    pin.framesForwardKinematics(rmodel, rdata, np.array(X_sim[-1, :7]))
    t_ee = rdata.oMf[rmodel.getFrameId('contact')]
    print(np.allclose(t_ee.translation, conf.target))