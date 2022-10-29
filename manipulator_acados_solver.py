from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from scipy.sparse import csc_matrix 
from qpsolvers import solve_qp
from scipy.stats import norm
import scipy.linalg as la
import pinocchio as pin 
import casadi as ca
import numpy as np
import utils
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
        qk = self.casadi_fwdDyn_model.initial_state[:7]
        x_opt = np.zeros((14, xdot.shape[1]))
        x_opt[:7, 0] = qk
        N = xdot.shape[1]
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
        # Linear least square end-effector cost
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
        self.ocp.solver_options.nlp_solver_type = "SQP"
        # self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
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
        P = ca.reshape(self.casadi_fwdDyn_model.p, (nx, nx))
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
        # ee_constraints = ca.vertcat(
        #     ca.norm_2(self.model.forwardKinematics(q)[1]),
        #     ca.norm_2(self.model.forwardKinematics(q)[1]),
        #     ca.norm_2(self.model.forwardKinematics(q)[1])
        # )
        x_obs = np.array(
                        [[0.65-0.05, -0.05, 0.4+0.05],    #top left                          
                         [0.65+0.05, -0.05, 0.4+0.05],    #top right
                         [0.65-0.05, -0.05, 0.4-0.05],    #bottom left
                         [0.65+0.05, -0.05, 0.4-0.05],    #bottom right
                         ])
        delta = 0.3                 
        eta = norm.ppf(1-self.model.epsilon)
        l1_constraint_mat = utils.l1_permut_mat(x_obs.shape[1])
        nb_l1_constraints = l1_constraint_mat.shape[0]
        expr = []
        for constraint_nb in range(x_obs.shape[0]): 
                expr = ca.vertcat(
                    expr, 
                    ca.norm_1(
                        self.model.forwardKinematics(q)[1]-x_obs[constraint_nb]
                        ) - delta
                )
                # expr = ca.vertcat(expr,
                #     l1_constraint_mat @ (self.model.forwardKinematics(q)[1] - x_obs[constraint_nb]) - delta*np.ones(nb_l1_constraints)     
                # )
        self.ocp.model.con_h_expr = expr 
        self.ocp.constraints.lh = np.zeros(expr.shape[0])
        self.ocp.constraints.uh =  1e8*np.ones(expr.shape[0])
        # print(ee_constraints.shape)
        # for constraint_idx in range(con_h_expr.shape[0]):
        #     constraint_i = con_h_expr[constraint_idx]
        #     for j in range(nx):
        #         constraint_i -= eta*ca.sqrt(constraint_i*P[j,j]*constraint_i)  
        # self.ocp.model.con_h_expr = con_h_expr
        # self.ocp.constraints.lh = np.array([0., 0., 0.])
        # self.ocp.constraints.uh = np.array([100, 100, 100])

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
        if self.RECEEDING_HORIZON:
            X_sim, U_sim = self.run_mpc()
        else:
            nx, nu = self.nx, self.nu
            x_ref_N, N = self.x_warm_start, self.N_traj
            solver = self.acados_solver
            x_goal = self.x_warm_start[:, -1]
            Sigma_k = np.zeros((nx, nx))
            # set stage references
            for time_idx in range(N):
                A_k = solver.get_from_qp_in(time_idx, 'A')
                B_k = solver.get_from_qp_in(time_idx, 'B')
                K_k = self.compute_riccatti_gains(A_k, B_k)
                Sigma_next = self.propagate_covariance(A_k, B_k, K_k, Sigma_k)
                solver.set(time_idx, 'p', Sigma_next.flatten(order='f'))
                x_ref_k = x_ref_N[:, time_idx]
                solver.set(time_idx, 'x', x_ref_k)
                solver.cost_set(
                    time_idx,'yref', np.concatenate([x_ref_k, np.zeros(nu)])
                    )
                # update covariance at the next time-step
                Sigma_k = np.copy(Sigma_next)     
            # set terminal references
            solver.set(N, 'x', x_goal)
            solver.cost_set(
                N,'yref', np.concatenate([x_goal])
                ) 
            # terminal constraints
            self.ocp.constraints.idxbx_e = np.array(range(self.nx))
            self.ocp.constraints.lbx_e = x_goal 
            self.ocp.constraints.ubx_e = x_goal  
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
    from utils import compute_5th_order_poly_traj, compute_3rd_order_poly_traj
    import conf_kuka as conf
    import pinocchio as pin
    import casadi as ca
    import meshcat

    def meshcat_material(r, g, b, a):
        material = meshcat.geometry.MeshPhongMaterial()
        material.color = int(r * 255) * 256 ** 2 + int(g * 255) * 256 + int(b * 255)
        material.opacity = a
        return material

    def addViewerBox(viz, name, sizex, sizey, sizez, rgba):
        if isinstance(viz, pin.visualize.MeshcatVisualizer):
            viz.viewer[name].set_object(meshcat.geometry.Box([sizex, sizey, sizez]),
                                    meshcat_material(*rgba))

    def meshcat_transform(x, y, z, q, u, a, t):
        return np.array(pin.XYZQUATToSE3([x, y, z, q, u, a, t]))
    
    def applyViewerConfiguration(viz, name, xyzquat):
        if isinstance(viz, pin.visualize.MeshcatVisualizer):
            viz.viewer[name].set_transform(meshcat_transform(*xyzquat))
        
    x_ref = np.reshape(conf.ee_target_pos, (1, 3))
    x_ref_N = np.repeat(x_ref, conf.N_traj, axis=0)
    T = conf.N_traj*conf.dt
    x, xdot, _ = compute_5th_order_poly_traj(conf.ee_init_pos, conf.ee_target_pos, T, conf.dt)
    ee_ref = np.concatenate([x, xdot], axis=0)
    solver = ManipulatorSolverAcados(
        fixedBaseManipulatorCasadiModel(conf), ee_ref, MPC=False
        )
    qk = solver.casadi_fwdDyn_model.initial_state[:7]
    q_sym = solver.model.q
    cmodel = solver.cmodel
    cdata =  solver.cdata
    rmodel = conf.rmodel
    rdata = conf.rdata  
    X_sim, U_sim = solver.solve()
    ee_sim = np.zeros((X_sim.shape[0], 3))
    robot = conf.robot
    viz = pin.visualize.MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model
        )
    viz.initViewer(open=True)
    viz.loadViewerModel()
    addViewerBox(viz, 'world/box1', .1, .1, .0, [1., .2, .2, .5])
    applyViewerConfiguration(viz, 'world/box1', [0.65, -0., 0.4, 1, 0, 0, 0])
    addViewerBox(viz, 'world/box2', .1, .0, .1, [1., .2, .2, .5])
    applyViewerConfiguration(viz, 'world/box2', [0.65, -0.05, 0.4+0.05, 1, 0, 0, 0])
    for i in range(conf.N_traj-1):
        x_des, tau_des = solver.interpolate_one_step(
            X_sim[i, :7], X_sim[i+1, :7], 
            X_sim[i, 7:14], X_sim[i+1, 7:14],
            U_sim[i], U_sim[i+1]
        )
        for t in range(10):
            q_des = x_des[t, :7]
            
            pin.framesForwardKinematics(rmodel, rdata, q_des)
            ee_pos = rdata.oMf[rmodel.getFrameId('contact')].translation
            # applyViewerConfiguration(viz, 'world/box1', [ee_pos[0], ee_pos[1], ee_pos[2], 1, 0, 0, 0])
            # applyViewerConfiguration(viz, 'world/box2', [ee_pos[0], ee_pos[1]-0.05, ee_pos[2]+0.05, 1, 0, 0, 0])
            if i == int(ee_sim.shape[0]/2):
                ee_sim[t, :] = ee_pos
            viz.display(q_des)
    q_ref = solver.x_warm_start.T
    import matplotlib.pyplot as plt
    fig, (q1, q2, q3, q4, q5, q6, q7) = plt.subplots(7, 1, sharex=True)
    time = np.arange(0, np.round(conf.N_traj*conf.dt, 2),conf.dt)
    q1.plot(time, X_sim[:-1, 0])
    q1.plot(time, q_ref[:, 0])
    q2.plot(time, X_sim[:-1, 1])
    q2.plot(time, q_ref[:, 1])
    q3.plot(time, X_sim[:-1, 2])
    q3.plot(time, q_ref[:, 2])
    q4.plot(time, X_sim[:-1, 3])
    q4.plot(time, q_ref[:, 3])
    q5.plot(time, X_sim[:-1, 4])
    q5.plot(time, q_ref[:, 4])
    q6.plot(time, X_sim[:-1, 5])
    q6.plot(time, q_ref[:, 5])
    q7.plot(time, X_sim[:-1, 6])
    q7.plot(time, q_ref[:, 6])
    plt.show()
    # print('final desired end-effector position = ', x[:, -1])
    # print('final actual end-effector position = ', t_ee)
    # print(conf.ee_target_pos)
