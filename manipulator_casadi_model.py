from pinocchio import casadi as cpin
import pinocchio as pin
import casadi as ca
import numpy as np 


class fixedBaseManipulatorCasadiModel:
    # constructor
    def __init__(self, conf, STOCHASTIC_OCP=True):
        self.dt = conf.dt
        self.dt_ctrl = conf.dt_ctrl
        self.N_traj = conf.N_traj 
        self.N_mpc = conf.N_mpc
        self.rmodel = conf.rmodel
        self.rdata = conf.rdata
        self.cmodel = conf.cmodel
        self.cdata  = conf.cdata
        self.ee_frame_name = conf.ee_frame_name
        self.alpha = conf.state_cost_weight
        self.beta  = conf.control_cost_weight
        self.gamma = conf.end_effector_position_weight
        self.IK_cost_weight = conf.IK_cost_weight
        self.IK_Kp_gains = conf.IK_Kp_gains
        self.Q = conf.Q 
        self.R = conf.R
        self.x_limits = conf.state_limits
        self.u_limits = conf.control_limits
        self.x_obs = conf.x_obs
        self.q0 = conf.q0
        self.STOCHASTIC_OCP = STOCHASTIC_OCP
        if STOCHASTIC_OCP:
            self.cov_w_dt = conf.cov_w_dt
            self.epsilon = conf.epsilon

        self.__fill_casadi_fwdDyn_model()
        self.forwardKinematics = self.forward_kinematics(self.ee_frame_name)
        self.jacobian = self.compute_jacobian(self.ee_frame_name)

    def __fill_casadi_fwdDyn_model(self):
        nq = self.cmodel.nq
        dt = self.dt
        # constraint jacobian symbols
        self.J_obs_avoidance = ca.SX.sym('J_obs_avoidance', 3)
        self.q = ca.SX.sym("q", nq, 1)
        self.v = ca.SX.sym("v", nq, 1)
        # state symbols
        x = ca.vertcat(self.q, self.v)
        # controls symbols
        u = ca.SX.sym("tau", self.cmodel.nv, 1)
        self.u = u
        # xdot 
        xdot = ca.vertcat(
            ca.SX.sym("qdot", nq), ca.SX.sym("qddot", nq)
            )
        aba = self.aba()(x, self.u)
        f = ca.vertcat(self.v, aba)
        ## CasADi Model
        # define structs
        self.casadi_fwdDyn_model = ca.types.SimpleNamespace()
        # fill casadi model
        self.casadi_fwdDyn_model.model_name = "fixed_base_forward_dynamics"
        self.casadi_fwdDyn_model.f_impl_expr = xdot - f
        self.casadi_fwdDyn_model.f_expl_expr = f
        self.casadi_fwdDyn_model.initial_state = np.concatenate(
            [self.q0, np.zeros(self.q.shape[0])]
            )
        self.casadi_fwdDyn_model.x = x
        self.casadi_fwdDyn_model.xdot = xdot
        self.casadi_fwdDyn_model.u = u
        self.casadi_fwdDyn_model.z = ca.vertcat([])
        self.casadi_fwdDyn_model.p = self.J_obs_avoidance
        self.casadi_fwdDyn_model.A =  ca.Function(
                                     "A",
                                     [x, u], [ca.jacobian(x + f*dt, x)],
                                     ['x', 'u'], ['J_x']
                                     )
        self.casadi_fwdDyn_model.B =  ca.Function(
                                     "B",
                                     [x, u], [ca.jacobian(x + f*dt, u)],
                                     ['x', 'u'], ['J_u']
                                     )
                                
    def forward_kinematics(self, frame_name):
        cmodel, cdata = self.cmodel, self.cdata
        q = self.q
        cpin.framesForwardKinematics(cmodel, cdata, q)
        fk = ca.Function(
            "fk",
            [q],
            [cdata.oMf[cmodel.getFrameId(frame_name)].rotation,
             cdata.oMf[cmodel.getFrameId(frame_name)].translation],
            ['q'], ['rotation', 'translation'], 
        )
        return fk

    def compute_jacobian(self, frame_name):
        cmodel, cdata = self.cmodel, self.cdata
        q = self.q
        cpin.framesForwardKinematics(cmodel, cdata, q)
        jacobian = ca.Function(
            "jacobian",
            [q],
            [cpin.computeFrameJacobian(
                cmodel, cdata, q, cmodel.getFrameId(frame_name), 
                )],
            ['q'], ['jacobian'], 
        )
        return jacobian
        
    def aba(self):
        cmodel, cdata  = self.cmodel, self.cdata
        q, v, u = self.q, self.v, self.u
        x = ca.vertcat(q,v) 
        return ca.Function(
            "aba",
            [x, u],
            [cpin.aba(cmodel, cdata, q, v, u)],
                {"enable_jacobian":True}
            )
            
    def compute_riccatti_gains(self, A, B):
        Q, R  = self.Q, self.R
        P = np.copy(Q)
        At_P  = A.T @ P
        At_P_B = At_P @ B
        P = (Q + (A.T @ P)) - (
            At_P_B @ ca.solve((R + B.T @ P @ B), At_P_B.T)
            )
        return -ca.solve((R + (B.T @ P @ B)), (B.T @ P @ A))

    def propagate_covariance(self, A, B, K, Sigma):
        AB = np.hstack([A, B])
        Sigma_Kt = Sigma @ K.T 
        Sigma_xu = np.vstack([np.hstack([Sigma      , Sigma_Kt]),
                              np.hstack([Sigma_Kt.T , K@Sigma_Kt])])
        return AB @ Sigma_xu @ AB.T + self._Cov_w    
 