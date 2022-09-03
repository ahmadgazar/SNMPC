from pinocchio import casadi as cpin
import casadi as ca
import numpy as np 

class fixedBaseManipulatorCasadiModel:
    # constructor
    def __init__(self, conf, STOCHASTIC_OCP=False):
        self.dt = conf.dt
        self.dt_ctrl = conf.dt_ctrl
        self.N_traj = conf.N_traj 
        self.N_mpc = conf.N_mpc
        self.rmodel = conf.rmodel
        self.cmodel = conf.cmodel
        self.cdata  = conf.cdata
        self.ee_frame_name = conf.ee_frame_name
        self.alpha = conf.state_cost_weight
        self.beta  = conf.control_cost_weight
        self.gamma = conf.end_effector_position_weight
        self.x_limits = conf.state_limits
        self.u_limits = conf.control_limits
        self.q0 = conf.q0
        self.__fill_casadi_model()

    def __fill_casadi_model(self):    
        self.q = ca.SX.sym("q", self.cmodel.nq, 1)
        self.v = ca.SX.sym("v", self.cmodel.nv, 1)
        # state
        x = ca.vertcat(self.q, self.v)
        # controls 
        u = ca.SX.sym("tau", self.cmodel.nv, 1)
        # xdot 
        xdot = ca.vertcat(
            ca.SX.sym("qdot", self.cmodel.nq), ca.SX.sym("qddot", self.cmodel.nv)
            )
        f = ca.vertcat(self.v, cpin.aba(self.cmodel, self.cdata, self.q, self.v, u))
        ## CasADi Model
        # define structs
        self.casadi_model = ca.types.SimpleNamespace()
        # fill casadi model
        self.casadi_model.model_name = "fixed_base_manipulator"
        self.casadi_model.f_impl_expr = xdot - f
        self.casadi_model.f_expl_expr = f
        self.casadi_model.initial_state = np.concatenate([self.q0, np.zeros(self.q.shape[0])])
        self.casadi_model.x = x
        self.casadi_model.xdot = xdot
        self.casadi_model.u = u
        self.casadi_model.z = ca.vertcat([])
        self.casadi_model.p = ca.vertcat([])
    
    def forward_kinematics(self, frame_name):
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.q)
        return self.cdata.oMf[self.cmodel.getFrameId(frame_name)].translation 
        
