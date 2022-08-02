import crocoddyl
import pinocchio
import numpy as np

class WholeBodyDDPSolver:
    # constructor
    def __init__(self, model, MPC=False, WARM_START=False):
        self.RECEEDING_HORIZON = MPC
        # timing
        self.N_mpc = model.N_mpc
        self.N_traj = model.N
        self.dt = model.dt
        self.dt_ctrl = model.dt_ctrl
        # whole-body croccodyl model
        self.whole_body_model = model    
        # initial condition and warm-start
        self.x0 = model.x0
        # flags
        self.WARM_START = WARM_START
        self.RECEEDING_HORIZON = MPC
        # initialize ocp and create DDP solver
        self.__init_ocp_and_solver()
        if MPC and WARM_START:
            self.__warm_start_mpc()

    def __init_ocp_and_solver(self):
        wbd_model, N_mpc = self.whole_body_model,  self.N_mpc
        if self.RECEEDING_HORIZON:
            for _ in range(N_mpc):
                wbd_model.running_models += [self.add_terminal_cost()]    
        else:
            wbd_model.terminal_model = self.add_terminal_cost()
            ocp = crocoddyl.ShootingProblem(self.x0, 
                                wbd_model.running_models, 
                                wbd_model.terminal_model)
            self.solver = crocoddyl.SolverFDDP(ocp)
            self.solver.setCallbacks([crocoddyl.CallbackLogger(),
                                    crocoddyl.CallbackVerbose()])            
    def __warm_start_mpc(self):
        print('\n'+'=' * 50)
        print('Warm-starting first MPC iteration ..')
        print('-' * 50)
        running_models = self.whole_body_model.running_models[:self.N_mpc]
        terminal_model = running_models[-1]
        ocp = crocoddyl.ShootingProblem(self.x0, running_models,terminal_model)
        solver = crocoddyl.SolverFDDP(ocp)
        solver.setCallbacks([crocoddyl.CallbackLogger(),
                          crocoddyl.CallbackVerbose()]) 
        x0 = self.x0
        xs = [x0]*(solver.problem.T + 1)
        us = solver.problem.quasiStatic([x0]*solver.problem.T)
        solver.solve(xs, us)
        self.x_init = solver.xs
        self.u_init = solver.us    
    
    def add_terminal_cost(self):
        wbd_model = self.whole_body_model
        final_contact_sequence = wbd_model.contact_sequence[-1]
        feetPos = [final_contact_sequence[1].pose.translation, 
                   final_contact_sequence[0].pose.translation,
                   final_contact_sequence[3].pose.translation, 
                   final_contact_sequence[2].pose.translation] 
        if wbd_model.rmodel.name == 'solo':
            supportFeetIds = [wbd_model.lfFootId, wbd_model.rfFootId, 
                              wbd_model.lhFootId, wbd_model.rhFootId]
        elif wbd_model.rmodel.name == 'talos':
            supportFeetIds = [wbd_model.lfFootId, wbd_model.rfFootId]   
        swingFootTask = []
        for i, p in zip(supportFeetIds, feetPos):
            swingFootTask += [[i, pinocchio.SE3(np.eye(3), p)]]
        terminalCostModel = wbd_model.createSwingFootModel(supportFeetIds, 
                    swingFootTask=swingFootTask, comTask=wbd_model.comRef, 
                                       centroidalTask=None,forceTask=None)
        return terminalCostModel

    def add_com_tracking_cost(self, diff_cost, com_des):
        state, nu = self.whole_body_model.state, self.whole_body_model.actuation.nu
        com_residual = crocoddyl.ResidualModelCoMPosition(state, com_des, nu)
        com_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 10.]))
        com_track = crocoddyl.CostModelResidual(state, com_activation, com_residual)
        diff_cost.addCost("comTrack", com_track, self.whole_body_model.task_weights['comTrack'])
    
    def update_com_reference(self, dam, com_ref, TERMINAL=False):
        for _, cost in dam.costs.todict().items():
            # update CoM reference
            if isinstance(cost.cost.residual,  
                crocoddyl.libcrocoddyl_pywrap.ResidualModelCoMPosition):
                # print("updating com tracking reference at node ")
                cost.cost.residual.reference = com_ref
                return True
        return False    

    def add_centroidal_momentum_tracking_cost(self, diff_cost, hg_des):
        wbd_model = self.whole_body_model
        state, nu = wbd_model.state, wbd_model.actuation.nu
        hg_residual = crocoddyl.ResidualModelCentroidalMomentum(state, hg_des, nu)
        hg_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1., 5., 10., 1.]))
        hg_track = crocoddyl.CostModelResidual(state, hg_activation, hg_residual)
        diff_cost.addCost("centroidalTrack", hg_track, wbd_model.task_weights['centroidalTrack'])  

    def add_centroidal_tracking_costs(self, centroidal_ref):
        self.centroidal_ref = centroidal_ref
        if self.RECEEDING_HORIZON:
            centroidal_ref_final = centroidal_ref[:, -1].reshape(centroidal_ref[:, -1].shape[0], 1)
            # append references to the same as the last time-step
            for _ in range(self.N_mpc):
                self.centroidal_ref = np.concatenate([self.centroidal_ref, 
                                            centroidal_ref_final], axis=1)
        else:
            ## update terminal model
            dam_final = self.whole_body_model.terminal_model.differential.costs
            com_ref_final = self.centroidal_ref[:3, -1]
            # update CoM reference
            FOUND_COM_COST = self.update_com_reference(dam_final, com_ref_final)
            # create a CoM terminal tracking cost
            if not FOUND_COM_COST:
                self.add_com_tracking_cost(dam_final, com_ref_final)
            self.add_centroidal_momentum_tracking_cost(dam_final, self.centroidal_ref[3:9, -1])                                
        ## update running model
        for time_idx, iam_k in enumerate(self.whole_body_model.running_models):
            dam_k = iam_k.differential.costs
            com_ref_k = self.centroidal_ref[:3, time_idx]
            hg_ref_k = self.centroidal_ref[3:9, time_idx]
            # update CoM reference
            FOUND_COM_COST = self.update_com_reference(dam_k, com_ref_k)
            # create a CoM tracking cost
            if not FOUND_COM_COST:
                # print("adding com tracking cost at node ", time_idx)
                self.add_com_tracking_cost(dam_k, com_ref_k)
            self.add_centroidal_momentum_tracking_cost(dam_k, hg_ref_k)
    
    def add_force_tasks(self, diff_cost, force_des, support_feet_ids):
        wbd_model = self.whole_body_model
        rmodel, rdata = wbd_model.rmodel, wbd_model.rdata 
        state, nu = wbd_model.state, wbd_model.actuation.nu        
        forceTrackWeight = wbd_model.task_weights['contactForceTrack']
        if rmodel.name == 'solo':
            nu_contact = 3
        elif rmodel.name == 'talos':
            nu_contact = 6
        for frame_idx in support_feet_ids:
            if frame_idx == wbd_model.rfFootId:
                spatial_force_des = rdata.oMf[frame_idx].actInv(
                    pinocchio.Force(force_des[0:3], np.zeros(3))
                    )
            if frame_idx == wbd_model.lfFootId:
                spatial_force_des = rdata.oMf[frame_idx].actInv(
                    pinocchio.Force(force_des[3:6], np.zeros(3))
                    )
            if frame_idx == wbd_model.rhFootId:
                spatial_force_des = rdata.oMf[frame_idx].actInv(
                    pinocchio.Force(force_des[6:9], np.zeros(3))
                    )
            if frame_idx == wbd_model.lhFootId:
                spatial_force_des = rdata.oMf[frame_idx].actInv(
                    pinocchio.Force(force_des[9:12], np.zeros(3))
                    )
            force_activation_weights = np.array([1., 1., 1.])
            force_activation = crocoddyl.ActivationModelWeightedQuad(force_activation_weights)
            force_residual = crocoddyl.ResidualModelContactForce(state, frame_idx, 
                                                spatial_force_des, nu_contact, nu)
            force_track = crocoddyl.CostModelResidual(state, force_activation, force_residual)
            diff_cost.addCost(rmodel.frames[frame_idx].name +"contactForceTrack", 
                                                   force_track, forceTrackWeight)

    def add_force_tracking_cost(self, force_ref):
        wbd_model = self.whole_body_model
        self.force_ref = force_ref
        if self.RECEEDING_HORIZON:
            force_ref_final = force_ref[:, -1].reshape(force_ref[:, -1].shape[0], 1)
            # append references to the same as the last time-step
            for _ in range(self.N_mpc):
                self.force_ref = np.concatenate([self.force_ref, 
                                       force_ref_final], axis=1)
        for time_idx, iam_k in enumerate(wbd_model.running_models):
            force_ref_k = self.force_ref[:, time_idx]
            dam_k = iam_k.differential.costs
            support_foot_ids = []
            for _, cost in dam_k.costs.todict().items():
                if isinstance(cost.cost.residual,  
                    crocoddyl.libcrocoddyl_pywrap.ResidualModelContactFrictionCone):
                    support_foot_ids += [cost.cost.residual.id]
            self.add_force_tasks(dam_k, force_ref_k, support_foot_ids)
    
    def solve(self, x_warm_start=False, u_warm_start=False, max_iter=100):
        solver = self.solver
        if x_warm_start and u_warm_start:
            solver.solve(x_warm_start, u_warm_start, max_iter)
        else:
            x0 = self.x0
            xs = [x0]*(solver.problem.T + 1)
            us = solver.problem.quasiStatic([x0]*solver.problem.T)
            solver.solve(xs, us, max_iter)    
    
    def update_ocp(self, running_models, terminal_model):
        ocp = crocoddyl.ShootingProblem(self.x0, running_models, terminal_model)
        self.solver = crocoddyl.SolverFDDP(ocp)
        self.solver.setCallbacks([crocoddyl.CallbackLogger(),
                                crocoddyl.CallbackVerbose()])         

    """
    open-loop MPC: re-solving the OCP from the next predicted state
    """
    def run_OL_MPC(self):
        running_models = self.whole_body_model.running_models
        N_traj, N_mpc = self.N_traj, self.N_mpc
        # create solution tuples
        sol = []
        for traj_time_idx in range(N_traj):
            # update models
            current_running_models = running_models[traj_time_idx:traj_time_idx+N_mpc]
            current_terminal_model = current_running_models[-1]
            print('\n'+'=' * 50)
            print('MPC Iteration ' + str(traj_time_idx))
            print('-' * 50)
            self.update_ocp(current_running_models, current_terminal_model)
            if traj_time_idx == 0:
                if self.WARM_START:
                    self.solver.solve(self.x_init, self.u_init)
                else:
                    self.solver.solve()    
            else:
                self.solver.solve(xs, us)    
            # warm-start ocp from the previous solution
            x = [self.solver.xs[i] for i in range(len(self.solver.xs))]
            u = [self.solver.us[i] for i in range(len(self.solver.us))]
            us = u[1:] + [u[-1]]    
            xs = x[1:] + [x[-1]]
            # save solution
            sol += [self.get_solution_trajectories()]
            # update initial condition
            self.x0 = self.solver.xs[1]
        return sol 

    def interpolate_solution(self, solution):
        x, tau = solution['centroidal'], solution['jointTorques']
        q, qdot, qddot = solution['jointPos'], solution['jointVel'], solution['jointAcc']
        gains = solution['gains']
        N_inner = int(self.dt/self.dt_ctrl)
        N_outer_u  = tau.shape[0]
        N_outer_x  = x.shape[0]
        tau_interpol = np.zeros((int((N_outer_u-1)*N_inner), tau.shape[1]))
        gains_interpol = np.zeros((int((N_outer_u-1)*N_inner), gains.shape[1], gains.shape[2]))
        q_interpol = np.zeros((int((N_outer_x-1)*N_inner), q.shape[1]))
        qdot_interpol = np.zeros((int((N_outer_x-1)*N_inner), qdot.shape[1]))
        rmodel = self.whole_body_model.rmodel
        # qddot_interpol = np.empty((int((N_outer_x-1)*N_inner), qddot.shape[1]))*nan
        x_interpol = np.zeros((int((N_outer_x-1)*N_inner), x.shape[1]))
        for i in range(N_outer_u-1):
            dtau = (tau[i+1] - tau[i])/float(N_inner)
            #TODO find more elegant way to interpolate LQR gains 
            dgains = (gains[i+1]-gains[i])/float(N_inner)
            for j in range(N_inner):
                k = i*N_inner + j
                tau_interpol[k] = tau[i] + j*dtau
                gains_interpol[k] = gains[i,:,:] #+j*dgains
        for i in range(N_outer_x-1):
            dx = (x[i+1] - x[i])/float(N_inner)
            # dqddot = (qddot[i+1] - qddot[i])/float(N_inner)
            dqdot = (qdot[i+1] - qdot[i])/float(N_inner)
            dq = pinocchio.difference(rmodel,q[i], q[i+1])/float(N_inner)
            for j in range(N_inner):
                k = i*N_inner + j
                x_interpol[k] = x[i] + j*dx
                if j == 0:
                    q_interpol[k] = q[i]
                else:
                    q_interpol[k] = pinocchio.integrate(rmodel, q_interpol[k-1], dq)
                qdot_interpol[k] = qdot[i] + j*dqdot
                # qddot_interpol[k] = qddot_interpol[i] + j*dqddot
        interpol_sol =  {'centroidal':x_interpol, 'jointPos':q_interpol, 
                  'jointVel':qdot_interpol, #'jointAcc': qddot_interpol,
                'jointTorques':tau_interpol, 'gains':gains_interpol}               
        return interpol_sol

    def get_solution_trajectories(self):
        xs, us, K = self.solver.xs, self.solver.us, self.solver.K
        rmodel, rdata = self.whole_body_model.rmodel, self.whole_body_model.rdata
        nq, nv, N = rmodel.nq, rmodel.nv, len(xs) 
        jointPos_sol = []#np.zeros((N, nq))
        jointVel_sol = []#np.zeros((N, nv))
        jointAcc_sol = []#np.zeros((N, nv))
        jointTorques_sol = []#np.zeros((N-1, nv-6))
        centroidal_sol = []#np.zeros((N, 9))
        gains = []#np.zeros((N-1, K[0].shape[0], K[0].shape[1]))
        for time_idx in range (N):
            q, v = xs[time_idx][:nq], xs[time_idx][nq:]
            pinocchio.framesForwardKinematics(rmodel, rdata, q)
            pinocchio.computeCentroidalMomentum(rmodel, rdata, q, v)
            centroidal_sol += [pinocchio.centerOfMass(rmodel, rdata, q, v), np.array(rdata.hg)]
            jointPos_sol += [q]
            jointVel_sol += [v]
            if time_idx < N-1:
                jointAcc_sol +=  [self.solver.problem.runningDatas[time_idx].xnext[nq::]] 
                if len(us[time_idx]) != 0:
                    jointTorques_sol += [us[time_idx]]
                    gains += [K[time_idx]]
        sol = {'centroidal':centroidal_sol, 'jointPos':jointPos_sol, 
               'jointVel':jointVel_sol, 'jointAcc':jointAcc_sol, 
               'jointTorques':jointTorques_sol, 'gains':gains}        
        return sol   




