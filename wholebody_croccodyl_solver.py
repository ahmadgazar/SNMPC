from re import M
import crocoddyl
import pinocchio
import numpy as np

class WholeBodyDDPSolver:
    # constructor
    def __init__(self, model, centroidalTask=None, forceTask=None, MPC=False, WARM_START=True):
        self.RECEEDING_HORIZON = MPC
        # timing
        self.N_mpc = model.N_mpc
        self.N_traj = model.N
        self.dt = model.dt
        self.dt_ctrl = model.dt_ctrl
        self.N_interpol = int(model.dt/model.dt_ctrl)
        # whole-body croccodyl model
        self.whole_body_model = model    
        # initial condition and warm-start
        self.x0 = model.x0
        # extra tasks
        self.centroidal_task = centroidalTask
        self.force_task = forceTask
        # flags
        self.WARM_START = WARM_START
        self.RECEEDING_HORIZON = MPC
        # initialize ocp and create DDP solver
        self.__add_tracking_tasks(centroidalTask, forceTask)
        self.__init_ocp_and_solver()
        if MPC:
            self.warm_start_mpc(centroidalTask, forceTask)

    def __init_ocp_and_solver(self):
        wbd_model, N_mpc = self.whole_body_model,  self.N_mpc
        if self.RECEEDING_HORIZON:
           self.add_extended_horizon_mpc_models()
        else:
            wbd_model.terminal_model = self.add_terminal_cost()
            ocp = crocoddyl.ShootingProblem(self.x0, 
                                wbd_model.running_models, 
                                wbd_model.terminal_model)
            self.solver = crocoddyl.SolverFDDP(ocp)
            self.solver.setCallbacks([crocoddyl.CallbackLogger(),
                                    crocoddyl.CallbackVerbose()])     
    
    def __add_tracking_tasks(self, centroidalTask, forceTask):
        if self.RECEEDING_HORIZON:
            N = self.N_mpc 
        else:
            N = self.N_traj
        running_models  = self.whole_body_model.running_models[:N]
        # if centroidalTask is not None:
        #     self.add_centroidal_costs(running_models, centroidalTask)
        if forceTask is not None:
            self.add_force_tracking_cost(running_models, forceTask)
    
    def warm_start_mpc(self, centroidalTask, forceTask):
        print('\n'+'=' * 50)
        print('Warm-starting first WBD MPC iteration ..')
        print('-' * 50)
        N_mpc = self.N_mpc
        running_models_N = self.whole_body_model.running_models[:N_mpc]
        # if centroidalTask is not None:
        #     centroidalTask_N = centroidalTask[:N_mpc]
        #     self.add_centroidal_costs(running_models_N, centroidalTask_N)
        if forceTask is not None:
            forceTask_N = forceTask[:N_mpc]
            self.add_force_tracking_cost(running_models_N, forceTask_N)    
        terminal_model = running_models_N[-1]
        ocp = crocoddyl.ShootingProblem(self.x0, running_models_N, terminal_model)
        solver = crocoddyl.SolverFDDP(ocp)
        solver.setCallbacks([crocoddyl.CallbackLogger(),
                          crocoddyl.CallbackVerbose()]) 
        x0 = self.x0
        xs = [x0]*(solver.problem.T + 1)
        us = solver.problem.quasiStatic([x0]*solver.problem.T)
        solver.solve(xs, us)
        self.x_init = solver.xs
        self.u_init = solver.us    
    
    def add_extended_horizon_mpc_models(self):
        N_mpc = self.N_mpc
        for _ in range(N_mpc):
            self.whole_body_model.running_models += [self.add_terminal_cost()]
        if self.force_task is not None:
            forceTaskTerminal_N = np.repeat(
                self.force_task[-1].reshape(1, self.force_task[-1].shape[0]), N_mpc, axis=0
                ) 
            self.add_force_tracking_cost(
                self.whole_body_model.running_models[self.N_mpc:], forceTaskTerminal_N 
            )

    def add_terminal_cost(self):
        wbd_model = self.whole_body_model
        final_contact_sequence = wbd_model.contact_sequence[-1]
        if wbd_model.rmodel.type == 'QUADRUPED':
            supportFeetIds = [wbd_model.lfFootId, wbd_model.rfFootId, 
                              wbd_model.lhFootId, wbd_model.rhFootId]
            feetPos = [final_contact_sequence[1].pose.translation, 
                       final_contact_sequence[0].pose.translation,
                       final_contact_sequence[3].pose.translation, 
                       final_contact_sequence[2].pose.translation]
        elif wbd_model.rmodel.type == 'HUMANOID':
            supportFeetIds = [wbd_model.lfFootId, wbd_model.rfFootId]   
            feetPos = [final_contact_sequence[1].pose.translation, 
                       final_contact_sequence[0].pose.translation]
        swingFootTask = []
        for i, p in zip(supportFeetIds, feetPos):
            swingFootTask += [[i, pinocchio.SE3(np.eye(3), p)]]
        terminalCostModel = wbd_model.createSwingFootModel(
            supportFeetIds, swingFootTask=swingFootTask, comTask=wbd_model.comRef
            )
        return terminalCostModel

    def add_com_task(self, diff_cost, com_des):
        state, nu = self.whole_body_model.state, self.whole_body_model.actuation.nu
        com_residual = crocoddyl.ResidualModelCoMPosition(state, com_des, nu)
        com_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
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
    
    def update_centroidal_reference(self, dam, hg_ref):
        for _, cost in dam.costs.todict().items():
            # update CoM reference
            if isinstance(cost.cost.residual,  
                crocoddyl.libcrocoddyl_pywrap.ResidualModelCentroidalMomentum):
                # print("updating com tracking reference at node ")
                cost.cost.residual.reference = hg_ref
                return True
        return False    
        
    def add_centroidal_momentum_task(self, diff_cost, hg_des):
        wbd_model = self.whole_body_model
        state, nu = wbd_model.state, wbd_model.actuation.nu
        hg_residual = crocoddyl.ResidualModelCentroidalMomentum(state, hg_des, nu)
        hg_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1., 1., 1., 1.]))
        hg_track = crocoddyl.CostModelResidual(state, hg_activation, hg_residual)
        diff_cost.addCost("centroidalTrack", hg_track, wbd_model.task_weights['centroidalTrack'])  

    def add_centroidal_costs(self, iam_N, centroidal_ref_N):                              
        ## update running model
        for centroidal_ref_k, iam_k in zip(centroidal_ref_N, iam_N):
            dam_k = iam_k.differential.costs
            com_ref_k = centroidal_ref_k[:3]
            hg_ref_k = centroidal_ref_k[3:9]
            # update references if cost exists
            FOUND_COM_COST = self.update_com_reference(dam_k, com_ref_k)
            FOUND_HG_COST = self.update_centroidal_reference(dam_k, hg_ref_k)
            # create cost otherwise
            if not FOUND_COM_COST:
                # print("adding com tracking cost at node ", time_idx)
                self.add_com_task(dam_k, com_ref_k)
            if not FOUND_HG_COST:     
                self.add_centroidal_momentum_task(dam_k, hg_ref_k)
    
    def update_force_reference(self, dam, f_ref):
        wbd_model = self.whole_body_model
        rmodel, rdata = wbd_model.rmodel, wbd_model.rdata
        COST_REF_UPDATED = False
        for _, cost in dam.costs.todict().items():
            if isinstance(cost.cost.residual,  
                crocoddyl.libcrocoddyl_pywrap.ResidualModelContactForce):
                frame_idx = cost.cost.residual.id
                # print("updating force tracking reference for contact id ", frame_idx)
                pinocchio.framesForwardKinematics(rmodel, rdata, self.x0[:rmodel.nq])
                if frame_idx == wbd_model.rfFootId:
                    cost.cost.residual.reference = rdata.oMf[frame_idx].actInv(
                                                pinocchio.Force(f_ref[0:3], np.zeros(3))
                                                )
                elif frame_idx == wbd_model.lfFootId:
                    cost.cost.residual.reference = rdata.oMf[frame_idx].actInv(
                                                pinocchio.Force(f_ref[3:6], np.zeros(3))
                                                ) 
                elif frame_idx == wbd_model.rhFootId:
                    cost.cost.residual.reference = rdata.oMf[frame_idx].actInv(
                                                pinocchio.Force(f_ref[6:9], np.zeros(3)) 
                                                )
                elif frame_idx == wbd_model.lhFootId: 
                    cost.cost.residual.reference = rdata.oMf[frame_idx].actInv(
                                                pinocchio.Force(f_ref[9:12], np.zeros(3))
                                                )
                COST_REF_UPDATED = True      
        return COST_REF_UPDATED    

    def add_force_tasks(self, diff_cost, force_des, support_feet_ids):
        wbd_model = self.whole_body_model
        rmodel, rdata = wbd_model.rmodel, wbd_model.rdata 
        pinocchio.framesForwardKinematics(rmodel, rdata, self.x0[:rmodel.nq])
        state, nu = wbd_model.state, wbd_model.actuation.nu        
        forceTrackWeight = wbd_model.task_weights['contactForceTrack']
        if rmodel.foot_type == 'POINT_FOOT':
            nu_contact = 3
            linear_forces = force_des
        elif rmodel.type == 'HUMANOIND' and rmodel.foot_type == 'FLAT_FOOT':
            nu_contact = 6
            linear_forces = np.concatenate([force_des[2:5], force_des[8:11]])
        for frame_idx in support_feet_ids:
            # print("adding force tracking reference for contact id ", frame_idx)
            if frame_idx == wbd_model.rfFootId:
                spatial_force_des = rdata.oMf[frame_idx].actInv(
                    pinocchio.Force(linear_forces[0:3], np.zeros(3))
                    )
            elif frame_idx == wbd_model.lfFootId:
                spatial_force_des = rdata.oMf[frame_idx].actInv(
                    pinocchio.Force(linear_forces[3:6], np.zeros(3))
                    )
            elif frame_idx == wbd_model.rhFootId:
                spatial_force_des = rdata.oMf[frame_idx].actInv(
                    pinocchio.Force(linear_forces[6:9], np.zeros(3))
                    )
            elif frame_idx == wbd_model.lhFootId:
                spatial_force_des = rdata.oMf[frame_idx].actInv(
                    pinocchio.Force(linear_forces[9:12], np.zeros(3))
                    )
            force_activation_weights = np.array([1., 1., 1.])
            force_activation = crocoddyl.ActivationModelWeightedQuad(force_activation_weights)
            force_residual = crocoddyl.ResidualModelContactForce(state, frame_idx, 
                                                spatial_force_des, nu_contact, nu)
            force_track = crocoddyl.CostModelResidual(state, force_activation, force_residual)
            diff_cost.addCost(rmodel.frames[frame_idx].name +"contactForceTrack", 
                                                   force_track, forceTrackWeight)

    def add_force_tracking_cost(self, iam_N, force_ref_N):
        for force_ref_k, iam_k in zip(force_ref_N, iam_N):
            dam_k = iam_k.differential.costs
            support_foot_ids = []
            for _, cost in dam_k.costs.todict().items():
                if isinstance(cost.cost.residual,  
                    crocoddyl.libcrocoddyl_pywrap.ResidualModelContactFrictionCone):
                    support_foot_ids += [cost.cost.residual.id]
            FOUND_FORCE_COST = self.update_force_reference(dam_k, force_ref_k)        
            if not FOUND_FORCE_COST:
                self.add_force_tasks(dam_k, force_ref_k, support_foot_ids)
    
    def solve(self, x_warm_start=False, u_warm_start=False, max_iter=100):
        solver = self.solver
        if x_warm_start and u_warm_start:
            solver.solve(x_warm_start, u_warm_start)
        else:
            x0 = self.x0
            xs = [x0]*(solver.problem.T + 1)
            us = solver.problem.quasiStatic([x0]*solver.problem.T)
            solver.solve(xs, us)    
    
    def update_ocp(self, time_idx, centroidalTask=None, forceTask=None):
        # update models
        N_mpc = self.N_mpc
        running_models_N = self.whole_body_model.running_models[time_idx:time_idx+N_mpc]
        if centroidalTask is not None:
            centroidalTask_N = centroidalTask[time_idx:time_idx+N_mpc]
            self.add_centroidal_costs(running_models_N, centroidalTask)
        if forceTask is not None:
            forceTask_N = forceTask[time_idx:time_idx+N_mpc]
            self.add_force_tracking_cost(running_models_N, forceTask)
        current_terminal_model = running_models_N[-1]
        ocp = crocoddyl.ShootingProblem(self.x0, running_models_N, current_terminal_model)
        self.solver = crocoddyl.SolverFDDP(ocp)
        self.solver.setCallbacks([crocoddyl.CallbackLogger(),
                                crocoddyl.CallbackVerbose()])         

    """
    open-loop MPC: re-solving the OCP from the next predicted state
    """
    def run_OL_MPC(self, centroidalTask=None, forceTask=None):
        N_traj = self.N_traj
        sol = []
        for traj_time_idx in range(N_traj):
            print('\n'+'=' * 50)
            print('MPC Iteration ' + str(traj_time_idx))
            print('-' * 50)
            self.update_ocp(traj_time_idx, centroidalTask, forceTask)
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
   
    def interpolate_one_step(self, q, q_next, qdot, qdot_next, tau, tau_next):
        nq, nv = len(q), len(qdot)
        N_interpol, rmodel = self.N_interpol, self.whole_body_model.rmodel
        x_interpol = np.zeros((N_interpol, nq+nv))
        tau_interpol = np.zeros((N_interpol, len(tau)))
        dtau = (tau_next - tau)/float(N_interpol)
        dqdot = (qdot_next - qdot)/float(N_interpol)
        dt = self.dt_ctrl/self.dt
        for interpol_idx in range(N_interpol):
            tau_interpol[interpol_idx] = tau + interpol_idx*dtau 
            x_interpol[interpol_idx, :nq] = pinocchio.interpolate(rmodel, q, q_next, interpol_idx*dt)
            x_interpol[interpol_idx, nq:] = qdot + interpol_idx*dqdot        
        return x_interpol, tau_interpol

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
        x = []
        for time_idx in range (N):
            q, v = xs[time_idx][:nq], xs[time_idx][nq:]
            pinocchio.framesForwardKinematics(rmodel, rdata, q)
            pinocchio.computeCentroidalMomentum(rmodel, rdata, q, v)
            centroidal_sol += [
                np.concatenate(
                    [pinocchio.centerOfMass(rmodel, rdata, q, v), np.array(rdata.hg)]
                    )
                    ]
            jointPos_sol += [q]
            jointVel_sol += [v]
            x += [xs[time_idx]]
            if time_idx < N-1:
                jointAcc_sol +=  [self.solver.problem.runningDatas[time_idx].xnext[nq::]] 
                jointTorques_sol += [us[time_idx]]
                gains += [K[time_idx]]
        sol = {'x':x, 'centroidal':centroidal_sol, 'jointPos':jointPos_sol, 
                          'jointVel':jointVel_sol, 'jointAcc':jointAcc_sol, 
                            'jointTorques':jointTorques_sol, 'gains':gains}        
        return sol    