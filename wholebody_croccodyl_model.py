import numpy as np
import crocoddyl
import pinocchio 

class WholeBodyModel:
    def __init__(self, conf):
        self.dt = conf.dt
        self.dt_ctrl = conf.dt_ctrl
        self.rmodel = conf.rmodel
        self.rdata = conf.rmodel.createData()
        self.ee_frame_names = conf.ee_frame_names 
        self.gait = conf.gait
        self.contact_sequence = conf.contact_sequence
        self.gait_templates = conf.gait_templates 
        self.task_weights = conf.whole_body_task_weights
        self.state_reg_weights = conf.wbd_state_reg_weights
        # Defining the friction coefficient and normal
        self.postImpact = None
        self.mu = conf.mu
        self.N = conf.N
        self.N_mpc = conf.N_mpc_wbd
        self.Rsurf = np.eye(3)
        if conf.rmodel.foot_type == 'FLAT_FOOT':
            self.foot_size = np.array([conf.lxp-conf.lxn,
                                       conf.lyp-conf.lyn])
        self.__initialize_robot(conf.q0)
        self.__set_contact_frame_names_and_indices()
        self.__fill_ocp_models()

    def __fill_ocp_models(self):
        if self.gait['type'] == 'TROT':
            self.create_trot_models()
        elif self.gait['type'] == 'PACE':    
            self.create_pace_models()
        elif self.gait['type'] == 'BOUND':    
            self.create_bound_models()
        elif self.gait['type'] == 'JUMP':    
            self.create_jump_models()

    def __set_contact_frame_names_and_indices(self):
        ee_frame_names = self.ee_frame_names
        rmodel = self.rmodel 
        if self.rmodel.type == 'QUADRUPED':
            self.lfFootId = rmodel.getFrameId(ee_frame_names[0])
            self.rfFootId = rmodel.getFrameId(ee_frame_names[1])
            self.lhFootId = rmodel.getFrameId(ee_frame_names[2])
            self.rhFootId = rmodel.getFrameId(ee_frame_names[3])
        elif rmodel.type == 'HUMANOID':
            self.lfFootId = rmodel.getFrameId(ee_frame_names[0])
            self.rfFootId = rmodel.getFrameId(ee_frame_names[1])

    def __initialize_robot(self, q0):
        self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])
        self.x0 = self.rmodel.defaultState
        # create croccodyl state and controls
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)

    def add_swing_feet_tracking_costs(self, cost, swing_feet_tasks):
        swingFootPosWeight = self.task_weights['swingFoot']['preImpact']['position']
        swingFootVelWeight = self.task_weights['swingFoot']['preImpact']['velocity']
        state, nu = self.state, self.actuation.nu
        for task in swing_feet_tasks:
            if self.rmodel.foot_type == 'POINT_FOOT':
                frame_pose_residual = crocoddyl.ResidualModelFrameTranslation(
                    state, task[0], task[1].translation, nu
                    )
            elif self.rmodel.foot_type == 'FLAT_FOOT':
                frame_pose_residual = crocoddyl.ResidualModelFramePlacement(
                    state, task[0], task[1], nu
                    )
            verticalFootVelResidual = crocoddyl.ResidualModelFrameVelocity(state, task[0],
                    pinocchio.Motion.Zero(), pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED, nu
                    )
            verticalFootVelAct = crocoddyl.ActivationModelWeightedQuad(np.array([0, 0, 1, 0, 0, 0]))
            verticalFootVelCost = crocoddyl.CostModelResidual(state, verticalFootVelAct, 
                                                                verticalFootVelResidual)
            cost.addCost(self.rmodel.frames[task[0]].name+  "__footVelTrack", verticalFootVelCost, 
                                                                                swingFootVelWeight)                               
            foot_track = crocoddyl.CostModelResidual(state, frame_pose_residual)
            cost.addCost(self.rmodel.frames[task[0]].name + "_footPosTrack", foot_track, 
                                                                     swingFootPosWeight)
    
    def add_pseudo_impact_costs(self, cost, swing_feet_tasks):
        state, nu = self.state, self.actuation.nu
        footPosImpactWeight = self.task_weights['swingFoot']['impact']['position']
        footVelImpactweight = self.task_weights['swingFoot']['impact']['velocity']
        for task in swing_feet_tasks:
            frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, 
                                                    task[0], task[1].translation, nu)
            frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(state, task[0], 
                                           pinocchio.Motion.Zero(), pinocchio.LOCAL, nu)                          
            footPosImpactCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)
            footVelImpactCost = crocoddyl.CostModelResidual(state, frameVelocityResidual)
            cost.addCost(self.rmodel.frames[task[0]].name + "_footPosImpact",
                                           footPosImpactCost, footPosImpactWeight)
            cost.addCost(self.rmodel.frames[task[0]].name + "_footVelImpact", 
                                           footVelImpactCost, footVelImpactweight)
        if self.rmodel.foot_type == 'FLAT_FOOT':
            # keep feet horizontal at the time of impact
            for task in swing_feet_tasks:
                footRotImpactWeight = self.task_weights['swingFoot']['impact']['orientation']
                frameRotResidual = crocoddyl.ResidualModelFrameRotation(
                    state, task[0], np.eye(3), nu
                    )
                frameRotAct = crocoddyl.ActivationModelWeightedQuad(np.array([1, 1, 0]))
                footRotImpactCost = crocoddyl.CostModelResidual(
                    state,frameRotAct , frameRotResidual
                    ) 
                cost.addCost(self.rmodel.frames[task[0]].name + "_footRotImpact",
                                            footRotImpactCost, footRotImpactWeight)                                                                                     

    def add_support_contact_costs(self, contact_model, cost, support_feet_ids):
        state, nu = self.state, self.actuation.nu
        rmodel = self.rmodel
        frictionConeWeight = self.task_weights['frictionCone']
        # check if it's a post-impact knot
        if self.postImpact is not None:
            self.add_pseudo_impact_costs(cost, self.postImpact)
        for frame_idx in support_feet_ids:
            R_cone_local = self.rdata.oMf[frame_idx].rotation.T.dot(self.Rsurf)
            if rmodel.foot_type == 'POINT_FOOT': 
                support_contact = crocoddyl.ContactModel3D(
                    state, frame_idx, np.array([0., 0., 0.]), nu, np.array([0., 50.])
                    )
                cone = crocoddyl.FrictionCone(R_cone_local, self.mu, 4, True)
                cone_residual = crocoddyl.ResidualModelContactFrictionCone(
                    state, frame_idx, cone, nu
                    )
            elif rmodel.foot_type == 'FLAT_FOOT':
                # friction cone
                support_contact = crocoddyl.ContactModel6D(
                    state, frame_idx, pinocchio.SE3.Identity(),nu, np.array([0., 50.])
                    )
                cone = crocoddyl.WrenchCone(
                    self.Rsurf, self.mu, np.array([self.foot_size[0], self.foot_size[1]])
                    )
                cone_residual = crocoddyl.ResidualModelContactWrenchCone(
                    state, frame_idx, cone, nu
                    )
                # CoP
                cop_box = crocoddyl.CoPSupport(self.Rsurf, self.foot_size)
                cop_residual = crocoddyl.ResidualModelContactCoPPosition(
                    state, frame_idx, cop_box, nu
                    )
                cop_activation = crocoddyl.ActivationModelQuadraticBarrier(
                    crocoddyl.ActivationBounds(cop_box.lb, cop_box.ub)
                    )
                cop = crocoddyl.CostModelResidual(state, cop_activation, cop_residual)
                cost.addCost(rmodel.frames[frame_idx].name + "_cop", cop, self.task_weights['cop'])
            contact_model.addContact(rmodel.frames[frame_idx].name + "_contact", support_contact) 
            cone_activation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
                )
            friction_cone = crocoddyl.CostModelResidual(state, cone_activation, cone_residual)
            cost.addCost(
                rmodel.frames[frame_idx].name + "_frictionCone", friction_cone, frictionConeWeight
                )
    
    def add_com_position_tracking_cost(self, cost, com_des):    
        com_residual = crocoddyl.ResidualModelCoMPosition(self.state, com_des, self.actuation.nu)
        com_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
        com_track = crocoddyl.CostModelResidual(self.state, com_activation, com_residual)
        cost.addCost("comTrack", com_track, self.task_weights['comTrack'])

    def add_stat_ctrl_reg_costs(self, cost):
        nu = self.actuation.nu 
        stateWeights = self.state_reg_weights
        if self.postImpact is not None:
            state_reg_weight, control_reg_weight = self.task_weights['stateReg']['impact'],\
                                                    self.task_weights['ctrlReg']['impact']
            self.postImpact = None                                        
        else:
            state_reg_weight, control_reg_weight = self.task_weights['stateReg']['stance'],\
                                                    self.task_weights['ctrlReg']['stance']
        state_bounds_weight = self.task_weights['stateBounds']
        # state regularization cost
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        cost.addCost("stateReg", stateReg, state_reg_weight)
        # state bounds cost
        lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        stateBoundsResidual = crocoddyl.ResidualModelState(self.state, nu)
        stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(lb, ub)
            )
        stateBounds = crocoddyl.CostModelResidual(
            self.state, stateBoundsActivation, stateBoundsResidual
            )
        cost.addCost("stateBounds", stateBounds, state_bounds_weight)
        # control regularization cost
        ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        cost.addCost("ctrlReg", ctrlReg, control_reg_weight)                  
        
    def create_trot_models(self):
        # Compute the current foot positions
        x0 = self.rmodel.defaultState
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation 
        self.comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        self.comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2].item()
        self.time_idx = 0
        # Defining the action models along the time instances
        loco3dModel = []
        for i, gait in enumerate(self.gait_templates):
            if self.gait['terrain'] == 'CLIMB' or (self.gait['terrain'] == 'HIKE' and i%2 ==0):
               HIKE_DIRECTION = 'UP'
            elif self.gait['terrain'] == 'DOWN' or (self.gait['terrain'] == 'HIKE' and i%2 !=0):
               HIKE_DIRECTION = 'DOWN'   
            else:
                HIKE_DIRECTION = 'FLAT'
            for phase in gait:
                if phase == 'doubleSupport':
                    loco3dModel += self.createDoubleSupportFootstepModels(
                        [lfFootPos0, rfFootPos0, lhFootPos0, rhFootPos0]
                        )
                elif phase == 'rflhStep':
                    loco3dModel += self.createSingleSupportFootstepModels(
                        [rfFootPos0, lhFootPos0], 
                        [self.lfFootId, self.rhFootId], 
                        [self.rfFootId, self.lhFootId],
                        HIKE_DIRECTION 
                        )
                elif phase == 'lfrhStep':
                    loco3dModel += self.createSingleSupportFootstepModels(
                        [lfFootPos0, rhFootPos0], 
                        [self.rfFootId, self.lhFootId],
                        [self.lfFootId, self.rhFootId],
                        HIKE_DIRECTION
                        )
        self.running_models = loco3dModel
        
    def create_pace_models(self):
        rmodel, rdata = self.rmodel, self.rdata
        robot_type = rmodel.type
        # Compute the current foot positions
        x0 = self.x0
        q0 = x0[:rmodel.nq]
        pinocchio.forwardKinematics(rmodel, rdata, q0)
        pinocchio.updateFramePlacements(rmodel, rdata)
        if robot_type == 'QUADRUPED':
            rfFootPos0 = rdata.oMf[self.rfFootId].translation
            rhFootPos0 = rdata.oMf[self.rhFootId].translation
            lfFootPos0 = rdata.oMf[self.lfFootId].translation
            lhFootPos0 = rdata.oMf[self.lhFootId].translation
            self.comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        elif robot_type == 'HUMANOID':
            # print('yes')
            rfFootPos0 = rdata.oMf[self.rfFootId].translation
            lfFootPos0 = rdata.oMf[self.lfFootId].translation
            self.comRef = (rfFootPos0 + lfFootPos0) / 2    
        self.comRef[2] = pinocchio.centerOfMass(rmodel, rdata, q0)[2].item()
        self.time_idx = 0
        # Defining the action models along the time instances
        loco3dModel = []
        for gait in self.gait_templates:
            for phase in gait:
                if robot_type == 'QUADRUPED' and phase == 'doubleSupport':
                    loco3dModel += self.createDoubleSupportFootstepModels(
                        [lfFootPos0, rfFootPos0, lhFootPos0, rhFootPos0]
                        )
                elif robot_type == 'HUMANOID' and phase == 'doubleSupport':
                    loco3dModel += self.createDoubleSupportFootstepModels([lfFootPos0, rfFootPos0])
                elif phase == 'rfrhStep':
                    loco3dModel += self.createSingleSupportFootstepModels(
                        [rfFootPos0, rhFootPos0], [self.lfFootId, self.lhFootId], [self.rfFootId, self.rhFootId]
                        )
                elif phase == 'rfStep':
                    loco3dModel += self.createSingleSupportFootstepModels(
                        [rfFootPos0], [self.lfFootId], [self.rfFootId]
                        )
                elif phase == 'lflhStep':
                    loco3dModel += self.createSingleSupportFootstepModels(
                        [lfFootPos0, lhFootPos0], [self.rfFootId, self.rhFootId], [self.lfFootId, self.lhFootId]
                        )
                elif phase == 'lfStep':
                    loco3dModel += self.createSingleSupportFootstepModels(
                        [lfFootPos0], [self.rfFootId], [self.lfFootId]
                        )                                                                
        #Rescaling the terminal weights
        costs = loco3dModel[-1].differential.costs.costs.todict()
        for c in costs.values():
            c.weight *= self.dt
        self.running_models = loco3dModel
      
    def create_bound_models(self):
         # Compute the current foot positions
        x0 = self.rmodel.defaultState
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        self.comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        self.comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2].item()
        self.time_idx = 0
        # Defining the action models along the time instances
        loco3dModel = []
        for i, gait in enumerate(self.gait_templates):
            if self.gait['terrain'] == 'CLIMB' or (self.gait['terrain'] == 'HIKE' and i%2 ==0): 
                HIKE_DIRECTION = 'UP'
            elif self.gait['terrain'] == 'DOWN' or (self.gait['terrain'] == 'HIKE' and i%2 !=0):
                HIKE_DIRECTION = 'DOWN'
            else:
                HIKE_DIRECTION = 'FLAT'        
            for phase in gait:
                if phase == 'doubleSupport':
                    loco3dModel += self.createDoubleSupportFootstepModels(
                        [lfFootPos0, rfFootPos0,lhFootPos0, rhFootPos0]
                        )
                elif phase == 'rflfStep':
                    loco3dModel += self.createSingleSupportFootstepModels(
                        [rfFootPos0, lfFootPos0], 
                        [self.rhFootId, self.lhFootId], 
                        [self.rfFootId, self.lfFootId],
                        HIKE_DIRECTION
                        )              
                elif phase == 'rhlhStep':
                    loco3dModel += self.createSingleSupportFootstepModels(
                        [rhFootPos0, lhFootPos0], 
                        [self.rfFootId, self.lfFootId], 
                        [self.rhFootId, self.lhFootId],
                        HIKE_DIRECTION
                        )
        self.running_models = loco3dModel

    def create_jump_models(self):
        # Compute the current foot positions
        x0 = self.rmodel.defaultState
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        self.comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        self.comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2].item()
        self.time_idx = 0
        # Defining the action models along the time instances
        loco3dModel = []
        for i, gait in enumerate(self.gait_templates):
            if self.gait['terrain'] == 'CLIMB' or (self.gait['terrain'] == 'HIKE' and i%2 ==0): 
                HIKE_DIRECTION = 'UP'
            elif self.gait['terrain'] == 'DOWN' or (self.gait['terrain'] == 'HIKE' and i%2 !=0):
                HIKE_DIRECTION = 'DOWN'
            else:
                HIKE_DIRECTION = 'FLAT'       
            for phase in gait:
                if phase == 'doubleSupport':
                    loco3dModel += self.createDoubleSupportFootstepModels(
                        [lfFootPos0, rfFootPos0, lhFootPos0, rhFootPos0]
                        )
                elif phase == 'NONE':
                    loco3dModel += self.createSingleSupportJumpFootstepModels(
                        [lfFootPos0, rfFootPos0, lhFootPos0, rhFootPos0], 
                        [], 
                        [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                        HIKE_DIRECTION
                        )              
        self.running_models = loco3dModel

    def createDoubleSupportFootstepModels(self, feetPos):
        if self.rmodel.type == 'QUADRUPED':
            supportFeetIds = [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId]
        elif self.rmodel.type == 'HUMANOID':
            supportFeetIds = [self.lfFootId, self.rfFootId]
        supportKnots = self.gait['supportKnots']
        doubleSupportModel = []
        for _ in range(supportKnots):
            swingFootTask = []
            for i, p in zip(supportFeetIds, feetPos):
                swingFootTask += [[i, pinocchio.SE3(np.eye(3), p)]]
          
            doubleSupportModel += [
                self.createSwingFootModel(supportFeetIds, swingFootTask=swingFootTask)
                ]               
        return doubleSupportModel

    def createSingleSupportJumpFootstepModels(self, feetPos0, supportFootIds, swingFootIds, HIKE_DIRECTION='FLAT'):
        numLegs = len(supportFootIds) + len(swingFootIds)
        stepLength, stepHeight = self.gait['stepLength'], self.gait['stepHeight']
        StepWidth, numKnots = self.gait['stepWidth'], self.gait['stepKnots']
        phKnots = numKnots/2
        stepHeight = self.gait['stepHeight']
        jumpHeight = self.gait['jumpHeight']
        df = stepHeight + feetPos0[0][2]
        # Action models for the foot swing
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            # swing feet tasks 
            for i, p in zip(swingFootIds, feetPos0):
                if HIKE_DIRECTION == 'UP':
                    phKnots = numKnots/2
                    # if k < phKnots:
                    #     dp = np.array(
                    #         [stepLength*(k+1)/numKnots, 0., 1.8*stepHeight*k/phKnots]
                    #         )
                    #     tref = p + dp
                    #     swingFootTask += [[i, pinocchio.SE3(np.eye(3), tref)]]
                    if k == phKnots:
                        dp = np.array(
                            [stepLength*(k+1)/numKnots, 0., 2.5*stepHeight]
                            )
                        tref = p + dp
                        swingFootTask += [[i, pinocchio.SE3(np.eye(3), tref)]]
                    # else:
                    #     dp = np.array(
                    #         [stepLength*(k+1)/numKnots, 0., 1.5*stepHeight*(1-float(k-phKnots)/phKnots)]
                    #         )
                    # swing foot at landing
                    if k == numKnots-1:
                        dp = np.array([stepLength, 0., stepHeight])  
                        tref = p + dp
                        swingFootTask += [[i, pinocchio.SE3(np.eye(3), tref)]]
                    # else:
                    #     swingFootTask = None        
            if k == numKnots-1:
                self.postImpact = swingFootTask
            else:
                self.postImpact = None
            # CoM task
            if HIKE_DIRECTION == 'UP':
                # flying
                if k < phKnots:
                    comTask = np.array(
                        [stepLength, 0., stepHeight+jumpHeight]
                        )*(k+1)/numKnots + self.comRef
                # at landing instance
                elif k == numKnots-1:
                    comTask = np.array(
                        [stepLength, 0., stepHeight]
                        ) + self.comRef
                # else:
                #     comTask = None
            footSwingModel += [
                self.createSwingFootModel(
                    supportFootIds, comTask=comTask, swingFootTask=swingFootTask
                )
                    ]
        # Updating the feet positions and CoM for next step
        if HIKE_DIRECTION == 'UP':
            self.comRef += [stepLength, 0., stepHeight]
            for p in feetPos0:
                p += [stepLength, 0., stepHeight]
        return footSwingModel 
            
    def createSingleSupportFootstepModels(self, feetPos0, supportFootIds, swingFootIds, HIKE_DIRECTION='FLAT'):
        numLegs = len(supportFootIds) + len(swingFootIds)
        comPercentage = float(len(swingFootIds)) / numLegs
        stepLength, stepHeight = self.gait['stepLength'], self.gait['stepHeight']
        StepWidth, numKnots = self.gait['stepWidth'], self.gait['stepKnots']
        if self.gait['type'] == 'JUMP':
            jumpHeight = self.gait['jumpHeight']
            df = jumpHeight - feetPos0[0][2]
        else:
            jumpHeight = 0.     
        # Action models for the foot swing
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            # swing feet tasks 
            for i, p in zip(swingFootIds, feetPos0):
                # define swing foot task based on the direction of the stepping
                if HIKE_DIRECTION == 'UP':
                    phKnots = numKnots - 5
                    if k < phKnots:
                        dp = np.array(
                            [stepLength*(k+1)/numKnots, 0., 1.8*stepHeight*k/phKnots]
                            )
                    elif k == phKnots:
                        dp = np.array(
                            [stepLength*(k+1)/numKnots, 0., 1.8*stepHeight]
                            )
                    else:
                        dp = np.array(
                            [stepLength*(k+1)/numKnots, 0., 1.8*stepHeight*(1-float(k-phKnots)/phKnots)]
                            )
                   
                elif HIKE_DIRECTION == 'FLAT':
                    phKnots = numKnots/2
                    if k < phKnots:
                        dp = np.array(
                            [stepLength*(k+1)/numKnots, 0., stepHeight*k / phKnots]
                            )
                    elif k == phKnots:
                        dp = np.array(
                            [stepLength*(k+1)/numKnots, 0., stepHeight]
                            )
                    else:
                        dp = np.array(
                            [stepLength*(k+1)/numKnots, 0., stepHeight*(1-float(k-phKnots)/phKnots)]
                            )
                elif HIKE_DIRECTION == 'DOWN':
                    phKnots = numKnots - 10
                    if k < phKnots:
                        dp = np.array(
                            [stepLength*(k+1)/numKnots, StepWidth*(k+1)/numKnots, 0.5*stepHeight*k / phKnots]
                            )
                    elif k == phKnots:
                        dp = np.array(
                            [stepLength*(k+1)/numKnots, StepWidth*(k+1)/numKnots, 0.5*stepHeight]
                            )
                    else:
                        dp = np.array(
                            [stepLength*(k+1)/numKnots, StepWidth*(k+1)/numKnots, 0.5*stepHeight*(1-float(k-phKnots) / phKnots)]
                            )
                tref = p + dp
                swingFootTask += [[i, pinocchio.SE3(np.eye(3), tref)]]  
            if k == numKnots-1:
                self.postImpact = swingFootTask
            else:
                self.postImpact = None
            # CoM task 
            if HIKE_DIRECTION == 'UP': 
                comTask = np.array(
                    [stepLength*(k+1)/numKnots, 0., stepHeight*(k+1)/numKnots]
                    )*comPercentage + self.comRef
            elif HIKE_DIRECTION == 'FLAT':    
                comTask = np.array(
                    [stepLength*(k+1)/numKnots, 0., 0.]
                    )*comPercentage + self.comRef
            elif HIKE_DIRECTION == 'DOWN':
                comTask = np.array(
                    [stepLength*(k+1)/numKnots, StepWidth*(k+1)/numKnots, -stepHeight*(k+1)/numKnots]
                    )*comPercentage + self.comRef
            footSwingModel += [
                self.createSwingFootModel(supportFootIds, comTask=comTask, swingFootTask=swingFootTask)
                    ]
        # Updating the feet positions and CoM for next step
        if HIKE_DIRECTION == 'UP':
            self.comRef += [stepLength*comPercentage, 0., 0.5*stepHeight]
            for p in feetPos0:
                p += [stepLength, 0., stepHeight] 
        elif HIKE_DIRECTION == 'FLAT':
            self.comRef += [stepLength*comPercentage, 0., 0.]
            for p in feetPos0:
                p += [stepLength, 0., 0.]
        elif HIKE_DIRECTION == 'DOWN':
            self.comRef += [stepLength*comPercentage, StepWidth*comPercentage, -0.5*stepHeight]
            for p in feetPos0:
                p += [stepLength, StepWidth, -stepHeight]                
        return footSwingModel 

    def createSwingFootModel(self, supportFootIds, comTask=None, swingFootTask=None):
        # Creating a multi-contact model
        nu = self.actuation.nu
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        if isinstance(comTask, np.ndarray):
            self.add_com_position_tracking_cost(costModel, comTask)
        if swingFootTask is not None:
            self.add_swing_feet_tracking_costs(costModel, swingFootTask)
        self.add_support_contact_costs(contactModel, costModel, supportFootIds)
        self.add_stat_ctrl_reg_costs(costModel)
        # Creating the action model for the KKT dynamics with simpletic Euler integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actuation, contactModel, costModel, 0., True
            )
        model = crocoddyl.IntegratedActionModelEuler(dmodel, self.dt)
        return model

    def get_solution_trajectories(self, solver):
        xs, us, K = solver.xs, solver.us, solver.K
        rmodel, rdata = self.rmodel, self.rdata
        nq, nv, N = rmodel.nq, rmodel.nv, len(xs) 
        jointPos_sol = np.zeros((N, nq))
        jointVel_sol = np.zeros((N, nv))
        jointAcc_sol = np.zeros((N, nv))
        jointTorques_sol = np.zeros((N-1, nv-6))
        centroidal_sol = np.zeros((N, 9))
        gains = np.zeros((N-1, K[0].shape[0], K[0].shape[1]))
        for time_idx in range (N):
            q, v = xs[time_idx][:nq], xs[time_idx][nq:]
            pinocchio.framesForwardKinematics(rmodel, rdata, q)
            pinocchio.computeCentroidalMomentum(rmodel, rdata, q, v)
            centroidal_sol[time_idx, :3] = pinocchio.centerOfMass(rmodel, rdata, q, v)
            centroidal_sol[time_idx, 3:9] = np.array(rdata.hg)
            jointPos_sol[time_idx, :] = q
            jointVel_sol[time_idx, :] = v
            if time_idx < N-1:
                jointAcc_sol[time_idx+1, :]= solver.problem.runningDatas[time_idx].xnext[nq::] 
                jointTorques_sol[time_idx, :] = us[time_idx]
                gains[time_idx, :,:] = K[time_idx]
        sol = {'centroidal':centroidal_sol, 'jointPos':jointPos_sol, 
               'jointVel':jointVel_sol, 'jointAcc':jointAcc_sol, 
               'jointTorques':jointTorques_sol, 'gains':gains}        
        return sol    

    def get_contact_positions_and_forces_solution(self, solver):
        contact_names = self.ee_frame_names
        contact_forces = np.zeros((len(solver.xs[:-1]), 3*len(contact_names)))
        contact_positions = np.zeros((len(solver.xs[:-1]), 3*len(contact_names)))
        contact_jacobians = []
        for i, d in enumerate(solver.problem.runningDatas):
            contact_jacobians_k = []
            pinocchio.framesForwardKinematics(
                self.rmodel, self.rdata, solver.xs[i][:self.rmodel.nq])
            m = solver.problem.runningModels[i]
            for k, c_key in enumerate(contact_names):
                c_id = self.rmodel.getFrameId(c_key)
                omf = self.rdata.oMf[c_id]
                contact_positions[i, 3*k:3*k+3] = np.resize(omf.translation, 3)
                try:
                    c_data = d.differential.multibody.contacts.contacts[c_key+'_contact']
                    contact_forces[i, 3*k:3*k+3] = np.resize(omf.act(c_data.jMf.actInv(c_data.f)).linear, 3)
                    contact_jacobians_k += [d.differential.multibody.contacts.Jc]
                except:
                    pass
            contact_jacobians += [contact_jacobians_k]    
        return contact_positions, contact_forces, contact_jacobians

    def interpolate_whole_body_solution(self, solution, centroidal_gains=None):
        x, tau = solution['centroidal'], solution['jointTorques']
        q, qdot, qddot = solution['jointPos'], solution['jointVel'], solution['jointAcc']
        if centroidal_gains is None:
            gains = solution['gains']
        else:
            gains = centroidal_gains
        N_inner = int(self.dt/self.dt_ctrl)
        N_outer_u  = tau.shape[0]
        N_outer_x  = x.shape[0]
        tau_interpol = np.zeros((int((N_outer_u-1)*N_inner), tau.shape[1]))
        gains_interpol = np.zeros((int((N_outer_u-1)*N_inner), gains.shape[1], gains.shape[2]))
        q_interpol = np.zeros((int((N_outer_x-1)*N_inner), q.shape[1]))
        qdot_interpol = np.zeros((int((N_outer_x-1)*N_inner), qdot.shape[1]))
        # qddot_interpol = np.empty((int((N_outer_x-1)*N_inner), qddot.shape[1]))*nan
        x_interpol = np.zeros((int((N_outer_x-1)*N_inner), x.shape[1]))
        for i in range(N_outer_u-1):
            dtau = (tau[i+1] - tau[i])/float(N_inner)
            #TODO find more elegant way to interpolate LQR gains 
            #for now they are scaled inside the simulation loop 
            for j in range(N_inner):
                k = i*N_inner + j
                tau_interpol[k] = tau[i] + j*dtau
                gains_interpol[k] = gains[i,:,:]*(self.dt_ctrl/N_inner)
        for i in range(N_outer_x-1):
            dx = (x[i+1] - x[i])/float(N_inner)
            # dqddot = (qddot[i+1] - qddot[i])/float(N_inner)
            dqdot = (qdot[i+1] - qdot[i])/float(N_inner)
            dq = pinocchio.difference(self.rmodel,q[i], q[i+1])/float(N_inner)
            for j in range(N_inner):
                k = i*N_inner + j
                x_interpol[k] = x[i] + j*dx
                if j == 0:
                    q_interpol[k] = q[i]
                else:
                    q_interpol[k] = pinocchio.integrate(self.rmodel, q_interpol[k-1], dq)
                qdot_interpol[k] = qdot[i] + j*dqdot
                # qddot_interpol[k] = qddot_interpol[i] + j*dqddot
        interpol_sol =  {'centroidal':x_interpol, 'jointPos':q_interpol, 
                  'jointVel':qdot_interpol, #'jointAcc': qddot_interpol,
                'jointTorques':tau_interpol, 'gains':gains_interpol}               
        return interpol_sol

    # save solution in dat files for real robot experiments
    def save_solution_dat(self, solution):
        dt_ctrl = self.dt_ctrl
        q, qdot, tau = solution['jointPos'], solution['jointVel'], solution['jointTorques']
        time_x = np.arange(0, np.round(q.shape[0]*dt_ctrl, 2), dt_ctrl)
        time_u = np.arange(0, np.round(tau.shape[0]*dt_ctrl, 2), dt_ctrl)
        q_dat = np.column_stack((np.array([time_x, q[:, 0], q[:, 1], q[:, 2],q[:, 3],q[:, 4],q[:, 5],q[:, 6], q[:, 7],q[:, 8], q[:, 9],q[:, 10],q[:, 11]])))
        qdot_dat = np.column_stack((np.array([time_x, qdot[:, 0], qdot[:, 1], qdot[:, 2], qdot[:, 3], qdot[:, 4],qdot[:, 5],qdot[:, 6], qdot[:, 7],qdot[:, 8], qdot[:, 9],qdot[:, 10],q[:, 11]])))
        tau_dat = np.column_stack((np.array([time_u, tau[:, 0], tau[:, 1], tau[:, 2],tau[:, 3], tau[:, 4], tau[:, 5],tau[:, 6], tau[:, 7],tau[:, 8], tau[:, 9],tau[:, 10],tau[:, 11]])))
        np.savetxt('quadruped_positions.dat', q_dat, fmt=['%.8e','%.8e', '%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e'])
        np.savetxt('quadruped_velocities.dat', qdot_dat, fmt=['%.8e','%.8e', '%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e'])
        np.savetxt('quadruped_feedforward_torque.dat', tau_dat, fmt=['%.8e','%.8e', '%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e'])

def plotSolution(solver, bounds=False, figIndex=1, figTitle="", show=True):
    import matplotlib.pyplot as plt
    xs, us = [], []
    if bounds:
        us_lb, us_ub = [], []
        xs_lb, xs_ub = [], []
    if isinstance(solver, list):
        rmodel = solver[0].problem.runningModels[0].state.pinocchio
        for s in solver:
            xs.extend(s.xs[:-1])
            us.extend(s.us)
            if bounds:
                models = s.problem.runningModels.tolist() + [s.problem.terminalModel]
                for m in models:
                    us_lb += [m.u_lb]
                    us_ub += [m.u_ub]
                    xs_lb += [m.state.lb]
                    xs_ub += [m.state.ub]
    else:
        rmodel = solver.problem.runningModels[0].state.pinocchio
        xs, us = solver.xs, solver.us
        if bounds:
            models = solver.problem.runningModels.tolist() + [solver.problem.terminalModel]
            for m in models:
                us_lb += [m.u_lb]
                us_ub += [m.u_ub]
                xs_lb += [m.state.lb]
                xs_ub += [m.state.ub]

    # Getting the state and control trajectories
    nx, nq, nu = xs[0].shape[0], rmodel.nq, us[0].shape[0]
    X = [0.] * nx
    U = [0.] * nu
    if bounds:
        U_LB = [0.] * nu
        U_UB = [0.] * nu
        X_LB = [0.] * nx
        X_UB = [0.] * nx
    for i in range(nx):
        X[i] = [np.asscalar(x[i]) for x in xs]
        if bounds:
            X_LB[i] = [np.asscalar(x[i]) for x in xs_lb]
            X_UB[i] = [np.asscalar(x[i]) for x in xs_ub]
    for i in range(nu):
        U[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else 0 for u in us]
        if bounds:
            U_LB[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else np.nan for u in us_lb]
            U_UB[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else np.nan for u in us_ub]

    # Plotting the joint positions, velocities and torques
    plt.figure(figIndex)
    plt.suptitle(figTitle)
    legJointNames = ['HAA', 'HFE', 'KFE']
    # LF foot
    plt.subplot(4, 3, 1)
    plt.title('joint position [rad]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(7, 10))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(7, 10))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(7, 10))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(4, 3, 2)
    plt.title('joint velocity [rad/s]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 6, nq + 9))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 6, nq + 9))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 6, nq + 9))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(4, 3, 3)
    plt.title('joint torque [Nm]')
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(0, 3))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(0, 3))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(0, 3))]
    plt.ylabel('LF')
    plt.legend()

    # LH foot
    plt.subplot(4, 3, 4)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(10, 13))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(10, 13))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(10, 13))]
    plt.ylabel('LH')
    plt.legend()
    plt.subplot(4, 3, 5)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 9, nq + 12))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 9, nq + 12))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 9, nq + 12))]
    plt.ylabel('LH')
    plt.legend()
    plt.subplot(4, 3, 6)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(3, 6))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(3, 6))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(3, 6))]
    plt.ylabel('LH')
    plt.legend()

    # RF foot
    plt.subplot(4, 3, 7)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(13, 16))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(13, 16))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(13, 16))]
    plt.ylabel('RF')
    plt.legend()
    plt.subplot(4, 3, 8)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 12, nq + 15))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 12, nq + 15))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 12, nq + 15))]
    plt.ylabel('RF')
    plt.legend()
    plt.subplot(4, 3, 9)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(6, 9))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(6, 9))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(6, 9))]
    plt.ylabel('RF')
    plt.legend()

    # RH foot
    plt.subplot(4, 3, 10)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(16, 19))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(16, 19))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(16, 19))]
    plt.ylabel('RH')
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(4, 3, 11)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 15, nq + 18))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 15, nq + 18))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 15, nq + 18))]
    plt.ylabel('RH')
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(4, 3, 12)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(9, 12))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(9, 12))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(9, 12))]
    plt.ylabel('RH')
    plt.legend()
    plt.xlabel('knots')

    plt.figure(figIndex + 1)
    plt.suptitle(figTitle)
    rdata = rmodel.createData()
    Cx = []
    Cy = []
    for x in xs:
        q = x[:nq]
        c = pinocchio.centerOfMass(rmodel, rdata, q)
        Cx.append(np.asscalar(c[0]))
        Cy.append(np.asscalar(c[1]))
    plt.plot(Cx, Cy)
    plt.title('CoM position')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    if show:
        plt.show()

