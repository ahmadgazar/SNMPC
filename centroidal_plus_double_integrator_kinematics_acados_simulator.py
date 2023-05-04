from acados_template import AcadosModel, AcadosSim, AcadosSimSolver
from robot_properties_solo.solo12wrapper import Solo12Config
import pinocchio as pin
from casadi import *
import numpy as np
import utils


class CentroidalPlusLegKinematicsAcadosSimulator:
    # constructor
    def __init__(self, model, solver):
        # fill model
        self.__fill_acados_disturbance_model(model, solver)
        # create simulator
        acados_simulator = AcadosSim()
        acados_simulator.model = self.acados_model
        acados_simulator.parameter_values = np.zeros(
            model.casadi_model.p.shape[0] + model.casadi_model.x.shape[0]
        )
        acados_simulator.solver_options.Tsim = model._dt
        # create integrator
        self.acados_integrator = AcadosSimSolver(
            acados_simulator, build=True, generate=True
            )
        # additive disturbance parameters
        self.Cov_w = model._Cov_w/model._dt # discretize later the disturbed model   
        self.dt = model._dt
        self.dt_ctrl = 0.001
        self.N = model._N 
        self.N_mpc = model._N_mpc
        # contact data
        self.contact_data = model._contact_data
        # box plus
        self.q_plus = model.casadi_model.q_plus
        self.debris_half_size = 0.5*model._step_bound
        self.contact_sequence = model._contact_sequence
        self.pin_model = model._rmodel
        self.ee_frame_names = model._ee_frame_names
        self.gait = model._gait
        self.STOCHASTIC_OCP = model._STOCHASTIC_OCP

    def __fill_acados_disturbance_model(self, model, solver):
        self.x_init = solver.x_init
        acados_model = AcadosModel()
        additive_disturbance = MX.sym('w', model.casadi_model.x.shape[0], 1) 
        f_disturbed =  model.casadi_model.f_expl_expr + additive_disturbance
        acados_model.f_impl_expr = model.casadi_model.xdot - f_disturbed 
        acados_model.f_expl_expr = f_disturbed
        acados_model.name = model.casadi_model.model_name
        acados_model.xdot = model.casadi_model.xdot
        acados_model.x = model.casadi_model.x
        acados_model.u = model.casadi_model.u
        acados_model.z = model.casadi_model.z
        acados_model.p = vertcat(model.casadi_model.p, additive_disturbance)
        self.acados_model = acados_model   

    def sample_addtitive_uncertainties(self):
        """
        draws uniform(?) sample from ellipsoid with center w and variability matrix Z
        """
        n = self.acados_model.x.shape[0]          
        mu = np.zeros(n)
        lam, v = np.linalg.eig(self.Cov_w)
        # sample in hypersphere
        r = np.random.rand()**(1/n)   # radial position of sample
        x = np.random.randn(n)
        x = x / np.linalg.norm(x)
        x *= r
        return v @ (np.sqrt(lam)*x) + mu
            
    def count_constraint_violations(self, x_sim, WITH_VISUALIZATION=False):
        nb_contact_location_constraint_violations = [0, 0, 0, 0]
        contacts_position_N = self.contact_data['contacts_position'] 
        contacts_logic_N = self.contact_data['contacts_logic']
        rmodel = self.pin_model
        rdata = rmodel.createData()
        x_ref_N = self.x_init
        simulated_ee_positions = []
        if WITH_VISUALIZATION:
            # visualize motion in meshcat
            dt = self.dt
            dt_ctrl = self.dt_ctrl
            N_ctrl =  int(dt/dt_ctrl)
            # create robot
            robot = Solo12Config.buildRobotWrapper()
            # load robot in meshcat viewer
            viz = pin.visualize.MeshcatVisualizer(
            robot.model, robot.collision_model, robot.visual_model)
            try:
                viz.initViewer(open=True)
            except ImportError as err:
                print(err)
                sys.exit(0)
            viz.loadViewerModel()
            # add nominal contact surfaces
            s = self.debris_half_size
            for i, contacts in enumerate(self.contact_sequence):
                for contact_idx, contact in enumerate(contacts):
                    if contact.ACTIVE:
                        t = contact.pose.translation
                        # debris box
                        if contact.CONTACT == 'FR' or contact.CONTACT == 'FL':
                            utils.addViewerBox(
                                viz, 'world/debris'+str(i)+str(contact_idx), 
                                2*s, 2*s, 0., [1., .2, .2, .5]
                                )
                        if contact.CONTACT == 'HR' or contact.CONTACT == 'HL':
                            utils.addViewerBox(
                                viz, 'world/debris'+str(i)+str(contact_idx),
                                2*s, 2*s, 0., [.2, .2, 1., .5]
                                )
                        utils.applyViewerConfiguration(
                            viz, 'world/debris'+str(i)+str(contact_idx), 
                            [t[0], t[1], t[2]-0.017, 1, 0, 0, 0]
                            )
                        utils.applyViewerConfiguration(
                            viz, 'world/debris_center'+str(i)+str(contact_idx), 
                            [t[0], t[1], t[2]-0.017, 1, 0, 0, 0]
                            )
        ee_positions_total = []                          
        # check simulated trajectories
        for sim in range(x_sim.shape[0]):
            print("starting monte-carlo simulation nb: ", sim, ".....")
            VIOLATED = False
            ee_positions_sim = []
            for time_idx in range(x_sim.shape[1]-1):
                ee_positions_k = []
                q_sim = np.concatenate(
                            [
                            x_sim[sim, time_idx, 9:12], 
                            np.array(
                                self.q_plus(x_ref_N[time_idx][12:16], x_sim[sim, time_idx, 12:15]*0.01)
                            ).squeeze(),
                            x_sim[sim, time_idx, 15:27]
                            ]
                        )
                # visualize simulated trajectories 
                if WITH_VISUALIZATION:
                    for _ in range(N_ctrl): 
                        viz.display(q_sim)              
                pin.framesForwardKinematics(rmodel, rdata, q_sim)
                # print("contact positions at time ", time_idx, ": ")
                for contact_idx in range(4):
                    contact_location = rdata.oMf[rmodel.getFrameId(self.ee_frame_names[contact_idx])].translation
                    ee_positions_k += [[contact_location[0], contact_location[1], contact_location[2]]]
                    # print(self.ee_frame_names[contact_idx]+"_ee = ", contact_location)
                    if WITH_VISUALIZATION:
                        utils.addViewerBox(
                            viz, 'world/'+ self.ee_frame_names[contact_idx]+"_ee"+str(sim)+str(time_idx),
                                        0.0015, 0.0015, 0.0015, [.2, 1, .2, .5]
                                        )
                        utils.applyViewerConfiguration(
                            viz,'world/'+ self.ee_frame_names[contact_idx]+"_ee"+str(sim)+str(time_idx), 
                                    [contact_location[0], contact_location[1], contact_location[2], 1, 0, 0, 0]
                                    )  
                    # the next is needed because the left and right contact 
                    # reference indices are swaped w.r.t. the optimizers ordering
                    # FL
                    if contact_idx==0:
                        IN_CONTACT = contacts_logic_N[time_idx][1]
                        debris_pos = contacts_position_N[time_idx][3:6]
                    # FR
                    elif contact_idx==1:
                        IN_CONTACT = contacts_logic_N[time_idx][0]
                        debris_pos = contacts_position_N[time_idx][0:3]
                    # HL
                    elif contact_idx==2:
                        IN_CONTACT = contacts_logic_N[time_idx][3]
                        debris_pos = contacts_position_N[time_idx][9:12]
                    # HR
                    elif contact_idx==3:
                        IN_CONTACT = contacts_logic_N[time_idx][2]
                        debris_pos = contacts_position_N[time_idx][6:9]
                    # check wether contact location constraints are violated for the current feet in contact     
                    if IN_CONTACT:
                        # print(self.ee_frame_names[contact_idx]+" contact reference location", debris_pos[:2])
                        # print("minimum allowed contact location  = ", debris_pos[:2] - self.debris_half_size)
                        # print(self.ee_frame_names[contact_idx]+" simulated contact location  = ", contact_location[:2])
                        # print("maximum allowed contact location  = ", debris_pos[:2] + self.debris_half_size)
                        if (contact_location[0] >= debris_pos[0] + self.debris_half_size) or (contact_location[0] <= debris_pos[0] - self.debris_half_size) or \
                           (contact_location[1] >= debris_pos[1] + self.debris_half_size) or (contact_location[1] <= debris_pos[1] - self.debris_half_size):
                            print("oh oh .. ", self.ee_frame_names[contact_idx]+"_foot", "is outside assigned debris at time knot ", time_idx)   
                            nb_contact_location_constraint_violations[contact_idx] += 1
                            # add contact position of the previous time step for the rest of non-violated contacts
                            counter = contact_idx + 1
                            while counter < 4:
                                ee_positions_k += [ee_positions_sim[time_idx-1][counter]]
                                counter += 1
                            VIOLATED = True
                            break
                    if VIOLATED:
                        break 
                if VIOLATED:
                    break
                # else:
                ee_positions_sim += [ee_positions_k]
            ee_positions_total += [ee_positions_sim]
            # print("total contact positions per simulation: ", ee_positions_total)
        return nb_contact_location_constraint_violations, ee_positions_total


    def simulate(self, x0, u, K, w_total, nb_sims=1, MPC=False, WITH_DISTURBANCES=False):
        nx, N = x0.shape[0], u.shape[0]
        x_sim = np.zeros((nb_sims, N+1, nx))
        x_sim[:, 0, :] = np.copy(x0)
        # nominal contact data
        contacts_logic_N = self.contact_data['contacts_logic']
        contacts_position_N = self.contact_data['contacts_position'] 
        contacts_norms_N = self.contact_data['contacts_orient']
        # get integrator
        acados_integrator = self.acados_integrator
        x_ref_N = self.x_init
        w = np.zeros(nx)
        SAMPLE_DISTURBANCE = True
        for sim in range(nb_sims):
            # start without disturbance on the initial state
            x0 = np.copy(x_sim[0, 0, :])
            x_cl = np.copy(x_sim[0, 0, :])
            for time_idx in range(N-1):
                # apply disturbances just before contact landing
                if SAMPLE_DISTURBANCE:
                    w = w_total[sim, time_idx]
                    SAMPLE_DISTURBANCE = False
                # don't add disturbances on the following
                w[9:12]  = np.zeros(3)                                              #base pos. 
                w[12:15] = np.zeros(3)                                              #base orientation
                w[29]    = 0.                                                       #base lin. vel. z-irection
                w[30:33] = np.zeros(3)                                              #base ang. vel.
                # contact loop
                for contact_idx in range(4):
                    w[33+2+(contact_idx*3)] = 0.                                    # knee joint velocity
                    # the next is needed because the left and right contact 
                    # reference indices are swaped from the optimizers ordering
                    # FL
                    if contact_idx==0:
                        CURR_IN_CONTACT = contacts_logic_N[time_idx][1]
                        NEXT_IN_CONTACT = contacts_logic_N[time_idx+1][1]
                    # FR
                    elif contact_idx==1:
                        CURR_IN_CONTACT = contacts_logic_N[time_idx][0]
                        NEXT_IN_CONTACT = contacts_logic_N[time_idx+1][0]
                    # HL
                    elif contact_idx==2:
                        CURR_IN_CONTACT = contacts_logic_N[time_idx][3]
                        NEXT_IN_CONTACT = contacts_logic_N[time_idx+1][3]
                    # HR
                    elif contact_idx==3:
                        CURR_IN_CONTACT = contacts_logic_N[time_idx][2]
                        NEXT_IN_CONTACT = contacts_logic_N[time_idx+1][2]
                    # landing (add disturbances on the base velocity to simulate impact dynamics)
                    if not CURR_IN_CONTACT and NEXT_IN_CONTACT:
                        w[27:30] = np.zeros(3)                                       # base lin. vel. 
                        w[33+(contact_idx*3):33+3+(contact_idx*3)] = np.zeros(3)     # joint vel.
                    # stance (don't add disturbances)
                    elif CURR_IN_CONTACT and NEXT_IN_CONTACT:
                        w[9:12]  = np.zeros(3)                                       # base pos. 
                        w[27:30] = np.zeros(3)                                       # base lin. vel. 
                        w[15+(contact_idx*3):15+3+(contact_idx*3)] = np.zeros(3)     # joint pos.
                        w[33+(contact_idx*3):33+3+(contact_idx*3)] = np.zeros(3)     # joint vel.
                    # taking-off (add disturbances only on the joint positions)
                    elif CURR_IN_CONTACT and not NEXT_IN_CONTACT:
                        SAMPLE_DISTURBANCE = True
                        w[9:12]  = np.zeros(3)                                       # base pos. 
                        w[33+(contact_idx*3):33+3+(contact_idx*3)] = np.zeros(3)     # joint vel.
                params_k = np.concatenate(
                    [
                    contacts_logic_N[time_idx],
                    contacts_position_N[time_idx], 
                    contacts_norms_N[time_idx].flatten(),  
                    x_ref_N[time_idx][12:16],
                    w 
                    ]
                )
                # feedback control policy
                ol_u = u[time_idx, :]
                # if self.STOCHASTIC_OCP:
                #     # print(print("closed-loop error = ,", x_cl - x0) )
                #     cl_u = ol_u + K[time_idx] @ (x_cl - x0)
                #     # simulate closed-loop dynamics with state feedback control policy and disturbances        
                #     acados_integrator.set("p", params_k)
                #     acados_integrator.set("x", x_cl)
                #     acados_integrator.set("u", cl_u)
                #     status = self.acados_integrator.solve()
                #     if status != 0:
                #         raise Exception('acados returned status {}. Exiting.'.format(status))
                #     # get next closed-loop state
                #     x_cl = acados_integrator.get("x")                        
                #     # save closed-loop trajectories
                #     x_sim[sim, time_idx+1,:] = np.copy(x_cl)   
                #     # simulate open-loop dynamics with open-loop without disturbances
                #     params_k[-nx::] = np.zeros(nx)
                #     acados_integrator.set("p", params_k)
                #     acados_integrator.set("x", x0)
                #     acados_integrator.set("u", ol_u)
                #     status = self.acados_integrator.solve()
                #     if status != 0:
                #         raise Exception('acados returned status {}. Exiting.'.format(status))
                #     x0 = acados_integrator.get("x")
                # else:
                # simulate closed-loop dynamics with open-loop actions and disturbances        
                acados_integrator.set("p", params_k)
                acados_integrator.set("x", x0)
                acados_integrator.set("u", ol_u)
                status = self.acados_integrator.solve()
                if status != 0:
                    raise Exception('acados returned status {}. Exiting.'.format(status))    
                # get next open-loop state
                x0 = acados_integrator.get("x")
                x_sim[sim, time_idx+1,:] = np.copy(x0)   
        return x_sim
    
if __name__ == "__main__":
    import conf_solo12_trot_step_adjustment_full_kinematics_mpc as conf
    from centroidal_plus_double_integrator_kinematics_casadi_model import CentroidalPlusLegKinematicsCasadiModel
    from centroidal_plus_double_integrator_kinematics_acados_solver import CentroidalPlusLegKinematicsAcadosSolver
    from wholebody_croccodyl_solver import WholeBodyDDPSolver
    from wholebody_croccodyl_model import WholeBodyModel
    import matplotlib.pylab as plt
    # DDP warm-start
    wbd_model = WholeBodyModel(conf)
    ddp_planner = WholeBodyDDPSolver(wbd_model, MPC=False, WARM_START=False)
    ddp_planner.solve()
    ddp_sol = ddp_planner.get_solution_trajectories()
    centroidal_warmstart = ddp_sol['centroidal']
    q_warmstart = ddp_sol['jointPos']
    qdot_warmstart = ddp_sol['jointVel']
    x_warmstart = []
    u_warmstart = []
    rmodel, rdata = conf.rmodel, conf.rdata
    for k in range(len(centroidal_warmstart)):
        x_warmstart.append(
            np.concatenate(
                [centroidal_warmstart[k],
                q_warmstart[k], 
                qdot_warmstart[k]]
                )
            )
        u_warmstart.append(np.concatenate([np.zeros(30)]))
    # build casadi and acados models
    model = CentroidalPlusLegKinematicsCasadiModel(conf, STOCHASTIC_OCP=False)
    solver = CentroidalPlusLegKinematicsAcadosSolver(
        model, x_warmstart, u_warmstart, MPC=True
        )
    x_sol, u_sol, lqr_gains = solver.solve()
    nb_sims = 5
    # build simulator 
    simulator = CentroidalPlusLegKinematicsAcadosSimulator(model, solver)
    # sample disturbances 
    w_total = np.zeros((nb_sims, conf.N-1, x_sol.shape[1]))
    for sim in range(nb_sims):
        for time_idx in range(conf.N-1):
            w_total[sim, time_idx, :] = simulator.sample_addtitive_uncertainties() 
    # python stuff
    x_sim = simulator.simulate(x_sol[0, :], u_sol, lqr_gains, np.copy(w_total), nb_sims=nb_sims, WITH_DISTURBANCES=True)
    # reuse c-generated code


    # # plot joint positions,velocities and accelerations
    # time = np.arange(0, np.round((x_sol.shape[0])*conf.dt, 2), conf.dt)
    # for contact in conf.ee_frame_names:
    #     plt.rc('font', family ='serif')
    #     contact_name = contact[0:2]
    #     if contact_name == 'FL':
    #         q_sol = x_sol[:, 15:18]
    #         qdot_sol = x_sol[:, 33:36]
    #         q_sim = x_sim[:, :, 15:18]
    #         qdot_sim = x_sim[:, :, 33:36]
    #     elif contact_name == 'FR':
    #         q_sol = x_sol[:, 18:21]
    #         qdot_sol = x_sol[:, 36:39]
    #         q_sim = x_sim[:, :, 18:21]
    #         qdot_sim = x_sim[:, :, 36:39]
    #     elif contact_name == 'HL':
    #         q_sol = x_sol[:, 21:24]
    #         qdot_sol = x_sol[:, 39:42]
    #         q_sim = x_sim[:, :, 21:24]
    #         qdot_sim = x_sim[:, :, 39:42]
    #     elif contact_name == 'HR':
    #         q_sol = x_sol[:, 24:27]
    #         qdot_sol = x_sol[:, 42:45]
    #         q_sim = x_sim[:, :, 24:27]
    #         qdot_sim = x_sim[:, :, 42:45]
    #     # joint positions
    #     fig1, (HAA, HFE, KFE) = plt.subplots(3, 1, sharex=True)
    #     HAA.step(time, q_sol[:, 0])
    #     HFE.step(time, q_sol[:, 1])
    #     KFE.step(time, q_sol[:, 2])
    #     for i in range(q_sim.shape[0]):
    #         HAA.step(time, q_sim[i, :, 0])
    #         HFE.step(time, q_sim[i, :, 1])
    #         KFE.step(time, q_sim[i, :, 2])
    #     fig1.suptitle(str(contact[0:2])+ ' joint positions (rad)')
    #     HAA.set_title('HAA')
    #     HFE.set_title('HFE')
    #     KFE.set_title('KFE')
    #     KFE.set_xlabel('Time (s)', fontsize=14)    
    #     # joint velocities
    #     fig2, (HAA, HFE, KFE) = plt.subplots(3, 1, sharex=True)
    #     HAA.step(time, qdot_sol[:, 0])
    #     HFE.step(time, qdot_sol[:, 1])
    #     KFE.step(time, qdot_sol[:, 2])
    #     for i in range(q_sim.shape[0]):
    #         HAA.step(time, qdot_sim[i, :, 0])
    #         HFE.step(time, qdot_sim[i, :, 1])
    #         KFE.step(time, qdot_sim[i, :, 2])
    #     fig2.suptitle(str(contact[0:2])+ ' joint velocities (rad/s)')
    #     HAA.set_title('HAA')    
    #     HFE.set_title('HFE')
    #     KFE.set_title('KFE')
    #     KFE.set_xlabel('Time (s)', fontsize=14)   
    # plt.show()    