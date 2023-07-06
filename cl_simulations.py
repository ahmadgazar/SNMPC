from centroidal_plus_double_integrator_kinematics_acados_simulator import CentroidalPlusLegKinematicsAcadosSimulator
from centroidal_plus_double_integrator_kinematics_acados_solver import CentroidalPlusLegKinematicsAcadosSolver
from centroidal_plus_double_integrator_kinematics_casadi_model  import CentroidalPlusLegKinematicsCasadiModel
import conf_solo12_bound_step_adjustment_full_kinematics_mpc as conf_bound
import conf_solo12_trot_step_adjustment_full_kinematics_mpc as conf_trot
from robot_properties_solo.solo12wrapper import Solo12Config
from wholebody_croccodyl_solver import WholeBodyDDPSolver
from wholebody_croccodyl_model import WholeBodyModel
import matplotlib.pylab as plt
import pinocchio as pin
import numpy as np
import utils
import sys

def call_trot(conf_trot, x_warmstart_trot, u_warmstart_trot, nb_sims):
    # nominal traj-opt trot
    model_nom_trot = CentroidalPlusLegKinematicsCasadiModel(conf_trot, STOCHASTIC_OCP=False)
    solver_nom_trot = CentroidalPlusLegKinematicsAcadosSolver(
        model_nom_trot, x_warmstart_trot, u_warmstart_trot, MPC=True
    )
    # stochastic traj-opt trot
    model_stoch_trot = CentroidalPlusLegKinematicsCasadiModel(conf_trot, STOCHASTIC_OCP=True)
    solver_stoch_trot = CentroidalPlusLegKinematicsAcadosSolver(
        model_stoch_trot, x_warmstart_trot, u_warmstart_trot, MPC=True
    )
    # nominal traj-opt trot with heurestic safety margin
    model_heuristic_trot = CentroidalPlusLegKinematicsCasadiModel(conf_trot, STOCHASTIC_OCP=False)
    solver_heuristic_trot = CentroidalPlusLegKinematicsAcadosSolver(
        model_heuristic_trot, x_warmstart_trot, u_warmstart_trot, conf_trot.heuristic_bound, MPC=True
    )
    # create simulators
    simulator_nom_trot = CentroidalPlusLegKinematicsAcadosSimulator(model_nom_trot, solver_nom_trot)
    simulator_stoch_trot = CentroidalPlusLegKinematicsAcadosSimulator(model_stoch_trot, solver_stoch_trot)
    simulator_heuristic_trot = CentroidalPlusLegKinematicsAcadosSimulator(model_heuristic_trot, solver_heuristic_trot)

    # sample addtitive disturbances trot and pass them to both simulators
    w_total_trot = np.zeros((nb_sims, conf_trot.N-1, conf_trot.n_x))
    for sim in range(nb_sims):
        for time_idx in range(conf_trot.N-1):
            w_total_trot[sim, time_idx, :] = simulator_nom_trot.sample_addtitive_uncertainties()
    # simulate nominal MPC trot
    x_sim_nom_trot, cost_nom_trot = simulator_nom_trot.simulate(
        solver_nom_trot.x0, np.copy(w_total_trot), nb_sims=nb_sims, WITH_DISTURBANCES=True
        )
    # simulate stochastic MPC trot
    x_sim_stoch_trot, cost_stoch_trot = simulator_stoch_trot.simulate(
        solver_stoch_trot.x0, np.copy(w_total_trot), nb_sims=nb_sims, WITH_DISTURBANCES=True
        )
    # simulate nominal MPC trot with heuristic safety margin
    x_sim_heuristic_trot, cost_heuristic_trot = simulator_heuristic_trot.simulate(
        solver_heuristic_trot.x0, np.copy(w_total_trot), nb_sims=nb_sims, WITH_DISTURBANCES=True
        )
    # count constraint violations trot
    violations_nom_trot, ee_positions_nom_trot = simulator_nom_trot.count_constraint_violations(
        x_sim_nom_trot, WITH_VISUALIZATION=False
        )
    violations_stoch_trot, ee_positions_stoch_trot = simulator_stoch_trot.count_constraint_violations(
        x_sim_stoch_trot, WITH_VISUALIZATION=False
        )
    violations_heuristic_trot, ee_positions_heuristic_trot = simulator_heuristic_trot.count_constraint_violations(
        x_sim_heuristic_trot, WITH_VISUALIZATION=False
        )
    # visualize end-effector trajectories trot
    robot = Solo12Config.buildRobotWrapper()
    # load robot in meshcat viewer
    viz = pin.visualize.MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model
        )
    try:
        viz.initViewer(open=True)
    except ImportError as err:
        print(err)
        sys.exit(0)
    viz.loadViewerModel()
    # add contact surfaces
    s = simulator_nom_trot.debris_half_size
    for i, contacts in enumerate(simulator_nom_trot.contact_sequence):
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
    # visualize end-effector trajectories (TODO: visualize trajectories in mujoco with an army of robots)
    for sim in range(nb_sims):
        FL_ee_nom_trot = []
        FR_ee_nom_trot = []
        HL_ee_nom_trot = []
        HR_ee_nom_trot = []
        FL_ee_stoch_trot = []
        FR_ee_stoch_trot = []
        HL_ee_stoch_trot = []
        HR_ee_stoch_trot = []
        FL_ee_heuristic_trot = []
        FR_ee_heuristic_trot = []
        HL_ee_heuristic_trot = []
        HR_ee_heuristic_trot = []
        for traj_length_nom in range(len(ee_positions_nom_trot[sim])):
            FL_ee_nom_trot += [ee_positions_nom_trot[sim][traj_length_nom][0]]
            FR_ee_nom_trot += [ee_positions_nom_trot[sim][traj_length_nom][1]]
            HL_ee_nom_trot += [ee_positions_nom_trot[sim][traj_length_nom][2]]
            HR_ee_nom_trot += [ee_positions_nom_trot[sim][traj_length_nom][3]]
        for traj_length_stoch in range(len(ee_positions_stoch_trot[sim])):
            FL_ee_stoch_trot += [ee_positions_stoch_trot[sim][traj_length_stoch][0]]
            FR_ee_stoch_trot += [ee_positions_stoch_trot[sim][traj_length_stoch][1]]
            HL_ee_stoch_trot += [ee_positions_stoch_trot[sim][traj_length_stoch][2]]
            HR_ee_stoch_trot += [ee_positions_stoch_trot[sim][traj_length_stoch][3]]
        for traj_length_heuristic in range(len(ee_positions_heuristic_trot[sim])):
            FL_ee_heuristic_trot += [ee_positions_heuristic_trot[sim][traj_length_heuristic][0]]
            FR_ee_heuristic_trot += [ee_positions_heuristic_trot[sim][traj_length_heuristic][1]]
            HL_ee_heuristic_trot += [ee_positions_heuristic_trot[sim][traj_length_heuristic][2]]
            HR_ee_heuristic_trot += [ee_positions_heuristic_trot[sim][traj_length_heuristic][3]]
        # plot end-effectors trajectories
        # nominal end-effector trajectories
        utils.addLineSegment(
            viz, 'ee_trajectory_nom_FL'+str(sim), np.array(FL_ee_nom_trot).astype(np.float32).T, [1,0,0,0.5]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_nom_FR'+str(sim), np.array(FR_ee_nom_trot).astype(np.float32).T, [1,0,0,0.5]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_nom_HL'+str(sim), np.array(HL_ee_nom_trot).astype(np.float32).T, [1,0,0,0.5]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_nom_HR'+str(sim), np.array(HR_ee_nom_trot).astype(np.float32).T, [1,0,0,0.5]
            )    
        # stochastic end-effector trajectories
        utils.addLineSegment(
            viz, 'ee_trajectory_stoch_FL'+str(sim), np.array(FL_ee_stoch_trot).astype(np.float32).T, [0,1,0,1]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_stoch_FR'+str(sim), np.array(FR_ee_stoch_trot).astype(np.float32).T, [0,1,0,1]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_stoch_HL'+str(sim), np.array(HL_ee_stoch_trot).astype(np.float32).T, [0,1,0,1]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_stoch_HR'+str(sim), np.array(HR_ee_stoch_trot).astype(np.float32).T, [0,1,0,1]
            )
        # heurestic end-effector trajectories
        utils.addLineSegment(
            viz, 'ee_trajectory_heurestic_FL'+str(sim), np.array(FL_ee_heuristic_trot).astype(np.float32).T, [0,0,1,0.5]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_heurestic_FR'+str(sim), np.array(FR_ee_heuristic_trot).astype(np.float32).T, [0,0,1,0.5]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_heurestic_HL'+str(sim), np.array(HL_ee_heuristic_trot).astype(np.float32).T, [0,0,1,0.5]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_heurestic_HR'+str(sim), np.array(HR_ee_heuristic_trot).astype(np.float32).T, [0,0,1,0.5]
            )  
    # compute the norm of the difference between the closed-loop contact location and the center of contact surface trot
    contacts_logic_N_trot = simulator_nom_trot.contact_data['contacts_logic']
    contacts_position_N_trot = simulator_nom_trot.contact_data['contacts_position'] 

    contact_location_nom_total_trot   = [] 
    contact_location_stoch_total_trot = []
    contact_location_heuristic_total_trot = [] 
 
    contact_surface_location_nom_total_trot = []
    contact_surface_location_stoch_total_trot = []
    contact_surface_location_heuristic_total_trot = []
    
    norm_contact_location_deviation_nom_total_trot = []
    norm_contact_location_deviation_stoch_total_trot = []
    norm_contact_location_deviation_heuristic_total_trot = []

    for sim in range(nb_sims):
        contact_location_nom_per_traj_trot = []
        contact_location_stoch_per_traj_trot = []
        contact_location_heuristic_per_traj_trot = []

        contact_surface_location_nom_per_traj_trot = []
        contact_surface_location_stoch_per_traj_trot = []
        contact_surface_location_heuristic_per_traj_trot = []
        
        norm_contact_location_deviation_nom_per_traj_trot = []
        norm_contact_location_deviation_stoch_per_traj_trot = []
        norm_contact_location_deviation_heuristic_per_traj_trot = []
        # loop over trajectory length
        for time_idx in range(len(ee_positions_stoch_trot[sim])):
            contact_location_nom_k_trot = []
            contact_location_stoch_k_trot = []
            contact_location_heuristic_k_trot = []
            
            contact_surface_location_nom_k_trot = []
            contact_surface_location_stoch_k_trot = []
            contact_surface_location_heuristic_k_trot = []
            # loop over contacts
            for contact_idx in range(len(conf_trot.ee_frame_names)):
                # FL
                if contact_idx==0:
                    CURR_IN_CONTACT_trot = contacts_logic_N_trot[time_idx][1]
                    contact_surface_location_per_foot_trot = contacts_position_N_trot[time_idx][3:6]
                # FR
                elif contact_idx==1:
                    CURR_IN_CONTACT_trot = contacts_logic_N_trot[time_idx][0]
                    contact_surface_location_per_foot_trot = contacts_position_N_trot[time_idx][0:3]
                # HL
                elif contact_idx==2:
                    CURR_IN_CONTACT_trot = contacts_logic_N_trot[time_idx][3]
                    contact_surface_location_per_foot_trot = contacts_position_N_trot[time_idx][9:12]
                # HR
                elif contact_idx==3:
                    CURR_IN_CONTACT_trot = contacts_logic_N_trot[time_idx][2]
                    contact_surface_location_per_foot_trot = contacts_position_N_trot[time_idx][6:9]
                # compute contact location deviation only for the activated contacts
                if CURR_IN_CONTACT_trot:
                    contact_surface_location_stoch_k_trot += [contact_surface_location_per_foot_trot]
                    contact_location_stoch_k_trot += [ee_positions_stoch_trot[sim][time_idx][contact_idx]]
                    # nominal
                    if time_idx < len(ee_positions_nom_trot[sim]):
                        contact_surface_location_nom_k_trot += [contact_surface_location_per_foot_trot]
                        contact_location_nom_k_trot += [ee_positions_nom_trot[sim][time_idx][contact_idx]]
                    # heurestic
                    if time_idx < len(ee_positions_heuristic_trot[sim]):
                        contact_surface_location_heuristic_k_trot += [contact_surface_location_per_foot_trot]
                        contact_location_heuristic_k_trot += [ee_positions_heuristic_trot[sim][time_idx][contact_idx]]      
            # since some trajectories fail before resuming till the end in the nominal case
            # nominal
            if time_idx < len(ee_positions_nom_trot[sim]):
                contact_location_nom_per_traj_trot += [contact_location_nom_k_trot]
                contact_surface_location_nom_per_traj_trot += [contact_surface_location_nom_k_trot]
                # compute the contact location at the time of landing
                norm_contact_location_deviation_nom_per_traj_trot.append(
                        np.linalg.norm(
                            np.asarray(contact_location_nom_k_trot) - \
                            np.asarray(contact_surface_location_nom_k_trot)
                        )
                    )
            # heuristic
            if time_idx < len(ee_positions_heuristic_trot[sim]):
                contact_location_heuristic_per_traj_trot += [contact_location_heuristic_k_trot]
                contact_surface_location_heuristic_per_traj_trot += [contact_surface_location_heuristic_k_trot]
                # compute the contact location at the time of landing
                norm_contact_location_deviation_heuristic_per_traj_trot.append(
                        np.linalg.norm(
                            np.asarray(contact_location_heuristic_k_trot) - \
                            np.asarray(contact_surface_location_heuristic_k_trot)
                        )
                    )
            # stochastic 
            contact_location_stoch_per_traj_trot += [contact_location_stoch_k_trot]
            contact_surface_location_stoch_per_traj_trot += [contact_surface_location_stoch_k_trot]
            norm_contact_location_deviation_stoch_per_traj_trot.append(
                        np.linalg.norm(
                            np.asarray(contact_location_stoch_k_trot) -\
                            np.asarray(contact_surface_location_stoch_k_trot)
                        )
                    )
        # some data collection for debugging
        contact_location_nom_total_trot += [contact_location_nom_per_traj_trot]
        contact_location_stoch_total_trot += [contact_location_stoch_per_traj_trot]
        contact_location_heuristic_total_trot += [contact_location_heuristic_per_traj_trot]
        contact_surface_location_nom_total_trot += [contact_surface_location_nom_per_traj_trot]
        contact_surface_location_stoch_total_trot += [contact_surface_location_stoch_per_traj_trot]
        contact_surface_location_heuristic_total_trot += [contact_surface_location_heuristic_per_traj_trot]        
        norm_contact_location_deviation_nom_total_trot += [norm_contact_location_deviation_nom_per_traj_trot]
        norm_contact_location_deviation_stoch_total_trot += [norm_contact_location_deviation_stoch_per_traj_trot]
        norm_contact_location_deviation_heuristic_total_trot += [norm_contact_location_deviation_heuristic_per_traj_trot]
    # compute statistics trot
    mean_nom_total_trot = np.zeros(conf_trot.N)
    std_nom_total_trot = np.zeros(conf_trot.N)
    mean_stoch_total_trot = np.zeros(conf_trot.N)
    std_stoch_total_trot = np.zeros(conf_trot.N)
    mean_heuristic_total_trot = np.zeros(conf_trot.N)
    std_heuristic_total_trot = np.zeros(conf_trot.N)
    for traj_idx in range(conf_trot.N):
        samples_nom_k_trot = []
        samples_stoch_k_trot = []
        samples_heuristic_k_trot = []
        for sim in range(nb_sims):
            # collect nominal samples
            if traj_idx < len(norm_contact_location_deviation_nom_total_trot[sim]):
                samples_nom_k_trot += [norm_contact_location_deviation_nom_total_trot[sim][traj_idx]]
            # collect heuristic samples
            if traj_idx < len(norm_contact_location_deviation_heuristic_total_trot[sim]):
                samples_heuristic_k_trot += [norm_contact_location_deviation_heuristic_total_trot[sim][traj_idx]]
            # collect stochastic samples
            samples_stoch_k_trot += [norm_contact_location_deviation_stoch_total_trot[sim][traj_idx]]
        mean_nom_total_trot[traj_idx] = np.mean(samples_nom_k_trot)
        std_nom_total_trot[traj_idx] = np.std(samples_nom_k_trot)
        mean_stoch_total_trot[traj_idx] = np.mean(samples_stoch_k_trot)
        std_stoch_total_trot[traj_idx] = np.std(samples_stoch_k_trot)
        mean_heuristic_total_trot[traj_idx] = np.mean(samples_heuristic_k_trot)
        std_heuristic_total_trot[traj_idx] = np.std(samples_heuristic_k_trot)
    print("number of contact location constraint violations for nominal MPC trot:", violations_nom_trot)
    print("number of contact location constraint violations for stochastic MPC trot:", violations_stoch_trot)
    return norm_contact_location_deviation_nom_total_trot, norm_contact_location_deviation_stoch_total_trot, \
           mean_nom_total_trot, mean_stoch_total_trot, std_nom_total_trot, std_stoch_total_trot, violations_nom_trot, \
           violations_stoch_trot, cost_nom_trot, cost_stoch_trot, cost_heuristic_trot, mean_heuristic_total_trot,\
           std_heuristic_total_trot, violations_heuristic_trot       

def call_bound(conf_bound, x_warmstart_bound, u_warmstart_bound, nb_sims):
    # nominal traj-opt bound
    model_nom_bound = CentroidalPlusLegKinematicsCasadiModel(conf_bound, STOCHASTIC_OCP=False)
    solver_nom_bound = CentroidalPlusLegKinematicsAcadosSolver(
        model_nom_bound, x_warmstart_bound, u_warmstart_bound, MPC=True
    )
    # stochastic traj-opt bound
    model_stoch_bound = CentroidalPlusLegKinematicsCasadiModel(conf_bound, STOCHASTIC_OCP=True)
    solver_stoch_bound = CentroidalPlusLegKinematicsAcadosSolver(
        model_stoch_bound, x_warmstart_bound, u_warmstart_bound, MPC=True
        )
    # nominal traj-opt trot with heurestic safety margin
    model_heuristic_bound = CentroidalPlusLegKinematicsCasadiModel(conf_bound, STOCHASTIC_OCP=False)
    solver_heuristic_bound = CentroidalPlusLegKinematicsAcadosSolver(
        model_heuristic_bound, x_warmstart_bound, u_warmstart_bound, conf_bound.heuristic_bound, MPC=True
        )
    # create simulators
    simulator_nom_bound = CentroidalPlusLegKinematicsAcadosSimulator(model_nom_bound, solver_nom_bound)
    simulator_stoch_bound = CentroidalPlusLegKinematicsAcadosSimulator(model_stoch_bound, solver_stoch_bound)
    simulator_heuristic_bound = CentroidalPlusLegKinematicsAcadosSimulator(model_heuristic_bound, solver_heuristic_bound)

    # sample addtitive disturbances bound and pass them to both simulators
    w_total_bound = np.zeros((nb_sims, conf_bound.N-1, conf_bound.n_x))
    for sim in range(nb_sims):
        for time_idx in range(conf_bound.N-1):
            w_total_bound[sim, time_idx, :] = simulator_nom_bound.sample_addtitive_uncertainties() 
    # simulate nominal MPC bound
    x_sim_nom_bound,  cost_nom_bound = simulator_nom_bound.simulate(
        solver_nom_bound.x0, np.copy(w_total_bound), nb_sims=nb_sims, WITH_DISTURBANCES=True
        )
    # simulate stochastic MPC bound
    x_sim_stoch_bound, cost_stoch_bound = simulator_stoch_bound.simulate(
        solver_stoch_bound.x0, np.copy(w_total_bound), nb_sims=nb_sims, WITH_DISTURBANCES=True
        )
    # simulate nominal MPC trot with heuristic safety margin
    x_sim_heuristic_bound, cost_heuristic_bound = simulator_heuristic_bound.simulate(
        solver_heuristic_bound.x0, np.copy(w_total_bound), nb_sims=nb_sims, WITH_DISTURBANCES=True
        )
    # count constraint violations bound
    violations_nom_bound, ee_positions_nom_bound = simulator_nom_bound.count_constraint_violations(
        x_sim_nom_bound, WITH_VISUALIZATION=False
        )
    violations_stoch_bound, ee_positions_stoch_bound = simulator_stoch_bound.count_constraint_violations(
        x_sim_stoch_bound, WITH_VISUALIZATION=False
        )
    violations_heuristic_bound, ee_positions_heuristic_bound = simulator_heuristic_bound.count_constraint_violations(
        x_sim_heuristic_bound, WITH_VISUALIZATION=False
        )
    # visualize end-effector trajectories bound
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
    s = simulator_nom_bound.debris_half_size
    for i, contacts in enumerate(simulator_nom_bound.contact_sequence):
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
    # visualize end-effector trajectories (TODO: visualize trajectories in mujoco with an army of robots)
    for sim in range(nb_sims):
        FL_ee_nom_bound = []
        FR_ee_nom_bound = []
        HL_ee_nom_bound = []
        HR_ee_nom_bound = []
        FL_ee_stoch_bound = []
        FR_ee_stoch_bound = []
        HL_ee_stoch_bound = []
        HR_ee_stoch_bound = []
        FL_ee_heuristic_bound = []
        FR_ee_heuristic_bound = []
        HL_ee_heuristic_bound = []
        HR_ee_heuristic_bound = []
        for traj_length_nom in range(len(ee_positions_nom_bound[sim])):
            FL_ee_nom_bound += [ee_positions_nom_bound[sim][traj_length_nom][0]]
            FR_ee_nom_bound += [ee_positions_nom_bound[sim][traj_length_nom][1]]
            HL_ee_nom_bound += [ee_positions_nom_bound[sim][traj_length_nom][2]]
            HR_ee_nom_bound += [ee_positions_nom_bound[sim][traj_length_nom][3]]
        for traj_length_stoch in range(len(ee_positions_stoch_bound[sim])):
            FL_ee_stoch_bound += [ee_positions_stoch_bound[sim][traj_length_stoch][0]]
            FR_ee_stoch_bound += [ee_positions_stoch_bound[sim][traj_length_stoch][1]]
            HL_ee_stoch_bound += [ee_positions_stoch_bound[sim][traj_length_stoch][2]]
            HR_ee_stoch_bound += [ee_positions_stoch_bound[sim][traj_length_stoch][3]]
        for traj_length_heuristic in range(len(ee_positions_heuristic_bound[sim])):
            FL_ee_heuristic_bound += [ee_positions_heuristic_bound[sim][traj_length_heuristic][0]]
            FR_ee_heuristic_bound += [ee_positions_heuristic_bound[sim][traj_length_heuristic][1]]
            HL_ee_heuristic_bound += [ee_positions_heuristic_bound[sim][traj_length_heuristic][2]]
            HR_ee_heuristic_bound += [ee_positions_heuristic_bound[sim][traj_length_heuristic][3]]    
        # plot end-effectors trajectories
        # nominal end-effector trajectories
        utils.addLineSegment(
            viz, 'ee_trajectory_nom_FL'+str(sim), np.array(FL_ee_nom_bound).astype(np.float32).T, [1,0,0,1]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_nom_FR'+str(sim), np.array(FR_ee_nom_bound).astype(np.float32).T, [1,0,0,1]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_nom_HL'+str(sim), np.array(HL_ee_nom_bound).astype(np.float32).T, [1,0,0,1]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_nom_HR'+str(sim), np.array(HR_ee_nom_bound).astype(np.float32).T, [1,0,0,1]
            )    
        # stochastic end-effector trajectories
        utils.addLineSegment(
            viz, 'ee_trajectory_stoch_FL'+str(sim), np.array(FL_ee_stoch_bound).astype(np.float32).T, [0,1,0,1]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_stoch_FR'+str(sim), np.array(FR_ee_stoch_bound).astype(np.float32).T, [0,1,0,1]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_stoch_HL'+str(sim), np.array(HL_ee_stoch_bound).astype(np.float32).T, [0,1,0,1]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_stoch_HR'+str(sim), np.array(HR_ee_stoch_bound).astype(np.float32).T, [0,1,0,1]
            )
        # heurestic end-effector trajectories
        utils.addLineSegment(
            viz, 'ee_trajectory_heurestic_FL'+str(sim), np.array(FL_ee_heuristic_bound).astype(np.float32).T, [0,0,1,0.5]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_heurestic_FR'+str(sim), np.array(FR_ee_heuristic_bound).astype(np.float32).T, [0,0,1,0.5]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_heurestic_HL'+str(sim), np.array(HL_ee_heuristic_bound).astype(np.float32).T, [0,0,1,0.5]
            )
        utils.addLineSegment(
            viz, 'ee_trajectory_heurestic_HR'+str(sim), np.array(HR_ee_heuristic_bound).astype(np.float32).T, [0,0,1,0.5]
            )  
    # compute the norm of the difference between the closed-loop contact location 
    # and the center of contact surface bound
    contacts_logic_N_bound = simulator_nom_bound.contact_data['contacts_logic']
    contacts_position_N_bound = simulator_nom_bound.contact_data['contacts_position'] 
    contact_location_nom_total_bound   = [] 
    contact_location_stoch_total_bound = []
    contact_location_heuristic_total_bound = [] 
 
    contact_surface_location_nom_total_bound = []
    contact_surface_location_stoch_total_bound = []
    contact_surface_location_heuristic_total_bound = []
    
    norm_contact_location_deviation_nom_total_bound = []
    norm_contact_location_deviation_stoch_total_bound = []
    norm_contact_location_deviation_heuristic_total_bound = []

    for sim in range(nb_sims):
        contact_location_nom_per_traj_bound = []
        contact_location_stoch_per_traj_bound = []
        contact_location_heuristic_per_traj_bound = []

        contact_surface_location_nom_per_traj_bound = []
        contact_surface_location_stoch_per_traj_bound = []
        contact_surface_location_heuristic_per_traj_bound = []
        
        norm_contact_location_deviation_nom_per_traj_bound = []
        norm_contact_location_deviation_stoch_per_traj_bound = []
        norm_contact_location_deviation_heuristic_per_traj_bound = []

        for time_idx in range(len(ee_positions_stoch_bound[sim])):
            contact_location_nom_k_bound = []
            contact_location_stoch_k_bound = []
            contact_location_heuristic_k_bound = []
            
            contact_surface_location_nom_k_bound = []
            contact_surface_location_stoch_k_bound = []
            contact_surface_location_heuristic_k_bound = []
            for contact_idx in range(len(conf_bound.ee_frame_names)):
                # FL
                if contact_idx==0:
                    CURR_IN_CONTACT_bound = contacts_logic_N_bound[time_idx][1]
                    contact_surface_location_per_foot_bound = contacts_position_N_bound[time_idx][3:6]
                # FR
                elif contact_idx==1:
                    CURR_IN_CONTACT_bound = contacts_logic_N_bound[time_idx][0]
                    contact_surface_location_per_foot_bound = contacts_position_N_bound[time_idx][0:3]
                # HL
                elif contact_idx==2:
                    CURR_IN_CONTACT_bound = contacts_logic_N_bound[time_idx][3]
                    contact_surface_location_per_foot_bound = contacts_position_N_bound[time_idx][9:12]
                # HR
                elif contact_idx==3:
                    CURR_IN_CONTACT_bound = contacts_logic_N_bound[time_idx][2]
                    contact_surface_location_per_foot_bound = contacts_position_N_bound[time_idx][6:9]
                # compute contact location deviation only for the activated contacts
                if CURR_IN_CONTACT_bound:
                    contact_surface_location_stoch_k_bound += [contact_surface_location_per_foot_bound]
                    contact_location_stoch_k_bound += [ee_positions_stoch_bound[sim][time_idx][contact_idx]]
                    if time_idx < len(ee_positions_nom_bound[sim]):
                        contact_surface_location_nom_k_bound += [contact_surface_location_per_foot_bound]
                        contact_location_nom_k_bound += [ee_positions_nom_bound[sim][time_idx][contact_idx]]
                    # heurestic
                    if time_idx < len(ee_positions_heuristic_bound[sim]):
                        contact_surface_location_heuristic_k_bound += [contact_surface_location_per_foot_bound]
                        contact_location_heuristic_k_bound += [ee_positions_heuristic_bound[sim][time_idx][contact_idx]]      
            # since some trajectories fail before resuming till the end in the nominal case
             # nominal
            if time_idx < len(ee_positions_nom_bound[sim]):
                contact_location_nom_per_traj_bound += [contact_location_nom_k_bound]
                contact_surface_location_nom_per_traj_bound += [contact_surface_location_nom_k_bound]
                # compute the contact location at the time of landing
                norm_contact_location_deviation_nom_per_traj_bound.append(
                        np.linalg.norm(
                            np.asarray(contact_location_nom_k_bound) - \
                            np.asarray(contact_surface_location_nom_k_bound)
                        )
                    )
            # heuristic
            if time_idx < len(ee_positions_heuristic_bound[sim]):
                contact_location_heuristic_per_traj_bound += [contact_location_heuristic_k_bound]
                contact_surface_location_heuristic_per_traj_bound += [contact_surface_location_heuristic_k_bound]
                # compute the contact location at the time of landing
                norm_contact_location_deviation_heuristic_per_traj_bound.append(
                        np.linalg.norm(
                            np.asarray(contact_location_heuristic_k_bound) - \
                            np.asarray(contact_surface_location_heuristic_k_bound)
                        )
                    )
            # stochastic 
            contact_location_stoch_per_traj_bound += [contact_location_stoch_k_bound]
            contact_surface_location_stoch_per_traj_bound += [contact_surface_location_stoch_k_bound]
            norm_contact_location_deviation_stoch_per_traj_bound.append(
                        np.linalg.norm(
                            np.asarray(contact_location_stoch_k_bound) -\
                            np.asarray(contact_surface_location_stoch_k_bound)
                        )
                    )
        # some data collection for debugging
        contact_location_nom_total_bound += [contact_location_nom_per_traj_bound]
        contact_location_stoch_total_bound += [contact_location_stoch_per_traj_bound]
        contact_location_heuristic_total_bound += [contact_location_heuristic_per_traj_bound]
        contact_surface_location_nom_total_bound += [contact_surface_location_nom_per_traj_bound]
        contact_surface_location_stoch_total_bound += [contact_surface_location_stoch_per_traj_bound]
        contact_surface_location_heuristic_total_bound += [contact_surface_location_heuristic_per_traj_bound]        
        norm_contact_location_deviation_nom_total_bound += [norm_contact_location_deviation_nom_per_traj_bound]
        norm_contact_location_deviation_stoch_total_bound += [norm_contact_location_deviation_stoch_per_traj_bound]
        norm_contact_location_deviation_heuristic_total_bound += [norm_contact_location_deviation_heuristic_per_traj_bound]
    # compute statistics trot
    mean_nom_total_bound = np.zeros(conf_bound.N)
    std_nom_total_bound = np.zeros(conf_bound.N)
    mean_stoch_total_bound = np.zeros(conf_bound.N)
    std_stoch_total_bound = np.zeros(conf_bound.N)
    mean_heuristic_total_bound = np.zeros(conf_bound.N)
    std_heuristic_total_bound = np.zeros(conf_bound.N)
    for traj_idx in range(conf_bound.N):
        samples_nom_k_bound = []
        samples_stoch_k_bound = []
        samples_heuristic_k_bound = []
        for sim in range(nb_sims):
            # collect nominal samples
            if traj_idx < len(norm_contact_location_deviation_nom_total_bound[sim]):
                samples_nom_k_bound += [norm_contact_location_deviation_nom_total_bound[sim][traj_idx]]
            # collect heuristic samples
            if traj_idx < len(norm_contact_location_deviation_heuristic_total_bound[sim]):
                samples_heuristic_k_bound += [norm_contact_location_deviation_heuristic_total_bound[sim][traj_idx]]
            # collect stochastic samples
            samples_stoch_k_bound += [norm_contact_location_deviation_stoch_total_bound[sim][traj_idx]]
        mean_nom_total_bound[traj_idx] = np.mean(samples_nom_k_bound)
        std_nom_total_bound[traj_idx] = np.std(samples_nom_k_bound)
        mean_stoch_total_bound[traj_idx] = np.mean(samples_stoch_k_bound)
        std_stoch_total_bound[traj_idx] = np.std(samples_stoch_k_bound)
        mean_heuristic_total_bound[traj_idx] = np.mean(samples_heuristic_k_bound)
        std_heuristic_total_bound[traj_idx] = np.std(samples_heuristic_k_bound)
    print("number of contact location constraint violations for nominal MPC bound:", violations_nom_bound)
    print("number of contact location constraint violations for stochastic MPC bound:", violations_stoch_bound)
    return norm_contact_location_deviation_nom_total_bound, norm_contact_location_deviation_stoch_total_bound, \
           mean_nom_total_bound, mean_stoch_total_bound, std_nom_total_bound, std_stoch_total_bound, violations_nom_bound, \
           violations_stoch_bound, cost_nom_bound, cost_stoch_bound, cost_heuristic_bound, mean_heuristic_total_bound,\
           std_heuristic_total_bound, violations_heuristic_bound       

if __name__ == "__main__":
    # DDP warm-start trot
    wbd_model_trot = WholeBodyModel(conf_trot)
    ddp_planner_trot = WholeBodyDDPSolver(wbd_model_trot, MPC=False, WARM_START=False)
    ddp_planner_trot.solve()
    ddp_sol_trot = ddp_planner_trot.get_solution_trajectories()
    centroidal_warmstart_trot = ddp_sol_trot['centroidal']
    q_warmstart_trot = ddp_sol_trot['jointPos']
    qdot_warmstart_trot = ddp_sol_trot['jointVel']
    x_warmstart_trot = []
    u_warmstart_trot = []
    rmodel_trot, rdata_trot = conf_trot.rmodel, conf_trot.rdata
    for k in range(len(centroidal_warmstart_trot)):
        x_warmstart_trot.append(
            np.concatenate(
                [centroidal_warmstart_trot[k],
                q_warmstart_trot[k], 
                qdot_warmstart_trot[k]]
                )
            )
        u_warmstart_trot.append(np.concatenate([np.zeros(30)]))
    # DDP warm-start bound
    wbd_model_bound = WholeBodyModel(conf_bound)
    ddp_planner_bound = WholeBodyDDPSolver(wbd_model_bound, MPC=False, WARM_START=False)
    ddp_planner_bound.solve()
    ddp_sol_bound = ddp_planner_bound.get_solution_trajectories()
    centroidal_warmstart_bound = ddp_sol_bound['centroidal']
    q_warmstart_bound = ddp_sol_bound['jointPos']
    qdot_warmstart_bound = ddp_sol_bound['jointVel']
    x_warmstart_bound = []
    u_warmstart_bound = []
    rmodel_bound, rdata_bound = conf_bound.rmodel, conf_bound.rdata
    for k in range(len(centroidal_warmstart_bound)):
        x_warmstart_bound.append(
            np.concatenate(
                [centroidal_warmstart_bound[k],
                q_warmstart_bound[k], 
                qdot_warmstart_bound[k]]
                )
            )
        u_warmstart_bound.append(np.concatenate([np.zeros(30)]))
    # run monte-carlo simulations
    nb_sims = 500
    # call trot controllers
    norm_contact_location_deviation_nom_total_trot, norm_contact_location_deviation_stoch_total_trot, \
    mean_nom_total_trot, mean_stoch_total_trot, std_nom_total_trot, std_stoch_total_trot, violations_nom_trot, \
    violations_stoch_trot, cost_nom_trot, cost_stoch_trot, cost_heuristic_trot, mean_heuristic_total_trot,\
    std_heuristic_total_trot, violations_heuristic_trot = call_trot(
        conf_trot, x_warmstart_trot, u_warmstart_trot, nb_sims
        )
    # call bound controllers
    norm_contact_location_deviation_nom_total_bound, norm_contact_location_deviation_stoch_total_bound, \
    mean_nom_total_bound, mean_stoch_total_bound, std_nom_total_bound, std_stoch_total_bound, violations_nom_bound, \
    violations_stoch_bound, cost_nom_bound, cost_stoch_bound, cost_heuristic_bound, mean_heuristic_total_bound,\
    std_heuristic_total_bound, violations_heuristic_bound = call_bound(
        conf_bound, x_warmstart_bound, u_warmstart_bound, nb_sims
        )

    # print constraint violations
    # trot
    print("number of contact location constraint violations for nominal MPC trot:", violations_nom_trot)
    print("number of contact location constraint violations for stochastic MPC trot:", violations_stoch_trot)
    print("number of contact location constraint violations for heuristic MPC trot:", violations_heuristic_trot)
    # bound
    print("number of contact location constraint violations for nominal MPC bound:", violations_nom_bound)
    print("number of contact location constraint violations for stochastic MPC bound:", violations_stoch_bound)
    print("number of contact location constraint violations for heuristic MPC bound:", violations_heuristic_bound)
    # plot the mean and std-dev of contact location deviations over the horizon length trot
    fig1, ax1 = plt.subplots(1, 1, sharex=True) 
    time_stoch_trot = np.arange(0, np.round(
        (len(norm_contact_location_deviation_stoch_total_trot[0]))*conf_trot.dt, 2), conf_trot.dt
        )
    # stochastic trot
    ax1.plot(time_stoch_trot, mean_stoch_total_trot, color='green', label='trot stoch.')
    ax1.fill_between(
            time_stoch_trot,
            mean_stoch_total_trot+2*std_stoch_total_trot,
            mean_stoch_total_trot-2*std_stoch_total_trot, 
            color='green', 
            alpha=0.1
        )
    # nominal trot
    ax1.plot(time_stoch_trot, mean_nom_total_trot, color='red', label='trot nominal')
    ax1.fill_between(
            time_stoch_trot, 
            mean_nom_total_trot+2*std_nom_total_trot, 
            mean_nom_total_trot-2*std_nom_total_trot, 
            color='red',
            alpha=0.1
        )
    # heuristic trot
    ax1.plot(time_stoch_trot, mean_heuristic_total_trot, color='blue', label='trot heuristic')
    ax1.fill_between(
            time_stoch_trot, 
            mean_heuristic_total_trot+2*std_heuristic_total_trot, 
            mean_heuristic_total_trot-2*std_heuristic_total_trot, 
            color='blue',
            alpha=0.1
        )
    ax1.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel('Norm of contact locations deviations (m)', fontsize=14)
    # plot the mean and std-dev of contact location deviations over the horizon length bound
    time_stoch_bound = np.arange(
        0, np.round((len(norm_contact_location_deviation_stoch_total_bound[0]))*conf_bound.dt, 2), conf_bound.dt
        )
    # stochastic bound
    fig, ax2 = plt.subplots(1, 1, sharex=True) 
    ax2.plot(time_stoch_bound, mean_stoch_total_bound, color='green', label='bound stochastic')
    ax2.fill_between(
        time_stoch_bound, 
        mean_stoch_total_bound+2*std_stoch_total_bound, 
        mean_stoch_total_bound-2*std_stoch_total_bound, 
        color='green',
        alpha=0.1
    )
    # nominal bound
    ax2.plot(time_stoch_bound, mean_nom_total_bound, color='red', label='bound nominal')
    ax2.fill_between(
        time_stoch_bound, 
        mean_nom_total_bound+2*std_nom_total_bound, 
        mean_nom_total_bound-2*std_nom_total_bound, 
        color='red', 
        alpha=0.1
    )
    # heuristic bound
    ax2.plot(time_stoch_bound, mean_heuristic_total_bound, color='blue', label='bound heuristic')
    ax2.fill_between(
        time_stoch_bound,
        mean_heuristic_total_bound+2*std_heuristic_total_bound, 
        mean_heuristic_total_bound-2*std_heuristic_total_bound, 
        color='blue',
        alpha=0.1
    )
    ax1.legend()
    ax2.legend()
    # # plot least squares cost
    # # compute statistics
    # # trot 
    # mean_LS_nom_total_trot = np.zeros(conf_trot.N)
    # std_LS_nom_total_trot = np.zeros(conf_trot.N)
    # mean_LS_stoch_total_trot = np.zeros(conf_trot.N)
    # std_LS_stoch_total_trot = np.zeros(conf_trot.N)
    # mean_LS_heuristic_total_trot = np.zeros(conf_trot.N)
    # std_LS_heuristic_total_trot = np.zeros(conf_trot.N)
    # # bound 
    # mean_LS_nom_total_bound = np.zeros(conf_bound.N)
    # std_LS_nom_total_bound = np.zeros(conf_bound.N)
    # mean_LS_stoch_total_bound = np.zeros(conf_bound.N)
    # std_LS_stoch_total_bound = np.zeros(conf_bound.N)
    # mean_LS_heuristic_total_bound = np.zeros(conf_bound.N)
    # std_LS_heuristic_total_bound = np.zeros(conf_bound.N)
    # # trajectory loop
    # for traj_idx in range(conf_trot.N):
    #     # bound least-squares cost samples
    #     samples_LS_nom_k_trot = []
    #     samples_LS_stoch_k_trot = []
    #     samples_LS_heuristic_k_trot = []
    #     # bound least-squares cost samples
    #     samples_LS_nom_k_bound = []
    #     samples_LS_stoch_k_bound = []
    #     samples_LS_heuristic_k_bound = []
    #     # samples loop
    #     for sim in range(nb_sims):
    #         # nominal
    #         if traj_idx < len(cost_nom_trot[sim]):
    #             samples_LS_nom_k_trot += [cost_nom_trot[sim, traj_idx]]
    #         if traj_idx < len(cost_nom_bound[sim]):
    #             samples_LS_nom_k_bound += [cost_nom_bound[sim, traj_idx]]    
    #         # heuristic
    #         if traj_idx < len(cost_heuristic_trot[sim]):
    #             samples_LS_heuristic_k_trot += [cost_heuristic_trot[sim, traj_idx]]
    #         if traj_idx < len(cost_heuristic_bound[sim]):
    #             samples_LS_heuristic_k_bound += [cost_heuristic_bound[sim, traj_idx]]        
    #         # stochastic
    #         samples_LS_stoch_k_trot += [cost_stoch_trot[sim, traj_idx]]
    #         samples_LS_stoch_k_bound += [cost_stoch_bound[sim, traj_idx]]
    #     # nominal
    #     mean_LS_nom_total_trot[traj_idx] = np.mean(samples_LS_nom_k_trot)
    #     std_LS_nom_total_trot[traj_idx] = np.std(samples_LS_nom_k_trot)
    #     mean_LS_nom_total_bound[traj_idx] = np.mean(samples_LS_nom_k_bound)
    #     std_LS_nom_total_bound[traj_idx] = np.std(samples_LS_nom_k_bound)
    #     # stochastic
    #     mean_LS_stoch_total_trot[traj_idx] = np.mean(samples_LS_stoch_k_trot)
    #     std_LS_stoch_total_trot[traj_idx] = np.std(samples_LS_stoch_k_trot)
    #     mean_LS_stoch_total_bound[traj_idx] = np.mean(samples_LS_stoch_k_bound)
    #     std_LS_stoch_total_bound[traj_idx] = np.std(samples_LS_stoch_k_bound)
    #     # heuristic
    #     mean_LS_heuristic_total_trot[traj_idx] = np.mean(samples_LS_heuristic_k_trot)
    #     std_LS_heuristic_total_trot[traj_idx] = np.std(samples_LS_heuristic_k_trot)
    #     mean_LS_heuristic_total_bound[traj_idx] = np.mean(samples_LS_heuristic_k_bound)
    #     std_LS_heuristic_total_bound[traj_idx] = np.std(samples_LS_heuristic_k_bound)
    
    # # plot the mean and std of the LS cost over the horizon length 
    # fig, ax = plt.subplots(1, 1, sharex=True) 
    # time_trot = np.arange(0, np.round((len(norm_contact_location_deviation_stoch_total_trot[0]))*conf_trot.dt, 2), conf_trot.dt)
    # time_bound = np.arange(0, np.round((len(norm_contact_location_deviation_stoch_total_bound[0]))*conf_bound.dt, 2), conf_bound.dt)

    # # stochastic 
    # # ----------
    # # trot
    # plt.plot(time_trot, mean_LS_stoch_total_trot, color='green')
    # plt.fill_between(
    #     time_trot,
    #     mean_LS_stoch_total_trot+2*std_LS_stoch_total_trot, 
    #     mean_LS_stoch_total_trot-2*std_LS_stoch_total_trot,
    #     color='green', alpha=0.1
    #     )
    # # bound
    # plt.plot(time_bound, mean_LS_stoch_total_bound, color='orange')
    # plt.fill_between(
    #     time_bound,
    #     mean_LS_stoch_total_bound+2*std_LS_stoch_total_bound, 
    #     mean_LS_stoch_total_bound-2*std_LS_stoch_total_bound,
    #     color='orange', alpha=0.1
    #     )
    
    # # nominal
    # # -------
    # # trot
    # plt.plot(time_trot, mean_LS_nom_total_trot, color='red')
    # plt.fill_between(
    # time_trot, 
    # mean_LS_nom_total_trot+2*std_LS_nom_total_trot,
    # mean_LS_nom_total_trot-2*std_LS_nom_total_trot,
    # color='red', 
    # alpha=0.1
    # )
    # # bound
    # # -----
    # plt.plot(time_bound, mean_LS_nom_total_bound, color='blue')
    # plt.fill_between(
    # time_bound, 
    # mean_LS_nom_total_bound+2*std_LS_nom_total_bound,
    # mean_LS_nom_total_bound-2*std_LS_nom_total_bound,
    # color='blue', 
    # alpha=0.1
    # )
    # # heuristic
    # # ---------
    # # trot
    # plt.plot(time_trot, mean_LS_heuristic_total_trot, color='blue')
    # plt.fill_between(
    # time_trot, 
    # mean_LS_heuristic_total_trot+2*std_LS_heuristic_total_trot,
    # mean_LS_heuristic_total_trot-2*std_LS_heuristic_total_trot,
    # color='blue', 
    # alpha=0.1
    # )
    # # bound
    # plt.plot(time_bound, mean_LS_heuristic_total_bound, color='cyan')
    # plt.fill_between(
    # time_bound, 
    # mean_LS_heuristic_total_bound+2*std_LS_heuristic_total_bound,
    # mean_LS_heuristic_total_bound-2*std_LS_heuristic_total_bound,
    # color='cyan', 
    # alpha=0.1
    # )
    # ax.set_xlabel('Time (s)', fontsize=14)
    # ax.set_ylabel('Least squares cost', fontsize=14)
    plt.show()
