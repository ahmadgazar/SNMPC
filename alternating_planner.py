from centroidal_acados_solver import CentroidalSolverAcados
from centroidal_casadi_model import CentroidalModelCasadi
from wholebody_croccodyl_solver import WholeBodyDDPSolver
from wholebody_croccodyl_model import WholeBodyModel
import numpy as np
import time 

class ComboPlanner:
    # constructor
    def __init__(self, conf, MPC=False):
        self.RECEEDING_HORIZON = MPC
        # timing
        self.N_mpc_centroidal = conf.N_mpc
        self.N_mpc_wbd = conf.N_mpc_wbd
        self.N_traj = conf.N
        self.dt = conf.dt
        self.__warm_start(conf)
        self.__init_sub_planners(conf, MPC)
    
    def __warm_start(self, conf):
        wbd_planner_warmstart = WholeBodyDDPSolver(WholeBodyModel(conf))
        # solving the WBD OCP to warm-start centroidal solver
        print('\n'+'=' * 100)
        print('Running whole-body OCP to warm start centroidal solver ... ' )
        print('-' * 100)
        wbd_planner_warmstart.solve()
        self.centroidal_warmstart = wbd_planner_warmstart.get_solution_trajectories()['centroidal']
        self.centroidal_model = CentroidalModelCasadi(conf)
        centroidal_planner_warm_start = CentroidalSolverAcados( 
                self.centroidal_model, self.centroidal_warmstart, MPC=False
                )
        # solving centroidal OCP to warm-start WBD tracking MPC
        print('\n'+'=' * 100)
        print('Running centroidal OCP to setup whole-body centroidal and force tracking tasks ... ' )
        print('-' * 100)
        self.wbdCentroidalTrackTask, self.wbdForceTrackTask = centroidal_planner_warm_start.solve()

    def __init_sub_planners(self, conf, MPC_):
        # initialize centroidal MPC planner
        if MPC_:
            self.centroidal_planner = CentroidalSolverAcados(
                self.centroidal_model, self.centroidal_warmstart, MPC=True)
            # repeat last MPC tail to be the same as terminal trajectory task  
            for _ in range(conf.N_mpc_wbd):
                self.wbdCentroidalTrackTask = np.concatenate(
                    [self.wbdCentroidalTrackTask, self.wbdCentroidalTrackTask[-1].reshape(1, conf.n_x)]
                    )
                self.wbdForceTrackTask = np.concatenate(
                    [self.wbdForceTrackTask, self.wbdForceTrackTask[-1].reshape(1, conf.n_u)]
                    )
        self.wbd_planner = WholeBodyDDPSolver(
            WholeBodyModel(conf), 
            centroidalTask=self.wbdCentroidalTrackTask, 
            forceTask=self.wbdForceTrackTask, 
            MPC=MPC_, WARM_START=True
            )

    def run_OL_MPC(self):
        N_traj, N_mpc_centroidal, N_mpc_wbd = self.N_traj, self.N_mpc_centroidal, self.N_mpc_wbd
        centroidal_planner = self.centroidal_planner
        wbd_planner = self.wbd_planner
        hg0 = centroidal_planner.x_init[0]
        # create open-loop solution tuples
        wbd_nx, wbd_nu = len(wbd_planner.x_init[0]), len(wbd_planner.u_init[0])
        centroidal_nx, centroidal_nu = 9, 12
        X_sim_centroidal = np.zeros((N_traj, N_mpc_centroidal+1, wbd_nx))
        U_sim_centroidal = np.zeros((N_traj, N_mpc_centroidal, wbd_nu))
        wbd_sol = []
        # create closed-loop solution tuples
        X_sol_centroidal = np.zeros((N_traj, centroidal_nx))
        U_sol_centroidal = np.zeros((N_traj, centroidal_nu))
        for traj_time_idx in range(N_traj):
            centroidal_planner.update_ocp(traj_time_idx, hg0)
            if centroidal_planner.ocp.solver_options.nlp_solver_type == 'SQP_RTI':
                # feedback rti_phase (solving QP)
                print('starting RTI feedback phase ' + '...')
                centroidal_planner.acados_solver.options_set('rti_phase', 2)
                t_feedback = time.time()
                status = centroidal_planner.acados_solver.solve()
                elapsed_feedback = time.time() - t_feedback
                print('RTI feedback phase took ' + str(elapsed_feedback) + " seconds")
                centroidal_planner.acados_solver.print_statistics()
                if status == 0:
                    print("HOORAY ! found a solution after :", 
                           centroidal_planner.elapsed_prep+elapsed_feedback, " seconds")
                else:
                    raise Exception(f'acados returned status {status}.')
            else:
                t = time.time()
                status = centroidal_planner.acados_solver.solve()
                elapsed_time= time.time() - t
                centroidal_planner.acados_solver.print_statistics()
                if status == 0:
                    print("HOORAY ! found a solution after :", elapsed_time, " seconds")
                else:
                    raise Exception(f'acados returned status {status}.')        
            x_sol = np.array([centroidal_planner.acados_solver.get(i,"x") for i in range(N_mpc_centroidal+1)])
            u_sol = np.array([centroidal_planner.acados_solver.get(i,"u") for i in range(N_mpc_centroidal)])
            # add WBD tracking costs from the centroidal solver solution
            wbd_planner.update_ocp(traj_time_idx, centroidalTask=x_sol[:N_mpc_wbd], forceTask=u_sol[:N_mpc_wbd])
            # solve WBD OCP
            if traj_time_idx == 0:
                wbd_planner.solver.solve(wbd_planner.x_init, wbd_planner.u_init)  
            else:
                wbd_planner.solver.solve(xs, us)
            xs = [wbd_planner.solver.xs[i] for i in range(len(wbd_planner.solver.xs))]
            us = [wbd_planner.solver.us[i] for i in range(len(wbd_planner.solver.us))]
            # save open-loop solution
            wbd_sol += [wbd_planner.get_solution_trajectories()]
            # # save closed-loop solution
            X_sol_centroidal[traj_time_idx] = x_sol[0]
            U_sol_centroidal[traj_time_idx] = u_sol[0]
            # warm-start solver from the previous solution 
            xs = xs[1:] + [xs[-1]]     
            us = us[1:] + [us[-1]]    
            # update solvers initial conditions
            hg0 = x_sol[1]
            wbd_planner.x0 = xs[0]
        return wbd_sol      
           