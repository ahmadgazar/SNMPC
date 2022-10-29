
from robot_properties_kuka.config import IiwaConfig
from pinocchio import casadi as cpin
import numpy as np
import pinocchio as pin 


q0 = np.array([0., 1.57,  -1.57 , -1.2,  1.57 ,  -1.57 ,  0.37 ])
ee_target_pos = np.array([0.1, 0.4, 0.45])
ee_target_pos2 = np.array([ 0.4, 0.5, 0.2])
# robot
ee_frame_name = 'contact'
robot = IiwaConfig.buildRobotWrapper()
rmodel = robot.model
rdata = rmodel.createData()
pin.framesForwardKinematics(rmodel, rdata, q0)
ee_init_pos = rdata.oMf[rmodel.getFrameId(ee_frame_name)].translation  
cmodel = cpin.Model(rmodel)
cdata = cmodel.createData()
# timing
dt = 0.01
dt_ctrl = 0.001
N_traj = 300
N_mpc  = 20
# QR control
nq = q0.shape[0]
Q = np.eye(2*nq)
R = 0.1*np.eye(nq)
# uncertainty parameters 
cov_w_dt = dt*np.eye(2*nq)
epsilon = 0.01
# cost function weights
IK_cost_weight = 5e-1
IK_Kp_gains = np.diag([0.5, 0.5, 0.5])
state_cost_weight = np.diag(
    [1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3,
     5e1, 5e1, 5e1, 5e1, 5e1, 5e1, 5e1]
    )
control_cost_weight = np.diag([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])
end_effector_position_weight = np.diag([1e3, 1e3, 1e3])
# box constraints
state_limits = np.array([
    2.97, 2.09, 2.97, 2.09, 2.97, 2.09, 3.05, # q
    1.48, 1.48, 1.75, 1.31, 2.27, 2.36, 2.36  # qdot
    ])
control_limits = np.array([320, 320, 176, 176, 110, 40, 40])

#  from __future__ import print_function
  
#  import numpy as np
#  from numpy.linalg import norm, solve
  
#  import pinocchio
  
#  model = pinocchio.buildSampleModelManipulator()
#  data  = model.createData()
  
#  JOINT_ID = 6
#  oMdes = pinocchio.SE3(np.eye(3), np.array([1., 0., 1.]))
  
#  q      = pinocchio.neutral(model)
#  eps    = 1e-4
#  IT_MAX = 1000
#  DT     = 1e-1
#  damp   = 1e-12
  
#  i=0
#  while True:
#      pinocchio.forwardKinematics(model,data,q)
#      dMi = oMdes.actInv(data.oMi[JOINT_ID])
#      err = pinocchio.log(dMi).vector
#      if norm(err) < eps:
#          success = True
#          break
#      if i >= IT_MAX:
#          success = False
#          break
#      J = pinocchio.computeJointJacobian(model,data,q,JOINT_ID)
#      v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
#      q = pinocchio.integrate(model,q,v*DT)
#      if not i % 10:
#          print('%d: error = %s' % (i, err.T))
#      i += 1
  
#  if success:
#      print("Convergence achieved!")
#  else:
#      print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")
  
#  print('\nresult: %s' % q.flatten().tolist  