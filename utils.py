import numpy as np
from casadi import *
import casadi as ca
#TODO change to jnp and jit

"""
inner approximation
"""
def construct_friction_pyramid_constraint_matrix(model):
    mu_linear = model._linear_friction_coefficient/np.sqrt(2)
    pyramid_constraint_matrix = np.array([[1. ,  0., -mu_linear], 
                                          [-1.,  0., -mu_linear],                                     
                                          [0. ,  1., -mu_linear], 
                                          [0. , -1., -mu_linear],
                                          [0. ,  0., -1.]])
    return pyramid_constraint_matrix

def compute_centroid(vertices):
    centroid = [0., 0., 0.]
    n = len(vertices)
    centroid[0] = np.sum(np.asarray(vertices)[:, 0])/n
    centroid[1] = np.sum(np.asarray(vertices)[:, 1])/n
    centroid[2] = np.sum(np.asarray(vertices)[:, 2])/n
    return centroid

def interpolate_centroidal_traj(conf, data):
    N = conf.N
    N_ctrl = conf.N_ctrl   
    N_inner = int(N_ctrl/N)
    result = {'state':np.zeros((conf.n_x, N_ctrl+N_inner)), 
                    'control':np.zeros((conf.n_u, N_ctrl)),
            'contact_sequence':np.array([]).reshape(0, 4)}
    for outer_time_idx in range(N+1):
        inner_time_idx = outer_time_idx*N_inner
        result['state'][:, inner_time_idx:inner_time_idx+N_inner] = np.tile(data['state'][:, outer_time_idx], (N_inner,1)).T
        if outer_time_idx < N:
            result['contact_sequence'] = np.vstack([result['contact_sequence'], 
                np.tile(data['contact_sequence'][outer_time_idx], (N_inner, 1))])  
            result['control'][:, inner_time_idx:inner_time_idx+N_inner] = \
                    np.tile(data['control'][:, outer_time_idx], (N_inner,1)).T
    return result 

# Generate trajectory using 3rd order polynomial with following constraints:
# x(0)=x0, x(T)=x1, dx(0)=dx(T)=0
# x(t) = a + b t + c t^2 + d t^3
# x(0) = a = x0
# dx(0) = b = 0
# dx(T) = 2 c T + 3 d T^2 = 0 => c = -3 d T^2 / (2 T) = -(3/2) d T
# x(T) = x0 + c T^2 + d T^3 = x1
#        x0 -(3/2) d T^3 + d T^3 = x1
#        -0.5 d T^3 = x1 - x0
#        d = 2 (x0-x1) / T^3
# c = -(3/2) T 2 (x0-x1) / (T^3) = 3 (x1-x0) / T^2
def compute_3rd_order_poly_traj(x0, x1, T, dt):
    a = x0
    b = np.zeros_like(x0)
    c = 3*(x1-x0) / (T**2)
    d = 2*(x0-x1) / (T**3)
    N = int(T/dt)
    n = x0.shape[0]
    x = np.zeros((n,N))
    dx = np.zeros((n,N))
    ddx = np.zeros((n,N))
    for i in range(N):
        t = i*dt
        x[:,i]   = a + b*t + c*t**2 + d*t**3
        dx[:,i]  = b + 2*c*t + 3*d*t**2
        ddx[:,i] = 2*c + 6*d*t
    return x, dx, ddx

def compute_5th_order_poly_traj(x0, x1, T, dt):
    # x(0)=x0, x(T)=x1, dx(0)=dx(T)=0
    # x(t) = a + b t + c t^2 + d t^3 + e t^4 + f t^5
    # x(0) = a = x0 
    # dx(0) = b = 0
    a = x0
    b = np.zeros_like(x0) 
    c = np.zeros_like(x0)
    f = np.zeros_like(x0)
    d = 2*(x1-x0 )/ (T**3)
    e = (x0-x1) / (T**4)
    N = int(T/dt)
    n = x0.shape[0]
    x = np.zeros((n,N))
    dx = np.zeros((n,N))
    ddx = np.zeros((n,N))
    for i in range(N):
        t = i*dt
        x[:,i]   = a + b*t + c*t**2 + d*t**3 + e*t**4 + f*t**5
        dx[:,i]  = b + 2*c*t + 3*d*t**2 + 4*e*t**3 + 5*f*t**4
        ddx[:,i] = 2*c + 6*d*t + 12*e*t**2 + 20*f*t**3
    return x, dx, ddx

def compute_norm_contact_slippage(contact_position):
  # first time instance the robot touches the ground
  for p in contact_position:
    if np.linalg.norm(p) > -1e-8 and  np.linalg.norm(p) < 1e-8:
      continue
    else:
      contact_pos_ref = p 
    break   
  contact_dev_norm = np.zeros(contact_position.shape[0])
  for time_idx in range(len(contact_position)-1):
    # ignore contact samples in the air  
    if np.linalg.norm(contact_position[time_idx], 2) > -1e-8 and np.linalg.norm(contact_position[time_idx], 2) < 1e-8:
      contact_pos_ref = contact_position[time_idx+1]
    else:
      slippage_norm = np.linalg.norm((contact_pos_ref-contact_position[time_idx]), 2)
      # ignore simulation spikes 
      if slippage_norm > 0.015:
        contact_dev_norm[time_idx] = contact_dev_norm[time_idx-1]
      else:
        contact_dev_norm[time_idx] = slippage_norm
  return contact_dev_norm    


def vec2sym_mat(vec, nx):
    # nx = (vec.shape[0])

    if isinstance(vec, np.ndarray):
        mat = np.zeros((nx,nx))
    else:
        mat = ca.SX.zeros(nx,nx)

    start_mat = 0
    for i in range(nx):
        end_mat = start_mat + (nx - i)
        aux = vec[start_mat:end_mat]
        mat[i,i:] = aux.T
        mat[i:,i] = aux
        start_mat += (nx-i)

    return mat


def sym_mat2vec(mat):
    nx = mat.shape[0]

    if isinstance(mat, np.ndarray):
        vec = np.zeros((int((nx+1)*nx/2),))
    else:
        vec = ca.SX.zeros(int((nx+1)*nx/2))

    start_mat = 0
    for i in range(nx):
        end_mat = start_mat + (nx - i)
        vec[start_mat:end_mat] = mat[i:,i]
        start_mat += (nx-i)

    return vec

def l1_permut_mat(nb_variables):
    permut_mat = np.zeros((2**nb_variables, nb_variables))
    for variable_idx in range(nb_variables):
        permut_mat[:, variable_idx] = np.array([(-1)**(j//(2**variable_idx)) for j in range(2**(nb_variables))])
    return permut_mat