import pinocchio as pin
from casadi import *
import numpy as np
import meshcat

"""
inner approximation
"""
def construct_friction_pyramid_constraint_matrix(model):
    mu_linear = model._linear_friction_coefficient/np.sqrt(2)
    pyramid_constraint_matrix = np.array([[1. ,  0., -mu_linear], 
                                          [-1.,  0., -mu_linear],                                     
                                          [0. ,  1., -mu_linear], 
                                          [0. , -1., -mu_linear]])
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
        result['state'][:, inner_time_idx:inner_time_idx+N_inner] = \
            np.tile(data['state'][:, outer_time_idx], (N_inner,1)).T
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


def normalize_quaternion(q):
    norm = np.sqrt((q[0]**2) + (q[1]**2) + (q[2]**2) + (q[3]**2))
    return q/norm 

def quaternion_multiplication(q1, q2):
    w1, v1 = q1[3], q1[0:3]
    w2, v2 = q2[3], q2[0:3]
    temp = w1*v2 + w2*v1 + (skew(v1) @ v2)  
    result = MX.zeros(4)
    result[0] = temp[0]
    result[1] = temp[1]
    result[2] = temp[2]
    result[3] = w1*w2 - (v1.T @ v2)
    return result

def quaternion_plus_casadi_fun():
    q1 = MX.sym('q', 4, 1)
    omega = MX.sym('w', 3, 1)
    # w1, v1 = q[3], q[0:3]
    q2 = vertcat(0.5*omega, 0.)
    q_next = quaternion_multiplication(q2, q1)
    # v_next = w1*v2 + w2*v1 + (mtimes(skew(v1), v2))
    return Function(
        'quaternion_plus',
        [q1, omega], 
        [q_next]
    )

def log_quaternion_casadi(q):
    """ lives on the tangent space of SO(3) """
    v = q[:3]
    w = q[3]
    vnorm = norm_2(v)
    q1log = 2*v / w*(1 - vnorm**2 / (3 * w**2))
    q2log = (2*atan2(vnorm, w)*v) / vnorm
    return if_else (
        vnorm <= 1.0e-6, q1log, q2log, True
        )

def quaternion_minus_casadi_fun():
    """computes the tangent vector from q1 to q2 at Identity
    returns vecotr w
    s.t. q2 = q1 circle-cross (exp(.5 * w))
    """
    q1 = MX.sym('q1', 4, 1)
    q2 = MX.sym('q2', 4, 1)
    # first compute dq s.t.  q2 = q1 (circle cross) dq
    q1conjugate = vertcat(-q1[0], -q1[1], -q1[2], q1[3])
    # order of multiplication is very essential here
    dq = quaternion_multiplication(q1conjugate, q2)
    # increment is log of dq
    return Function(
        'quaternion_minus',
        [q1, q2],
        [log_quaternion_casadi(dq)]
        )

def quatToRot_casadi(q):
    R = SX.zeros(3, 3)
    qi = q[0]; qj = q[1]; qk = q[2]; qr = q[3]
    R[0, 0] = 1. - 2. * (qj * qj + qk * qk)
    R[0, 1] = 2. * (qi * qj - qk * qr)
    R[0, 2] = 2. * (qi * qk + qj * qr)
    R[1, 0] = 2. * (qi * qj + qk * qr)
    R[1, 1] = 1. - 2. * (qi * qi + qk * qk)
    R[1, 2] = 2. * (qj * qk - qi * qr)
    R[2, 0] = 2. * (qi * qk - qj * qr)
    R[2, 1] = 2. * (qj * qk + qi * qr)
    R[2, 2] = 1. - 2. * (qi * qi + qj * qj)
    return R

def rotToQuat_casadi(R):
    q = MX.zeros(4)
    q[0] = 0.5*sign(R[2, 1] - R[1, 2])*sqrt(R[0, 0] - R[1, 1] - R[2, 2] + 1)
    q[1] = 0.5*sign(R[0, 2] - R[2, 0])*sqrt(R[1, 1] - R[2, 2] - R[0, 0] + 1)
    q[2] = 0.5*sign(R[1, 0] - R[0, 1])*sqrt(R[2, 2] - R[0, 0] - R[1, 1] + 1)
    q[3] = 0.5*sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1)    
    return q

def vec2sym_mat(vec, nx):
    # nx = (vec.shape[0])

    if isinstance(vec, np.ndarray):
        mat = np.zeros((nx,nx))
    else:
        mat = SX.zeros(nx,nx)

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
        vec = SX.zeros(int((nx+1)*nx/2))

    start_mat = 0
    for i in range(nx):
        end_mat = start_mat + (nx - i)
        vec[start_mat:end_mat] = mat[i:,i]
        start_mat += (nx-i)

    return vec

def l1_permut_mat(nb_variables):
    permut_mat = np.zeros((2**nb_variables, nb_variables))
    for variable_idx in range(nb_variables):
        permut_mat[:, variable_idx] = \
            np.array([(-1)**(j//(2**variable_idx)) for j in range(2**(nb_variables))])
    return permut_mat

def meshcat_material(r, g, b, a):
        material = meshcat.geometry.MeshPhongMaterial()
        material.color = int(r * 255) * 256 ** 2 + int(g * 255) * 256 + int(b * 255)
        material.opacity = a
        material.linewidth = 5.0
        return material

def addViewerBox(viz, name, sizex, sizey, sizez, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_object(meshcat.geometry.Box([sizex, sizey, sizez]),
                                meshcat_material(*rgba))

def addLineSegment(viz, name, vertices, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_object(meshcat.geometry.LineSegments(
                    meshcat.geometry.PointsGeometry(np.array(vertices)),     
                    meshcat_material(*rgba)
                    ))

def addPoint(viz, name, vertices, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_object(meshcat.geometry.Points(
                    meshcat.geometry.PointsGeometry(np.array(vertices)),     
                    meshcat_material(*rgba)
                    ))

def meshcat_transform(x, y, z, q, u, a, t):
    return np.array(pin.XYZQUATToSE3([x, y, z, q, u, a, t]))

def applyViewerConfiguration(viz, name, xyzquat):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_transform(meshcat_transform(*xyzquat))

   
def interpolate_one_step(
        model, dt_plan, dt_ctrl, 
        q, q_next, 
        qdot, qdot_next,
        qddot, qddot_next, 
        f, f_next
    ):
    nq, nv = len(q), len(qdot)
    nb_actuated_joints = nv-6 
    N_interpol = int(dt_plan/dt_ctrl)
    q_interpol = np.zeros((N_interpol, nq))
    qdot_interpol = np.zeros((N_interpol, nv))
    qddot_interpol = np.zeros((N_interpol, nv))
    f_interpol = np.zeros((N_interpol, nb_actuated_joints))
    df = (f_next - f)/float(N_interpol)
    dqdot = (qdot_next - qdot)/float(N_interpol)
    dqddot = (qddot_next - qddot)/float(N_interpol)
    for interpol_idx in range(N_interpol):
        q_interpol[interpol_idx, :] = pin.interpolate(
            model, q, q_next, interpol_idx*dt_ctrl/dt_plan
            )
        qdot_interpol[interpol_idx, :] = qdot + interpol_idx*dqdot
        qddot_interpol[interpol_idx, :] = qddot + interpol_idx*dqddot        
        f_interpol[interpol_idx, :] = f + interpol_idx*df
    return q_interpol, qdot_interpol, qddot_interpol, f_interpol 