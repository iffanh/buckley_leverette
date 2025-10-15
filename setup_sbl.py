from dataclasses import dataclass, field
import numpy as np
import casadi as ca
from solver_socp import Socp

from buckley_leverette import BuckleyLeverette
    
@dataclass
class BLParamsSmpc:
    umin: np.ndarray = field(default_factory=lambda: np.array([0.8])) # lower bound on u
    umax: np.ndarray = field(default_factory=lambda: np.array([1.2])) # upper bound on u
    uinit: float = 1.0  # initial condition for control
    
    length: float = 1.00
    Swc: float = 0.2  # connate water saturation
    Sor: float = 0.2  # residual oil saturation
    mu_w: float = 1.0
    mu_o: float = 5.0
    
    N: int = 100
    dt: float = 0.03
    time_window: float = N*dt
    total_time: float = 3
    nx: int = 25 # number of spatial discretization points
    
    ro: float = 1.0
    rwp: float = 0.01
    rwi: float = 0.1
    
    Ne: int = 10 # number of perturbation
    # fractional flow parameters
    aw: np.ndarray = np.array([21.33475586, 22.34223509, 19.30977293, 17.1096296 , 20.89934636, 19.63353215, 16.56611449, 19.4582751 , 18.93305358, 23.15931784])
    bw: np.ndarray = np.array([5.84997699, 5.8895987 , 4.62704057, 4.58838645, 6.03736705, 4.63427309, 5.01686782, 4.00604146, 5.30302979, 5.48509878])

@dataclass
class BLParamsSmpcShort:
    umin: np.ndarray = field(default_factory=lambda: np.array([0.8])) # lower bound on u
    umax: np.ndarray = field(default_factory=lambda: np.array([1.2])) # upper bound on u
    uinit: float = 1.0  # initial condition for control
    
    length: float = 1.00
    Swc: float = 0.2  # connate water saturation
    Sor: float = 0.2  # residual oil saturation
    mu_w: float = 1.0
    mu_o: float = 5.0
    
    N: int = 10
    dt: float = 0.03
    time_window: float = N*dt
    total_time: float = 3
    nx: int = 25 # number of spatial discretization points
    
    ro: float = 1.0
    rwp: float = 0.01
    rwi: float = 0.1
    
    Ne: int = 10 # number of perturbation
    # fractional flow parameters
    aw: np.ndarray = np.array([21.33475586, 22.34223509, 19.30977293, 17.1096296 , 20.89934636, 19.63353215, 16.56611449, 19.4582751 , 18.93305358, 23.15931784])
    bw: np.ndarray = np.array([5.84997699, 5.8895987 , 4.62704057, 4.58838645, 6.03736705, 4.63427309, 5.01686782, 4.00604146, 5.30302979, 5.48509878])

def setup_sbl_ocp(params_mpc: BLParamsSmpc) -> Socp:

    # # model parameter values
    length = params_mpc.length
    Swc = params_mpc.Swc  # connate water saturation
    Sor = params_mpc.Sor  # residual oil saturation
    mu_w = params_mpc.mu_w
    mu_o = params_mpc.mu_o
    
    dt = params_mpc.dt
    time_window = params_mpc.time_window
    nx = params_mpc.nx  # number of spatial discretization points
    
    ro = params_mpc.ro
    rwp = params_mpc.rwp
    rwi = params_mpc.rwi
    
    Ne = params_mpc.Ne
    aw = params_mpc.aw
    bw = params_mpc.bw
    
    bl = BuckleyLeverette(nx=nx, 
                          length=length, 
                          Swc=Swc, 
                          Sor=Sor, 
                          mu_w=mu_w, 
                          mu_o=mu_o, 
                          dt=dt, 
                          total_time=time_window,
                          ro=ro,
                          rwp=rwp,
                          rwi=rwi)

    # states
    x = ca.SX.sym("Sw", nx*Ne)  # water saturation profile along the 1D domain

    # controls
    # u = ca.SX.sym("q", params_mpc.N)  # injection rate
    u = ca.SX.sym("q")  # injection rate
    u_prev = ca.SX.sym("q_prev")  # previous injection rate for control change penalty

    # dynamics
    f_discrete = []
    for i, (a, b) in enumerate(zip(aw, bw)):
        f_discrete.append(bl.simulate_at_k(x[i*nx:(i+1)*nx], u, a, b))
    
    # f_discrete = ca.SX(*f_discrete)
    # print(f_discrete)
    
    # TODO: stage cost and terminal cost
    # stage_cost = bl.stage_cost(x, u, qtk_prev=u_prev, alpha=0.001)
    
    stage_cost = []
    for i in range(Ne):
        stage_cost.append(bl.stage_cost(x[i*nx:(i+1)*nx], u, qtk_prev=u_prev, alpha=0.0))

    stage_constr = u + 0 
    stage_constr_lb = params_mpc.umin
    stage_constr_ub = params_mpc.umax
    stage_constr_init = params_mpc.uinit

    ocp = Socp(
        x = x,
        u = u,
        u_prev = u_prev,
        Ne = Ne,
        N = params_mpc.N,
        dyn_expr = f_discrete,
        stage_cost_expr = stage_cost,
        stage_constr_expr = stage_constr,
        stage_constr_init = stage_constr_init,
        stage_constr_lb = stage_constr_lb,
        stage_constr_ub = stage_constr_ub,
    )

    return ocp

