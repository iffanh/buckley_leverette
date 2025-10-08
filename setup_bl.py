from dataclasses import dataclass, field
import numpy as np
import casadi as ca
from solver_ocp import Ocp

from buckley_leverette import BuckleyLeverette

@dataclass
class BLParamsMpc:
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

@dataclass
class BLParamsMpcShort:
    umin: np.ndarray = field(default_factory=lambda: np.array([0.8])) # lower bound on u
    umax: np.ndarray = field(default_factory=lambda: np.array([1.2])) # upper bound on u
    uinit: float = 1.0  # initial condition for control
    
    length: float = 1.00
    Swc: float = 0.2  # connate water saturation
    Sor: float = 0.2  # residual oil saturation
    mu_w: float = 1.0
    mu_o: float = 5.0
    
    N: int = 50
    dt: float = 0.03
    time_window: float = N*dt
    total_time: float = 3
    nx: int = 25 # number of spatial discretization points
    
    ro: float = 1.0
    rwp: float = 0.01
    rwi: float = 0.1

def setup_bl_ocp(params_mpc: BLParamsMpc) -> Ocp:

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
    x = ca.SX.sym("Sw", nx)  # water saturation profile along the 1D domain

    # controls
    # u = ca.SX.sym("q", params_mpc.N)  # injection rate
    u = ca.SX.sym("q")  # injection rate
    u_prev = ca.SX.sym("q_prev")  # previous injection rate for control change penalty

    # dynamics
    f_discrete = bl.simulate_at_k(x, u)
    
    # TODO: stage cost and terminal cost
    # stage_cost = bl.stage_cost(x, u, qtk_prev=u_prev, alpha=0.001)
    stage_cost = bl.stage_cost(x, u, qtk_prev=u_prev, alpha=0.0)

    stage_constr = u + 0 
    stage_constr_lb = params_mpc.umin
    stage_constr_ub = params_mpc.umax
    stage_constr_init = params_mpc.uinit

    ocp = Ocp(
        x = x,
        u = u,
        u_prev = u_prev,
        N = params_mpc.N,
        dyn_expr = f_discrete,
        stage_cost_expr = stage_cost,
        stage_constr_expr = stage_constr,
        stage_constr_init = stage_constr_init,
        stage_constr_lb = stage_constr_lb,
        stage_constr_ub = stage_constr_ub,
    )

    return ocp

