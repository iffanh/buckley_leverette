from dataclasses import dataclass, field
import numpy as np
import casadi as ca
from solver_socp_te import Socp

from buckley_leverette_te import BuckleyLeverette
 
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
    
    N_mpc: int = 10
    dt: float = 0.03
    time_window: float = N_mpc*dt
    total_time: float = 3
    Nt:int = 100 #int(total_time/dt)
    nx: int = 25 # number of spatial discretization points
    
    ro: float = 1.0
    rwp: float = 0.3
    rwi: float = 0.1
    
    # Ne: int = 10 # number of perturbation
    # # fractional flow parameters
    # aw: np.ndarray = np.array([21.33475586, 22.34223509, 19.30977293, 17.1096296 , 20.89934636, 19.63353215, 16.56611449, 19.4582751 , 18.93305358, 23.15931784])
    # bw: np.ndarray = np.array([5.84997699, 5.8895987 , 4.62704057, 4.58838645, 6.03736705, 4.63427309, 5.01686782, 4.00604146, 5.30302979, 5.48509878])
    
    Ne: int = 3 # number of perturbation
    # fractional flow parameters
    
    aw: np.ndarray = field(default_factory=lambda: np.array([21.33475586, 22.34223509, 19.30977293]))
    bw: np.ndarray = field(default_factory=lambda: np.array([5.84997699, 5.8895987 , 4.62704057])) 

    
@dataclass
class BLParamsTrue:
    umin: np.ndarray = field(default_factory=lambda: np.array([0.8])) # lower bound on u
    umax: np.ndarray = field(default_factory=lambda: np.array([1.2])) # upper bound on u
    uinit: float = 1.0  # initial condition for control
    
    length: float = 1.00
    Swc: float = 0.2  # connate water saturation
    Sor: float = 0.2  # residual oil saturation
    mu_w: float = 1.0
    mu_o: float = 5.0
    
    N_mpc: int = 10
    dt: float = 0.03
    time_window: float = N_mpc*dt
    total_time: float = 3
    Nt = int(total_time/dt)
    nx: int = 25 # number of spatial discretization points
    
    ro: float = 1.0
    rwp: float = 0.3
    rwi: float = 0.1
        
    Ne: int = 1 # number of perturbation
    # fractional flow parameters
    aw: np.ndarray = field(default_factory=lambda: np.array([20.0]))
    bw: np.ndarray = field(default_factory=lambda: np.array([5.0]))
    
def setup_sbl_ocp(params_mpc: BLParamsSmpcShort, qmpc, qocp) -> Socp:

    # # model parameter values
    length = params_mpc.length
    Swc = params_mpc.Swc  # connate water saturation
    Sor = params_mpc.Sor  # residual oil saturation
    mu_w = params_mpc.mu_w
    mu_o = params_mpc.mu_o
    
    N_mpc = params_mpc.N_mpc
    dt = params_mpc.dt
    time_window = params_mpc.time_window
    total_time = params_mpc.total_time
    Nt = params_mpc.Nt
    nx = params_mpc.nx  # number of spatial discretization points
    
    ro = params_mpc.ro
    rwp = params_mpc.rwp
    rwi = params_mpc.rwi
    
    Ne = params_mpc.Ne
    aw = params_mpc.aw
    bw = params_mpc.bw
    
    itk = len(qmpc)
    
    bl = BuckleyLeverette(nx=nx, 
                          length=length, 
                          Swc=Swc, 
                          Sor=Sor, 
                          mu_w=mu_w, 
                          mu_o=mu_o, 
                          dt=dt, 
                          total_time=total_time,
                          ro=ro,
                          rwp=rwp,
                          rwi=rwi)

    # states
    x = ca.SX.sym("Sw", nx*Ne)  # water saturation profile along the 1D domain

    # controls
    # u = ca.SX.sym("q", params_mpc.N)  # injection rate
    u = ca.SX.sym("q", N_mpc)  # injection rate

    # simulate the state up until the iteration itk
    
    stage_cost = []
    for i in range(Ne):
        stage_cost.append(bl.stage_cost(Sw=x[i*nx:(i+1)*nx], 
                                           qt0=qmpc,
                                           qtk=u, 
                                           qocp=qocp, 
                                           itk=itk,
                                           N_mpc=N_mpc,
                                           Nt=Nt, 
                                           a=aw[:,i], 
                                           b=bw[:,i]))

    stage_constr = u + 0 
    stage_constr_lb = params_mpc.umin
    stage_constr_ub = params_mpc.umax
    stage_constr_init = params_mpc.uinit

    ocp = Socp(
        x = x,
        u = u,
        Ne = Ne,
        N_mpc = params_mpc.N_mpc,
        # dyn_expr = f_discrete,
        stage_cost_expr = stage_cost,
        stage_constr_expr = stage_constr,
        stage_constr_init = stage_constr_init,
        stage_constr_lb = stage_constr_lb,
        stage_constr_ub = stage_constr_ub,
        bl = bl
    )

    return ocp

