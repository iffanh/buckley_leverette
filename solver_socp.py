from dataclasses import dataclass, field
import numpy as np
import casadi as ca

@dataclass
class Socp:
    x: ca.SX  # state symbol
    u: ca.SX  # control symbol
    u_prev: ca.SX  # previous control symbol for control change penalty
    Ne: int # number of perturbation
    N: int  # OCP horizon
    dyn_expr: ca.SX  # dynamics expression
    stage_cost_expr: ca.SX  # stage cost expression
    stage_constr_expr: ca.SX = ca.SX()  # stage constraint expression
    stage_constr_init: ca.SX = ca.SX()  # initial condition for stage constraint
    stage_constr_lb: np.ndarray = field(default_factory=lambda: np.array([]))   # lower bound on stage constraint
    stage_constr_ub: np.ndarray = field(default_factory=lambda: np.array([]))   # upper bound on stage constraint

    @property
    def nx(self) -> int:
        return self.x.shape[0]
    @property
    def nu(self) -> int:
        return self.u.shape[0]


class SolverOcp():
    def __init__(self, problem: Socp, itk: int = 0):
        self.problem = problem
        self.itk = itk  # iteration counter
        self._build_solver()
        self.success = False
        self._sol = None
        self._init_guess = 0
    

    def _build_solver(self) -> None:

        # dynamics and cost functions
        dyn_func = ca.Function("dyn", [self.problem.x, self.problem.u], [*self.problem.dyn_expr])
        stage_cost_func = ca.Function("stage_cost", [self.problem.x, self.problem.u, self.problem.u_prev], [*self.problem.stage_cost_expr])
        stage_constr_func = ca.Function("stage_constr", [self.problem.x, self.problem.u], [self.problem.stage_constr_expr])

        # build OCP
        # decision variables
        X = ca.SX.sym("X", self.problem.nx, self.problem.N+1)
        U = ca.SX.sym("U", self.problem.nu, self.problem.N)
        # vector of all decision variables ordered as [x0, u0, x1, u1, ..., xN]
        decvar = ca.veccat(X[:,0], ca.vertcat(U, X[:,1:]))
        # to extract state and control trajectories in nice shape from decvar
        self._extract_traj = ca.Function("extract_traj", [decvar], [X, U])
        self._traj_to_vec = ca.Function("traj_to_vec", [X, U], [decvar])

        # objective
        obj = 0

        constr = []      # constraint expressions
        constr_lb = []    # lower bounds on constraints
        constr_ub = []    # upper bounds on constraints

        # iterate through stages
        for k in range(self.problem.N):
            
            # if k == 0:
            #     obj += stage_cost_func(X[:,k], U[:,k], U[:,k])*0.99**(k + self.itk)  # no previous control, so we pass current control as previous
            # else:
            #     obj += stage_cost_func(X[:,k], U[:,k], U[:,k-1])*0.99**(k + self.itk)
                
            if k == 0:
                costs = stage_cost_func(X[:,k], U[:,k], U[:,k])
                  # no previous control, so we pass current control as previous
            else:
                costs = stage_cost_func(X[:,k], U[:,k], U[:,k-1])
                
            obj += ca.sum(ca.vertcat(*[cost*0.99**(k + self.itk) for cost in costs]))/self.problem.Ne
                
            # TODO add dynamics constraint
            constr.append(ca.vertcat(*dyn_func(X[:,k], U[:,k])) - X[:,k+1])
            constr_lb.append(np.zeros((self.problem.nx,)))
            constr_ub.append(np.zeros((self.problem.nx,)))

            # TODO: stage constraints
            if k == 0:
                constr.append(stage_constr_func(X[:,k], U[:,k]))
                constr_lb.append([self.problem.stage_constr_init])
                constr_ub.append([self.problem.stage_constr_init])
            else:
                constr.append(stage_constr_func(X[:,k], U[:,k]))
                constr_lb.append(self.problem.stage_constr_lb)
                constr_ub.append(self.problem.stage_constr_ub)

        constr = ca.veccat(*constr)

        nlp = {"x": decvar, "f": obj, "g": constr}
        opts = {'ipopt.print_level':0, 'print_time':0}
        self._solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        self._decvar = decvar
        self._X = X
        self._U = U
        self._constr_lb = np.concatenate(constr_lb)
        self._constr_ub = np.concatenate(constr_ub)
        self._lbx = -np.inf * np.ones(decvar.shape[0])
        self._ubx = np.inf * np.ones(decvar.shape[0])

    def solve(self, x0) -> np.ndarray:
        '''
        Solve the OCP for initial state x0.
        Returns state and control trajectory as arrays of shape (nx, N+1) and (nu, N).
        '''

        # TODO: set initial state constraint
        # (we can set constraints which are directly on a single variable (box constraints) via lbx, ubx)
        # (We set an equality constraint by using an identical lower and upper bound)    
        self._lbx[:self.problem.nx] = x0
        self._ubx[:self.problem.nx] = x0

        self._sol = self._solver(x0=self._init_guess, lbx=self._lbx, ubx=self._ubx, lbg=self._constr_lb, ubg=self._constr_ub)

        Xopt, Uopt = self._extract_traj(self._sol["x"])
        return Xopt.full(), Uopt.full()

    def set_initial_guess(self, X, U) -> None:
        '''
        Set the initial guess for the optimization problem.
        X: state trajectory of shape (nx, N+1)
        U: control trajectory of shape (nu, N)
        '''
        
        self._init_control = [U[0]]  # initial control for initial constraint
        self._init_guess = self._traj_to_vec(X, U).full().flatten()
