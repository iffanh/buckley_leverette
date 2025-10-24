import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

class BuckleyLeverette(object):
    def __init__(self, nx=100, length=1.0, Swc=0.2, Sor=0.2, mu_w=1.0, mu_o=5.0, dt=0.01, total_time=3.0, ro=1.0, rwp=0.1, rwi=0.1):
        self.nx = nx
        self.length = length
        self.dx = length / (nx - 1)
        self.x = np.linspace(0, length, nx)
        self.Swc = Swc  # connate water saturation
        self.Sor = Sor  # residual oil saturation
        self.mu_w = mu_w
        self.mu_o = mu_o
        self.dt = dt
        self.total_time = total_time
        self.nt = int(np.floor(total_time/dt))
        self.t = np.linspace(0, total_time, self.nt)
        # self.Swinit = np.ones(nx) * Swc  # initial water saturation
        # self.Swinit[0] = 1.0  # inject water at left boundary
        
        self.ro = ro
        self.rwp = rwp
        self.rwi = rwi

    def traditional_fractional_flow(self, Sw):
        # Corey-type relative permeability
        Swn = (Sw - self.Swc) / (1 - self.Swc - self.Sor)
        Swn = np.clip(Swn, 0, 1)
        krw = Swn ** 2
        kro = (1 - Swn) ** 2
        fw = krw / self.mu_w / (krw / self.mu_w + kro / self.mu_o)
        return fw
    
    def fractional_flow(self, Sw, a=20.0, b=5.0):
        fw = 1/(1 + (self.mu_w/self.mu_o) * a*np.exp(-b*Sw))
        return ca.vertcat(fw)

    def simulate(self, Sw, qt, a, b):
        Sw_end = Sw + 0
        nsteps = int(self.total_time / self.dt)
        for i in range(nsteps):
            Sw_end = self.simulate_at_k(Sw_end, qt[i], a, b)
        return Sw_end
            
    def simulate_at_k(self, Sw, qtk, a, b):
        
        Sw_end = ca.vertcat(Sw) + 0
        fw = self.fractional_flow(Sw_end, a, b)
        fw_right = casadi_roll_left(fw)
        
        # Upwind scheme
        dSw = -(self.dt / self.dx) * qtk *(fw_right - fw)
        Sw_end[1:] += dSw[0:-1]
        Sw_end = casadi_clip(Sw_end, self.Swc, 1.0)
        return Sw_end
            
    def simulate_w_plot(self, Sw, qt):
        nsteps = int(self.total_time / self.dt)
        for i in range(nsteps):
            
            Sw = self.simulate_at_k(Sw, qt[i])
            
            if i % (nsteps // 10) == 0:
                plt.plot(self.x, Sw, label=f'Time={i*self.dt:.2f}')

        plt.xlabel('Distance')
        plt.ylabel('Water Saturation')
        plt.title('Buckley-Leverette Waterflood')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    def cost_function(self, qt, Sw, gamma=0.99):
        nsteps = int(self.total_time / self.dt)
        list_npv = []
        npv = 0.0
        for i in range(nsteps):
            
            Sw = self.simulate_at_k(Sw, qt[i])
            cashflow = self.stage_cost(Sw, qt[i])
            npv += cashflow*gamma**(i*self.dt)
            list_npv.append(npv)
            
        return list_npv
            
    def stage_cost(self, Sw, qt0, qtk, qocp, itk, N_mpc, Nt, a, b):
        
        ## to calculate the stage cost at time itk, 
        ## we need to simulate the state from time 0 to itk using the previous solutions from MPC
        ## then simulate the state from itk to itk+n using the current control
        ## then simulate the state from itk+n to the total_time using the terminal control found by OCP 
        
        # Swinit : initial state at time 0
        # qt0 : control input from time 0 to itk-1 (previous solutions from MPC) 
        # qtk : a vector of control inputs from time itk to itk+n (variable to solve in MPC)
        # qocp : a vector of control inputs from time itk+n to total time (solutions from OCP)
        # itk : current iterations, also the stage of the MPC
        # N_mpc : the window length of the MPC
        # N : the total window length of the OCP
        
        cost = 0.0
        
        q_seq = []
            
        # print(f"initial cost", cost)
        
        for i in range(itk):
            cost += self._stage_cost(Sw, qt0[i])*(0.99**(i) )
            Sw = self.simulate_at_k(Sw, qt0[i], a, b) # we use qtk[i] as the control input from time 0 to itk-1
            # cost += self._stage_cost(Sw, qt0[i])*(0.99**(i) )
            q_seq.append(qt0[i])    
            
        for i in range(N_mpc):
            if itk + i < Nt:
                cost += self._stage_cost(Sw, qtk[i])*(0.99**(i+itk))
                Sw = self.simulate_at_k(Sw, qtk[i], a, b) # we use qtk[i] as the control input from time itk to itk+N-1
                # cost += self._stage_cost(Sw, qtk[i])*(0.99**(i+itk))
                q_seq.append(qtk[i])
                
        for i in range(Nt - (itk + N_mpc)):
            if itk + N_mpc + i < Nt:
                cost += self._stage_cost(Sw, qocp[i])*(0.99**(i+itk+N_mpc))
                Sw = self.simulate_at_k(Sw, qocp[i], a, b) # we use qocp[i] as the control input from time itk+N to total_time
                q_seq.append(qocp[i])
        
        print(f"Control sequence used in cost calculation: {q_seq}")
                
        return cost
    
    def _stage_cost(self, Swk, qtk):
    
        fwN = self.fractional_flow(Swk[-1])
        cost = -qtk*((1-fwN)*self.ro - self.rwp*fwN - ppf_approx(qtk))
        
        return cost

        
    def plot(self):
        plt.plot(self.x, self.Sw)
        plt.xlabel('Distance')
        plt.ylabel('Water Saturation')
        plt.title('Buckley-Leverette Waterflood')
        plt.grid(True)
        plt.show()
        
def casadi_roll_left(arr):
    # arr is a CasADi MX or SX vector
    return ca.vertcat(arr[1:], arr[0])

def casadi_clip(arr, min_val, max_val):
    return ca.fmin(ca.fmax(arr, min_val), max_val)

def ppf_approx(u, alpha=0.055): # polynomial approximation of the energy price function due to water injection by pumps
    return (3.15 - (- 1.1 * u**3 + 0.4 * u**2 + 2.0 * u))*alpha