import numpy as np

# Ensemble kalman filter for state estimation given 
# the input ensemble and the measurement
def enkf_step(ens_x_prior, meas_y, H, R):
    """
    Perform one step of the Ensemble Kalman Filter (EnKF) for state estimation.

    Parameters:
    ens_x_prior : np.ndarray
        Prior ensemble of states with shape (state_dim, ensemble_size).
    meas_y : np.ndarray
        Measurement vector with shape (measurement_dim,).
    H : np.ndarray
        Observation matrix with shape (measurement_dim, state_dim).
    R : np.ndarray
        Measurement noise covariance matrix with shape (measurement_dim, measurement_dim).

    Returns:
    ens_x_post : np.ndarray
        Posterior ensemble of states after assimilation with shape (state_dim, ensemble_size).
    """
    state_dim, ensemble_size = ens_x_prior.shape
    measurement_dim = meas_y.shape[0]

    # Compute the prior ensemble mean and perturbations
    ens_x_mean = np.mean(ens_x_prior, axis=1, keepdims=True)
    ens_x_perturb = ens_x_prior - ens_x_mean

    # Compute the predicted measurements for each ensemble member
    ens_y_prior = H @ ens_x_prior  # Shape: (measurement_dim, ensemble_size)
    ens_y_mean = np.mean(ens_y_prior, axis=1, keepdims=True)
    ens_y_perturb = ens_y_prior - ens_y_mean

    # Compute the covariance matrices
    P_xy = (ens_x_perturb @ ens_y_perturb.T) / (ensemble_size - 1)  # Cross-covariance
    P_yy = (ens_y_perturb @ ens_y_perturb.T) / (ensemble_size - 1) + R  # Innovation covariance

    # Compute the Kalman gain
    K = P_xy @ np.linalg.inv(P_yy)

    # Update each ensemble member
    ens_x_post = np.zeros_like(ens_x_prior)
    for i in range(ensemble_size):
        innovation = meas_y + np.random.multivariate_normal(np.zeros(measurement_dim), R) - ens_y_prior[:, i]
        ens_x_post[:, i] = ens_x_prior[:, i] + K @ innovation

    return ens_x_post

# compute observation matrix H for enkf
def compute_observation_matrix(nx, Ne, meas_indices):
    """
    Compute the observation matrix H for the Ensemble Kalman Filter (EnKF).

    Parameters:
    nx : int
        Number of spatial discretization points for each ensemble member.
    Ne : int
        Number of ensemble members.
    meas_indices : list of int
        Indices of the state vector that are observed.

    Returns:
    H : np.ndarray
        Observation matrix with shape (len(meas_indices), nx * Ne).
    """
    nmeas = len(meas_indices)
    H = np.zeros((nmeas, nx * Ne))

    for i, idx in enumerate(meas_indices):
        for j in range(Ne):
            H[i, j * nx + idx] = 1.0

    return H

## unscented kalman filter
def ukf_step(x_prior, P_prior, meas_y, H, R, alpha=1e-3, beta=2, kappa=0):
    """
    Perform one step of the Unscented Kalman Filter (UKF) for state estimation.

    Parameters:
    x_prior : np.ndarray
        Prior state estimate with shape (state_dim,).
    P_prior : np.ndarray
        Prior state covariance with shape (state_dim, state_dim).
    meas_y : np.ndarray
        Measurement vector with shape (measurement_dim,).
    H : np.ndarray
        Observation matrix with shape (measurement_dim, state_dim).
    R : np.ndarray
        Measurement noise covariance matrix with shape (measurement_dim, measurement_dim).
    alpha : float
        Spread of the sigma points.
    beta : float
        Incorporates prior knowledge of the distribution (2 is optimal for Gaussian).
    kappa : float
        Secondary scaling parameter.

    Returns:
    x_post : np.ndarray
        Posterior state estimate after assimilation with shape (state_dim,).
    P_post : np.ndarray
        Posterior state covariance after assimilation with shape (state_dim, state_dim).
    """
    state_dim = x_prior.shape[0]
    measurement_dim = meas_y.shape[0]

    # Calculate lambda
    lambda_ = alpha**2 * (state_dim + kappa) - state_dim

    # Calculate weights
    Wm = np.full(2 * state_dim + 1, 1 / (2 * (state_dim + lambda_)))
    Wc = np.full(2 * state_dim + 1, 1 / (2 * (state_dim + lambda_)))
    Wm[0] = lambda_ / (state_dim + lambda_)
    Wc[0] = lambda_ / (state_dim + lambda_) + (1 - alpha**2 + beta)

    # Generate sigma points
    sigma_points = np.zeros((2 * state_dim + 1, state_dim))
    sqrt_P = np.linalg.cholesky((state_dim + lambda_) * P_prior)
    sigma_points[0] = x_prior
    for i in range(state_dim):
        sigma_points[i + 1] = x_prior + sqrt_P[:, i]
        sigma_points[i + 1 + state_dim] = x_prior - sqrt_P[:, i]

    # Predict measurements for each sigma point
    sigma_meas = np.array([H @ sp for sp in sigma_points])
    meas_pred = np.sum(Wm[:, np.newaxis] * sigma_meas, axis=0)
    
    # Compute innovation covariance and cross-covariance
    P_yy = R.copy()
    P_xy = np.zeros((state_dim, measurement_dim))
    for i in range(2 * state_dim + 1):  
        y_diff = sigma_meas[i] - meas_pred
        x_diff = sigma_points[i] - x_prior
        P_yy += Wc[i] * np.outer(y_diff, y_diff)
        P_xy += Wc[i] * np.outer(x_diff, y_diff)
    # Compute Kalman gain
    K = P_xy @ np.linalg.inv(P_yy)  
    
    # Update state estimate and covariance
    innovation = meas_y - meas_pred
    x_post = x_prior + K @ innovation
    P_post = P_prior - K @ P_yy @ K.T   
    
    return x_post, P_post

class ESMDA():
    def __init__(self, m:np.ndarray, g_func:callable, g_obs:np.ndarray, alphas:list, cd:list) -> None:
        
        self.m = m
        self.g_func = g_func
        self.g_obs = g_obs
        self.alphas = alphas
        self.cd = cd
        self.Ne = m.shape[0]
        self.Nc = m.shape[1]
        self.Nd = g_obs.shape[0]

        assert self.g_obs.shape[0] == len(self.cd)
        # assert np.sum(self.alphas) == 1

    def calculate_covariance(self, matrix1:np.ndarray, matrix2:np.ndarray):

        def shifted_mean(matrix):
            meanM = np.mean(matrix, axis=0)
            dmatrix = matrix - meanM
            return dmatrix
        
        """
        Function responsible for calculating covariance between two matrices of ensembles. 

        matrix1: Ne x a
        matrix2: Ne x b
        """

        assert matrix1.shape[0] == matrix2.shape[0]

        dmatrix1 = shifted_mean(matrix1)
        dmatrix2 = shifted_mean(matrix2)
        
        cov = 0
        Ne = matrix1.shape[0]
        for i in range(Ne):
            cov += np.outer(dmatrix1[i,:], dmatrix2[i,:])

        cov = cov/(Ne-1)

        return cov
    
    def update_member(self, m_prior:np.ndarray, g_prior:np.ndarray, g_obs:np.ndarray, cov_mg:np.ndarray, cov_gg:np.ndarray, alpha:float, cd:np.ndarray):

        """
        Update equation per ensemble member

        m_prior     : matrix of prior parameters (Ne x a)
        g_prior     : matrix of prior output (Ne x b)
        g_obs       : vector of (perturbed) observation (1 x b)
        cov_mg      : matrix of covariance between parameters and output (a x b)
        cov_gg      : matrix of output auto-covariance (b x b)
        alpha       : es-mda sub-step
        cd          : vector of measurement error (1 x b)
        """
        K = np.matmul(cov_mg,np.linalg.pinv(cov_gg + (1/alpha)*np.diag(cd))) 
        dg = g_obs - g_prior
        m_posterior = m_prior + np.matmul(K,dg)
        
        return m_posterior
    
    def update(self, m:np.ndarray, g:np.ndarray, g_obs:np.ndarray, alpha:float, cd:np.ndarray):
    
        """
        Function responsible for updating the ensemble (prior to posterior) 
        by performing ES-MDA
        
        m       : matrix of prior parameters (Ne x a)
        g       : matrix of prior output (Ne x b)
        g_obs   : vector of measurement (1 x b)
        alpha   : es-mda sub-step
        cd      : vector of measurement error (1 x b) 
        """

        Ne = m.shape[0]

        cov_mg = self.calculate_covariance(m, g)
        cov_gg = self.calculate_covariance(g, g)

        #Main update equation
        m_post = np.zeros(m.shape)
        for j in range(Ne):
            _g_obs = np.random.normal(g_obs, np.abs(cd), g_obs.shape[0])
            m_post[j,:] = self.update_member(m[j,:], g[j,:], _g_obs, cov_mg, cov_gg, alpha, cd)
            
        return m_post
    
    def ens_g_func(self, m:np.ndarray, g_func:callable):
        """
        Calculate forward simulation for the ensemble

        m       : matrix of prior paramters (Ne x a)
        g_func  : function dynamics
        """

        g = []
        for j in range(m.shape[0]):
            _g = g_func(m[j,:])
            g.append(_g)

        return np.array(g)

    def run(self):
        """
        Function responsible for performing ES-MDA

        m       : matrix of prior parameters (Ne x a)
        g_func  : function call for the simulation
        g_obs   : vector of measurement (1 x b)
        alpha   : list of es-mda sub-step
        cd      : vector of measurement error (1 x b) 
        
        """

        m = self.m
        g = self.ens_g_func(m, self.g_func)

        self.result.m.append(m)
        self.result.g.append(g)

        assert g.shape[1] == self.g_obs.shape[0]
        
        for alpha in self.alphas:
            m = self.update(m=m, g=g, g_obs=self.g_obs, alpha=alpha, cd=self.cd)
            g = self.ens_g_func(m, self.g_func) #Calculate new forecast based on the predicted parameters

            self.result.m.append(m)
            self.result.g.append(g)

        return m, g

