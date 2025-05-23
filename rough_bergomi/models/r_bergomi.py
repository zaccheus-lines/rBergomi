import numpy as np
from scipy.stats import norm
from typing import Tuple, Optional
from rough_bergomi.utils.utils import *
from rough_bergomi.models.base import BaseModel
from rough_bergomi.fbm.fbm_process import FBMProcess

class RoughBergomiModel(BaseModel):
    """
    Rough Bergomi model implementation.
    
    Parameters
    ----------
    xi : float
        Initial variance
    H : float
        Hurst parameter
    rho : float
        Correlation between price and variance
    eta : float
        Volatility of variance
    fbm_method : str
        Method for simulating fractional Brownian motion ('cholesky', 'davies_harte', or 'hybrid')
    n : int
        Number of paths to simulate
    """
    
    def __init__(self, 
                 xi: float = 0.235**2, 
                 H: float = 0.1, 
                 rho: float = -0.7,
                 eta: float = 1.9,
                 fbm_method: str = 'hybrid',
                 n: int = 100):
        super().__init__()
        self.xi = xi
        self.H = H
        self.rho = rho
        self.eta = eta
        self.a = H - 0.5  # Roughness parameter
        self.fbm_method = fbm_method
        self.n = n
        # Initialize the fBM simulator with the specified method
        self.fbm_simulator = FBMProcess(method=fbm_method, n=n, H=H)

       
    def simulate_paths(self, n_paths: int, n_steps: int, T: float, S0: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths of the Rough Bergomi model.
        
        Args:
            n_paths: Number of paths to simulate
            n_steps: Number of time steps per path
            T: Time horizon
            S0: Initial price (default: 1.0)
            
        Returns:
            Tuple of (price_paths, variance_paths) arrays
        """
        # Time grid
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)
        
        # Initialize arrays for paths
        variance_paths = np.zeros((n_paths, n_steps + 1))
        price_paths = np.zeros((n_paths, n_steps + 1))
        
        # Set initial values
        variance_paths[:, 0] = self.xi
        price_paths[:, 0] = S0
        
        
        # Generate fBM paths for each simulation
        for i in range(n_paths):
            # Generate fBM for this path
            fbm_path, dW1 = self.fbm_simulator.generate_fBM()

            # Calculate correlated Brownian motion for variance
            dW2= np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
            
            dB = self.rho * dW1[:,:,0] + np.sqrt(1 - self.rho**2) * dW2

            # Update variance process
            for j in range(n_steps):
                variance_paths[i, j+1] = self.xi * np.exp(
                    -0.5 * self.eta**2 * t[j+1]**(2*self.H) + 
                    self.eta * fbm_path[j]
                )
            
            # Update price process
            for j in range(n_steps):
                price_paths[i, j+1] = price_paths[i, j] * np.exp(
                    -0.5 * variance_paths[i, j] * dt + 
                    np.sqrt(variance_paths[i, j]) * dB[i, j]
                )
        
        return price_paths, variance_paths
    
    def price_european(self, 
                      S: np.ndarray, 
                      K: float, 
                      T: float, 
                      r: float = 0.0,
                      t: int = 0,
                      use_regression: bool = False) -> float:
        """
        Price European options using Monte Carlo or regression.
        """
        _, N = S.shape
        
        payoff = np.maximum(S[:, -1] - K, 0)
        return np.mean(payoff) * np.exp(-r * T)