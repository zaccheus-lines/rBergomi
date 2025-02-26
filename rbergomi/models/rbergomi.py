import numpy as np
from rbergomi.utils.utils import *
from rbergomi.fbm.fbm_process import FBMProcess

class rBergomi:
    """
    Class for generating paths of the rBergomi model using any fractional Brownian motion (fBM) process.
    """

    def __init__(self, fbm_simulator, n=100, m=1000, T=1.00, a=-0.4, rho = -0.5):
        """
        Constructor for the rBergomi model.
        :param fbm_simulator: An instance of an fBM simulator (HybridFBM, DaviesHarteFBM, etc.)
        :param n: Number of time steps per year
        :param N: Number of simulation paths
        :param T: Maturity
        :param a: Alpha parameter (H-0.5)
        """
        self.fbm_simulator = fbm_simulator  # Accepts an fBM generator
        self.T = T # Maturity
        self.n = n # Granularity (steps per year)
        self.dt = 1.0/self.n # Step size
        self.s = int(self.n * self.T) # Steps
        self.t = np.linspace(0, self.T, self.s+1)[np.newaxis,:] # Time grid
        self.a = a # Alpha
        self.m = m # Paths
        self.rho = rho

        # Construct hybrid scheme correlation structure for kappa = 1
        self.e = np.array([0,0])
        self.c = cov(self.a, self.n)
    
    def Y(self):
        Y = np.array([self.fbm_simulator.generate_path() for _ in range(self.m)])
        #Y = self.fbm_simulator.generate_path()
        return Y

    def V(self, xi = 1.0, eta = 1.0):
        """
        rBergomi variance process.
        """
        self.xi = xi
        self.eta = eta
        Y = self.Y()

        V = xi * np.exp(eta * Y - 0.5 * eta**2 * self.t**(2 * self.a + 1))
        return V

    def dW1(self):
        """
        Obtain orthogonal increments.
        """
        return np.random.randn(self.m, self.s) * np.sqrt(self.dt)
    
    def dW2(self):
        """
        Obtain orthogonal increments.
        """
        return np.random.randn(self.m, self.s) * np.sqrt(self.dt)

    def dB(self, rho = 0.1):
        """
        Constructs correlated price Brownian increments, dB.
        """
        self.rho = rho
        dB = rho * self.dW1() + np.sqrt(1 - rho**2) * self.dW2()
        return dB
    
    def S(self, V, dB, S0 = 1):
        """
        rBergomi price process.
        """
        self.S0 = S0
        dt = self.dt

        # Construct non-anticipative Riemann increments
        increments = np.sqrt(V[:,:-1]) * dB - 0.5 * V[:,:-1] * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis = 1)

        S = np.zeros_like(V)
        S[:,0] = S0
        S[:,1:] = S0 * np.exp(integral)
        return S