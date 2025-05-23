from .base_fbm import FBMSimulator
import numpy as np
from ..utils.utils import *

class HybridFBM(FBMSimulator):
    """
    Hybrid fBM simulation using the Volterra process for fractional noise generation.
    """

    def __init__(self, n=100, T=1.0, H=0.1):
        super().__init__(n, T, H)    
        self.t = np.linspace(0, self.T, 1 + self.s) # Time grid

    def dW1(self):
        """
        Produces random numbers for variance process with required
        covariance structure.
        """
        rng = np.random.multivariate_normal
        return rng(self.e, self.c, (self.m, self.s))  # Ensure correct shape
    
    def Y(self, dW):
        """
        Constructs Volterra process from appropriately
        correlated 2d Brownian increments.
        """
        Y1 = np.zeros((self.m, 1 + self.s)) # Exact integrals
        Y2 = np.zeros((self.m, 1 + self.s)) # Riemann sums

        # Construct Y1 through exact integral
        for i in np.arange(1, 1 + self.s, 1):
            Y1[:,i] = dW[:,i-1,1] # Assumes kappa = 1

        G = np.zeros(1 + self.s) # Gamma
        for k in np.arange(2, 1 + self.s, 1):
            G[k] = g(b(k, self.a)/self.n, self.a)

        X = dW[:,:,0] # Xi

        # Initialise convolution result, GX
        GX = np.zeros((self.m, len(X[0,:]) + len(G) - 1))

        for i in range(self.m):
            GX[i,:] = np.convolve(G, X[i,:])
        Y2 = GX[:,:1 + self.s]
        Y = np.sqrt(2 * self.a + 1) * (Y1 + Y2)
        return Y

    def generate_fBM(self):
        dW = self.dW1()
        fBm = self.Y(dW)
        return fBm[0], dW

