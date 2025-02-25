import numpy as np
from abc import ABC, abstractmethod
from ..utils.utils import *

class FBMSimulator(ABC):
    """
    Abstract base class for fBM simulation methods.
    """
    def __init__(self, n=100, T=1.0, H=0.1, m = 1):
        self.n = n  # Time steps
        self.T = T  # Total time
        self.H = H  # Hurst parameter
        self.dt = T / n  # Step size
        self.s = int(self.n * self.T)  # Total steps
        self.t = np.linspace(0, self.T, self.s) # Time grid
        self.m = m
        self.a = H - 0.5  # Roughness parameter
        self.e = np.array([0, 0])
        self.c = cov(self.a, self.n)
        

    @abstractmethod
    def generate_fBM(self, num_samples=1):
        """
        Abstract method to generate fBM paths.
        """
        pass