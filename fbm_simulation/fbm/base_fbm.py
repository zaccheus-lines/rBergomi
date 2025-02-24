import numpy as np
from abc import ABC, abstractmethod

class FBMSimulator(ABC):
    """
    Abstract base class for fBM simulation methods.
    """
    def __init__(self, n=100, T=1.0, H=0.4):
        self.n = n  # Time steps
        self.T = T  # Total time
        self.H = H  # Hurst parameter
        self.dt = T / n  # Step size
        self.t = np.linspace(0, T, n)  # Time grid

    @abstractmethod
    def generate_fBM(self, num_samples=1):
        """
        Abstract method to generate fBM paths.
        """
        pass