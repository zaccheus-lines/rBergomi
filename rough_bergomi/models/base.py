"""
Base model class for financial models.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional

class BaseModel(ABC):
    """
    Abstract base class for financial models.
    """
    
    def __init__(self):
        """
        Initialize the base model.
        """
        self.regression_models = {}
    
    @abstractmethod
    def simulate_paths(self, n_paths: int, n_steps: int, T: float, S0: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths for the model.
        
        Parameters
        ----------
        n_paths : int
            Number of paths to simulate
        n_steps : int
            Number of time steps per path
        T : float
            Time horizon
        S0 : float, optional
            Initial price (default: 1.0)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (price_paths, variance_paths) arrays
        """
        pass
    
    @abstractmethod
    def price_european(self, 
                      S: np.ndarray, 
                      K: float, 
                      T: float, 
                      r: float = 0.0,
                      t: int = 0,
                      use_regression: bool = False) -> float:
        """
        Price European options using Monte Carlo simulation.
        
        Parameters
        ----------
        S : np.ndarray
            Price paths
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        t : int, optional
            Time step index at which to price the option
        use_regression : bool, optional
            Whether to use the pre-trained regression model
            
        Returns
        -------
        float
            Option price
        """
        pass
