import numpy as np
from typing import Tuple
from rough_bergomi.models.base import BaseModel


class GBMModel:
    def __init__(self, 
                 sigma: float = 0.2, 
                 mu: float = 0.0):
        self.sigma = sigma  # Volatility
        self.mu = mu        # Drift

    def simulate_paths(self, n_paths: int, n_steps: int, T: float, S0: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        dt = T / n_steps
        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = S0

        # Simulate GBM paths
        for i in range(n_steps):
            Z = np.random.normal(0, 1, n_paths)
            S[:, i + 1] = S[:, i] * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z)

        # Return (S, V), where V is constant sigma^2 for all paths and time steps
        V = np.full_like(S, self.sigma**2)
        return S, V

    def price_european(self, 
                       S: np.ndarray, 
                       K: float, 
                       T: float, 
                       r: float = 0.0) -> float:
        payoff = np.maximum(S[:, -1] - K, 0)
        return np.exp(-r * T) * np.mean(payoff)
