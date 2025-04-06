import numpy as np
from typing import Tuple
from rough_bergomi.models.base import BaseModel

class HestonModel(BaseModel):
    """
    Heston model implementation.
    
    Parameters
    ----------
    v0 : float
        Initial variance
    kappa : float
        Rate of mean reversion
    theta : float
        Long-term variance
    sigma : float
        Volatility of variance (vol of vol)
    rho : float
        Correlation between asset and variance processes
    """

    def __init__(self, v0: float = 0.235**2, kappa: float = 2.0, theta: float = 0.04, 
                 sigma: float = 0.5, rho: float = -0.7):
        super().__init__()
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

    def simulate_paths(self, n_paths: int, n_steps: int, T: float, S0: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        dt = T / n_steps

        prices = np.zeros((n_paths, n_steps + 1))
        variances = np.zeros((n_paths, n_steps + 1))

        prices[:, 0] = S0
        variances[:, 0] = self.v0

        for i in range(n_paths):
            for t in range(n_steps):
                z1 = np.random.normal(0, 1)
                z2 = np.random.normal(0, 1)
                
                dw1 = np.sqrt(dt) * z1
                dw2 = np.sqrt(dt) * (self.rho * z1 + np.sqrt(1 - self.rho**2) * z2)

                v_t = max(variances[i, t], 0)
                variances[i, t + 1] = (
                    variances[i, t] + self.kappa * (self.theta - v_t) * dt +
                    self.sigma * np.sqrt(v_t) * dw2
                )
                variances[i, t + 1] = max(variances[i, t + 1], 0)  # ensure positivity

                prices[i, t + 1] = prices[i, t] * np.exp(
                    -0.5 * v_t * dt + np.sqrt(v_t) * dw1
                )

        return prices, variances

    def price_european(self, S: np.ndarray, K: float, T: float, r: float = 0.0) -> float:
        payoff = np.maximum(S[:, -1] - K, 0)
        return np.mean(payoff) * np.exp(-r * T)