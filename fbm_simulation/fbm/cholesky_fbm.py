from .base_fbm import FBMSimulator
import numpy as np

class CholeskyFBM(FBMSimulator):
    """fBM simulation using Cholesky decomposition."""

    def __init__(self, n=100, T=1.0, H=0.4):
        super().__init__(n, T, H)
        self.Gamma = self.calc_Gamma()
        self.cholesky_matrix = np.linalg.cholesky(self.Gamma)

    def calc_Gamma(self):
        """Constructs covariance matrix for fBM using the Hurst exponent."""
        t_grid = np.arange(1, self.n + 1) * self.dt
        Gamma = np.zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                t_i, t_j = t_grid[i], t_grid[j]
                Gamma[i, j] = 0.5 * (t_i**(2 * self.H) + t_j**(2 * self.H) - abs(t_i - t_j)**(2 * self.H))

        return Gamma
    
    def cholesky_banachiewicz(self, Gamma):

        n = Gamma.shape[0]
        Sigma = np.zeros_like(Gamma)

        for i in range(n):
            for j in range(i + 1):
                sum_val = sum(Sigma[i][k] * Sigma[j][k] for k in range(j))

                if i == j:
                    Sigma[i][j] = np.sqrt(Gamma[i][i] - sum_val)
                else:
                    Sigma[i][j] = (Gamma[i][j] - sum_val) / Sigma[j][j]

        return Sigma

    def generate_fBM(self, num_samples=1):

        # Generate independent standard normal variables v ~ N(0, I)
        v = np.random.randn(self.s, num_samples)

        # Transform v into correlated samples: u = Σ v
        u = self.cholesky_matrix @ v

        return u