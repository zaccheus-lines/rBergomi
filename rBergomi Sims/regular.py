import numpy as np
import matplotlib.pyplot as plt
from rBergomi import rBergomi  # Import the rBergomi class
from utils import bsinv  # Import Black-Scholes IV inversion

# Set Parameters for the rBergomi Model
n = 100        # Steps per year
N = 30000      # Number of paths
T = 1.0        # Maximum maturity
a = -0.43      # Roughness index (H = a + 0.5)

# Instantiate rBergomi Model
rB = rBergomi(n=n, N=N, T=T, a=a)

# Fix random seed for reproducibility
np.random.seed(0)

# Generate required Brownian increments
dW1 = rB.dW1()
dW2 = rB.dW2()

# Construct the Volterra Process Y_t^H
Y = rB.Y(dW1)

# Correlate the orthogonal increments
rho = -0.90
dB = rB.dB(dW1, dW2, rho=rho)

# Construct the Variance Process V_t
xi = 0.235**2
eta = 1.9
V = rB.V(Y, xi=xi, eta=eta)

# Construct the Price Process S_t
S0 = 1  # Normalized initial price
S = rB.S(V, dB, S0=S0)

# Define Log-Strike Range for IV Calculation
k = np.arange(-0.5, 0.51, 0.01)

# Compute Call Payoffs and Implied Volatility
ST = S[:, -1][:, np.newaxis]  # Terminal prices at T
K = np.exp(k)[np.newaxis, :]  # Log-strike transformation
call_payoffs = np.maximum(ST - K, 0)  # Payoff function
call_prices = np.mean(call_payoffs, axis=0)  # Monte Carlo price expectation
implied_vols = np.vectorize(bsinv)(call_prices, 1, np.transpose(K), rB.T)  # Compute IV

# Plot Implied Volatility Curve
plt.figure(figsize=(10, 6))
plt.plot(k, implied_vols, 'r', lw=2)
plt.xlabel("Log-Strike $k$", fontsize=16)
plt.ylabel(r"$\sigma_{BS}(k, t)$", fontsize=16)
plt.title(fr"$\xi={xi:.3f}, \eta={eta:.2f}, \rho={rho:.2f}, \alpha={a:.2f}$", fontsize=16)
plt.grid(True)
plt.show()