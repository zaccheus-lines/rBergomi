import numpy as np
import matplotlib.pyplot as plt
from rbergomi.fbm.fbm_process import FBMProcess
from rbergomi.models.rbergomi import rBergomi  # Import rBergomi
from rbergomi.utils.utils import bsinv  # Black-Scholes inversion for IV calculation

# Vectorize Black-Scholes inversion function
vec_bsinv = np.vectorize(bsinv)

# ✅ Simulation parameters
n = 100     # Steps per year
m = 30    # Number of paths (Large for accuracy)
T = 1.0       # Maturity
H = 0.4      # Hurst parameter
a = H - 0.5   # Alpha parameter
rho = -0.9    # Correlation between price and variance process
xi = 0.235**2  # Initial variance level
eta = 1.9      # Volatility of variance process

# ✅ Generate fractional Brownian motion using HybridFBM
# Generate multiple fBM paths explicitly for Monte Carlo simulation
fbm_simulator = FBMProcess(method = "cholesky", n=n, T=T, H=H, m = m)

# ✅ Initialize rBergomi model
rB = rBergomi(fbm_simulator, n=n, m=m, T=T, a=a)

#print(rB.V())

#print(rB.dB())

V = rB.V()
dB = rB.dB()

S = rB.S(V,dB)

r= 0
K_values = np.linspace(0.5, 1.5, 20)  # Strike range from 0.5 to 1.5
ST = S[:, -1]  # Last column of S (final prices at maturity)
call_prices = [np.mean(np.maximum(ST - K, 0)) * np.exp(-r * T) for K in K_values]
plt.figure(figsize=(10,6))
plt.plot(K_values, call_prices, marker='o', linestyle='-')
plt.show()

plt.figure(figsize=(10,6))
for s in S:
    plt.plot(s)
plt.xlabel("Time Step")
plt.ylabel("Price Process")
plt.title("Simulated Price Paths from rBergomi Model")
plt.grid(True)
plt.show()