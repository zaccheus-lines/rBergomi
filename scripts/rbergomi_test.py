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
H = 0.1      # Hurst parameter
a = H - 0.5   # Alpha parameter
rho = -1    # Correlation between price and variance process
xi = 0.235**2  # Initial variance level
eta = 1.9      # Volatility of variance process

# ✅ Generate fractional Brownian motion using HybridFBM
# Generate multiple fBM paths explicitly for Monte Carlo simulation
fbm_simulator = FBMProcess(method = "hybrid", n=n, T=T, H=H, m = m)

# ✅ Initialize rBergomi model
rB = rBergomi(fbm_simulator, n=n, m=m, T=T, a=a)

#print(rB.V())

#print(rB.dB())

V = rB.V()
dB = rB.dB()

S = rB.S(V,dB)

'''plt.figure(figsize=(10,6))
for s in S:
    plt.plot(s)
plt.xlabel("Time Step")
plt.ylabel("Price Process")
plt.title("Simulated Price Paths from rBergomi Model")
plt.grid(True)
plt.show()'''