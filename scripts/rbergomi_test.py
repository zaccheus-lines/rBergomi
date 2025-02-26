import numpy as np
import matplotlib.pyplot as plt
from rbergomi.fbm.fbm_process import FBMProcess
from rbergomi.models.rbergomi import rBergomi
from rbergomi.utils.utils import bsinv

vec_bsinv = np.vectorize(bsinv)

n = 100
m = 300000
T = 1.0
H = 0.4
a = H - 0.5
rho = -0.9
xi = 0.235**2 
eta = 1.9

fbm_simulator = FBMProcess(method = "cholesky", n=n, T=T, H=H, m = m)
rB = rBergomi(fbm_simulator, n=n, m=m, T=T, a=a)
V = rB.V()
dB = rB.dB()
S = rB.S(V,dB)

k = np.arange(-0.5, 0.51, 0.01)

K = np.exp(k)[np.newaxis, :]  # Strike prices as a row vector

plt.figure(figsize=(10, 6))

# Define different Hurst parameter values to compare
H_values = [0.03, 0.07, 0.1, 0.15]

for H in H_values:
    a = H - 0.5
    fbm_simulator = FBMProcess(method="cholesky", n=n, T=T, H=H, m=m)
    rB = rBergomi(fbm_simulator, n=n, m=m, T=T, a=a)
    V = rB.V()
    dB = rB.dB()
    S = rB.S(V, dB)
    
    ST = S[:, -1][:, np.newaxis]  # Terminal prices
    call_payoffs = np.maximum(ST - K, 0)
    call_prices = np.mean(call_payoffs, axis=0)[:, np.newaxis]
    implied_vols = vec_bsinv(call_prices, 1., np.transpose(K), rB.T)
    
    plt.plot(k, implied_vols, lw=2, label=f"H = {H}")

plt.xlabel(r'$K$', fontsize=16)
plt.ylabel(r'$\sigma_{BS}(k,t=%.2f)$' % rB.T, fontsize=16)
title = r'$\xi=%.3f,\ \eta=%.2f,\ \rho=%.2f$'
plt.title(title % (rB.xi, rB.eta, rB.rho), fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()