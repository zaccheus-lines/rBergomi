import matplotlib.pyplot as plt
from fbm.fbm_process import FBMProcess

# Choose method ("cholesky" or "circulant")
fbm_model = FBMProcess(method="cholesky", n=100, T=1.0, H=0.4)

# Generate fBM paths
paths = fbm_model.generate_paths(num_samples=1)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(fbm_model.simulator.t, paths, label=f'fBM path (H={fbm_model.simulator.H:.2f})', color='blue')
plt.xlabel("Time")
plt.ylabel("fBM Value")
plt.title(f"Simulated fBM Path Using {fbm_model.method_name.capitalize()} Method")
plt.legend()
plt.grid(True)
plt.show()