import matplotlib.pyplot as plt
from rbergomi.fbm.fbm_process import FBMProcess

methods = ["cholesky", "davies_harte", "hybrid"]

plt.figure(figsize=(10, 5))

for method in methods:
    fbm_model = FBMProcess(method=method, n=100, T=1.0, H=0.4, m=1)
    path = fbm_model.generate_path()
    plt.plot(fbm_model.simulator.t, path, label=f'{method.replace("_", " ").capitalize()} Method')

plt.xlabel("Time")
plt.ylabel("fBM Value")
plt.title("Comparison of fBM Paths: Cholesky vs Davies-Harte vs Hybrid")
plt.legend()
plt.grid(True)

plt.show()