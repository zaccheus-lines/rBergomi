import matplotlib.pyplot as plt
from rbergomi.fbm.fbm_process import FBMProcess

# Define methods to compare
methods = ["cholesky", "davies_harte", "hybrid"]

# Create a figure for plotting
plt.figure(figsize=(10, 5))

# Loop through each method and plot its fBM path
for method in methods:
    # Initialize the model with the selected method
    fbm_model = FBMProcess(method=method, n=100, T=1.0, H=0.4, m=1)
    #print(fbm_model.simulator.s)
    #print(fbm_model.simulator.t)

    # Generate fBM paths
    path = fbm_model.generate_path()
    #print(paths)

    # Plot the paths
    plt.plot(fbm_model.simulator.t, path, label=f'{method.replace("_", " ").capitalize()} Method')

# Plot settings
plt.xlabel("Time")
plt.ylabel("fBM Value")
plt.title("Comparison of fBM Paths: Cholesky vs Davies-Harte vs Hybrid")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()