import numpy as np
import matplotlib.pyplot as plt

def plot_paths(S, V, t, n_plot=25, title_prefix=''):
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, S[:n_plot].T, alpha=0.3, color='navy')
    plt.title(f'{title_prefix}Price Paths (First {n_plot})')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, V[:n_plot].T, alpha=0.3, color='darkred')
    plt.title(f'{title_prefix}Variance Paths (First {n_plot})')
    plt.xlabel('Time')
    plt.ylabel('Variance')
    plt.grid(True)

    plt.tight_layout()
    plt.show()