from base_fbm import FBMSimulator
import numpy as np

class DaviesHarteFBM(FBMSimulator):
    """fBM simulation using the Davies–Harte method."""
    
    def __init__(self, n=100, T=1.0, H=0.4):
        super().__init__(n, T, H)

    def davies_harte_path(self):
        """
        Generates a single fractional Brownian motion path using the Davies–Harte method.
        The method returns an array of length n+1 with the first element fixed at 0.
        """
        N = self.n
        T = self.T
        H = self.H
        
        # Define the autocovariance function gamma for fBm increments.
        gamma = lambda k: 0.5 * (abs(k - 1)**(2 * H) - 2 * abs(k)**(2 * H) + abs(k + 1)**(2 * H))
        g = [gamma(k) for k in range(N)]
        # Form the circulant vector r.
        r = g + [0] + g[::-1][:N - 1]

        # Compute the FFT-based eigenvalues.
        j = np.arange(0, 2 * N)
        k_val = 2 * N - 1
        r_arr = np.array(r)
        lk = np.fft.fft(r_arr * np.exp(2 * np.pi * 1j * k_val * j / (2 * N)))[::-1]

        # Generate independent Gaussian random variables for the simulation.
        Vj = np.zeros((2 * N, 2), dtype=np.complex128)
        Vj[0, 0] = np.random.standard_normal()
        Vj[N, 0] = np.random.standard_normal()
        for i in range(1, N):
            v1 = np.random.standard_normal()
            v2 = np.random.standard_normal()
            Vj[i, 0] = v1
            Vj[i, 1] = v2
            Vj[2 * N - i, 0] = v1
            Vj[2 * N - i, 1] = v2

        # Construct the sequence wk using the computed eigenvalues and the Gaussian variables.
        wk = np.zeros(2 * N, dtype=np.complex128)
        wk[0] = np.sqrt(lk[0] / (2 * N)) * Vj[0, 0]
        wk[1:N] = np.sqrt(lk[1:N] / (4 * N)) * (Vj[1:N, 0] + 1j * Vj[1:N, 1])
        wk[N] = np.sqrt(lk[0] / (2 * N)) * Vj[N, 0]
        wk[N + 1: 2 * N] = np.sqrt(lk[N + 1: 2 * N] / (4 * N)) * (
            np.flip(Vj[1:N, 0]) - 1j * np.flip(Vj[1:N, 1])
        )

        # Apply FFT to obtain the fractional Gaussian noise and then integrate.
        Z = np.fft.fft(wk)
        fGn = Z[:N]
        fBm = np.cumsum(fGn).real * (N ** (-H)) * (T ** H)
        # Prepend the initial zero to the fBM path.
        return np.concatenate(([0], fBm))