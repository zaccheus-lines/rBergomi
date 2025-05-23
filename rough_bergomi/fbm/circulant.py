from .base_fbm import FBMSimulator
import numpy as np

class DaviesHarteFBM(FBMSimulator):
    """fBM simulation using the Davies–Harte method."""
    
    def __init__(self, n=100, T=1.0, H=0.4):
        super().__init__(n, T, H)

    def generate_fBM(self):
        """
        Generates a single fractional Brownian motion path using the Davies–Harte method.
        The method returns an array of length n+1 with the first element fixed at 0.
        """

        # Define the autocovariance function gamma for fBm increments.
        gamma = lambda k: 0.5 * (abs(k - 1)**(2 * self.H) - 2 * abs(k)**(2 * self.H) + abs(k + 1)**(2 * self.H))
        g = [gamma(k) for k in range(self.n)]
        # Form the circulant vector r.
        r = g + [0] + g[::-1][:self.n - 1]

        # Compute the FFT-based eigenvalues.
        j = np.arange(0, 2 * self.n)
        k_val = 2 * self.n - 1
        r_arr = np.array(r)
        lk = np.fft.fft(r_arr * np.exp(2 * np.pi * 1j * k_val * j / (2 * self.n)))[::-1]

        # Generate independent Gaussian random variables for the simulation.
        Vj = np.zeros((2 * self.n, 2), dtype=np.complex128)
        Vj[0, 0] = np.random.standard_normal()
        Vj[self.n, 0] = np.random.standard_normal()
        for i in range(1, self.n):
            v1 = np.random.standard_normal()
            v2 = np.random.standard_normal()
            Vj[i, 0] = v1
            Vj[i, 1] = v2
            Vj[2 * self.n - i, 0] = v1
            Vj[2 * self.n - i, 1] = v2

        # Construct the sequence wk using the computed eigenvalues and the Gaussian variables.
        wk = np.zeros(2 * self.n, dtype=np.complex128)
        wk[0] = np.sqrt(lk[0] / (2 * self.n)) * Vj[0, 0]
        wk[1:self.n] = np.sqrt(lk[1:self.n] / (4 * self.n)) * (Vj[1:self.n, 0] + 1j * Vj[1:self.n, 1])
        wk[self.n] = np.sqrt(lk[0] / (2 * self.n)) * Vj[self.n, 0]
        wk[self.n + 1: 2 * self.n] = np.sqrt(lk[self.n + 1: 2 * self.n] / (4 * self.n)) * (
            np.flip(Vj[1:self.n, 0]) - 1j * np.flip(Vj[1:self.n, 1])
        )

        # Apply FFT to obtain the fractional Gaussian noise and then integrate.
        Z = np.fft.fft(wk)
        fGn = Z[:self.n]
        fBm = np.cumsum(fGn).real * (self.n ** (-self.H)) * (self.T ** self.H)
        # Prepend the initial zero to the fBM path.
        fBm = np.insert(fBm, 0, 0.0)  # Inserts 0 at the start, making it shape (n+1,)

        return fBm, None