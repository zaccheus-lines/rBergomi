from .cholesky import CholeskyFBM
from .circulant import DaviesHarteFBM
from .hybrid import HybridFBM

class FBMProcess:
    """Manages fBM simulation using different methods."""

    def __init__(self, method="cholesky", n=100, T=1.0, H=0.1, m =1):
        self.method_name = method.lower()
        
        if self.method_name == "cholesky":
            self.simulator = CholeskyFBM(n, T, H)
        elif self.method_name == "davies_harte":
            self.simulator = DaviesHarteFBM(n, T, H)
        elif self.method_name == "hybrid":
            self.simulator = HybridFBM(n, T, H)
        else:
            raise ValueError(f"Unknown method: {method}")

    def generate_path(self):
        return self.simulator.generate_fBM()