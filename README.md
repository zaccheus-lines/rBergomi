# Rough Bergomi Package

A Python package for implementing the Rough Bergomi model and delta hedging strategies.

## Features

- Implementation of the Rough Bergomi stochastic volatility model
- Delta hedging strategies
- Fractional Brownian motion simulation
- Visualization tools for price paths and hedging performance

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

```bash
git clone https://github.com/yourusername/rough_bergomi.git
cd rough_bergomi
pip install -e .
```

## Usage

The package provides a modular structure for working with the Rough Bergomi model:

```python
from rough_bergomi.models import RoughBergomiModel
from rough_bergomi.hedging.strategies import DeltaHedger
from rough_bergomi.utils.plotting import plot_price_paths, plot_hedging_error

# Create model instance
model = RoughBergomiModel(xi=0.235**2, H=0.1, rho=-0.7)

# Simulate paths
S, V = model.simulate_paths(T=1.0, N=252, M=1000)

# Plot results
plot_price_paths(S, T=1.0)
```

## Project Structure

```
rough_bergomi/
├── models/
│   ├── rough_bergomi_model.py  # Core Rough Bergomi model implementation
│   └── base.py                 # Base model class
├── fbm/
│   ├── fbm_process.py         # Fractional Brownian motion process
│   └── cholesky.py           # Cholesky decomposition method
├── hedging/
│   └── strategies.py         # Hedging strategies implementation
└── utils/
    ├── utils.py             # Utility functions
    └── plotting.py          # Plotting functions
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the rough Bergomi model by Christian Bayer, Peter Friz, and Jim Gatheral
- Implementation inspired by various academic papers and research in rough volatility 