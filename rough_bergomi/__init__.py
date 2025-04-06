"""
Rough Bergomi package for stochastic volatility model implementation and delta hedging.
"""

from .models.r_bergomi import RoughBergomiModel
from .hedging.strategies import DeepHedger
from .utils.utils import bsinv
from .utils.plotting import (
    plot_price_paths,
    plot_hedging_error,
    plot_greeks,
    plot_hedging_metrics,
    plot_hedging_error_distribution
)

__version__ = '0.1.0'
