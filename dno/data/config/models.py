"""DNO model configuration."""
from typing import List, Optional
from zencfg import ConfigBase

class DnoModelConfig(ConfigBase):
    """DNO (Diffeomorphic Neural Operator) model configuration.

    Per-case (in_channels -> out_channels):
    # darcy:     4 -> 1   [C, X_map, Y_map, Mask] -> [U]
    #   n_modes=[16,16], hidden_channels=32, n_layers=4
    #
    # fluid:     3 -> 3   [X_map, Y_map, Re] -> [U, V, P]
    #   n_modes=[24,24], hidden_channels=32, n_layers=6
    #
    # reservoir: 39 -> 72  [X, Y, P0, S0, Src0..34] -> [P1,S1..P36,S36]
    #   n_modes=[24,24], hidden_channels=48, n_layers=6
    """
    n_modes: List[int] = [16, 16]
    hidden_channels: int = 32
    in_channels: int = 4
    out_channels: int = 1
    n_layers: int = 4
    use_grid_bias: bool = True
