"""
Import bridge for original HRM components
"""

import sys
import os

# Add HRM submodule to path
hrm_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "HRM")
if hrm_path not in sys.path:
    sys.path.append(hrm_path)

# Import their components
from models.common import trunc_normal_init_  # noqa: E402
from models.layers import (  # noqa: E402
    rms_norm,
    SwiGLU,
    Attention,
    RotaryEmbedding,
    CastedEmbedding,
    CastedLinear,
    CosSin,
)
from models.sparse_embedding import CastedSparseEmbedding  # noqa: E402

__all__ = [
    "trunc_normal_init_",
    "rms_norm",
    "SwiGLU",
    "Attention",
    "RotaryEmbedding",
    "CastedEmbedding",
    "CastedLinear",
    "CosSin",
    "CastedSparseEmbedding",
]
