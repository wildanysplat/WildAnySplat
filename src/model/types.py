from dataclasses import dataclass
from typing import Optional

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    scales: Float[Tensor, "batch gaussian 3"]
    rotations: Float[Tensor, "batch gaussian 4"]
    covariances: Optional[Float[Tensor, "batch gaussian dim dim"]] = None
    vars: Optional[Float[Tensor, "batch gaussian"]] = None  # variance per gaussian
    # levels: Float[Tensor, "batch gaussian"]
