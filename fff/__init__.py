try:
    import os

    if "GEOMSTATS_BACKEND" in os.environ and os.environ["GEOMSTATS_BACKEND"] != "pytorch":
        raise ValueError(f"GEOMSTATS_BACKEND must be unset or pytorch, but is "
                         f"{os.environ['GEOMSTATS_BACKEND']!r}")

    os.environ["GEOMSTATS_BACKEND"] = "pytorch"
    import geomstats.backend as gs

    # Why on earth would the import of a library cause a side effect?
    gs.set_default_dtype("float32")
except ImportError:
    pass


import fff.loss as loss
import fff.data as data
import fff.model as model
from .fff import FreeFormFlowHParams, FreeFormFlow
from .fif import FreeFormInjectiveFlowHParams, FreeFormInjectiveFlow
from .m_fff import ManifoldFreeFormFlowHParams, ManifoldFreeFormFlow
