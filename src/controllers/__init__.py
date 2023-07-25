REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_continous_controller import ContinuousMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["continuous_mac"] = ContinuousMAC




