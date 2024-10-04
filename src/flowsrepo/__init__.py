from typing import Dict, Type
from .base_flow import BaseFlow
from .earth_flow import EarthFlow
from .dragons_flow import DragonsFlow
from .satellite_flow import SatelliteFlow
from .meltingman_flow import MeltingManFlow
from .glass_flow import GlassFlow

example_registry: Dict[str, Type[BaseFlow]] = {
    "earth": EarthFlow,
    "dragons": DragonsFlow,
    "satellite": SatelliteFlow,
    "meltingman": MeltingManFlow,
    "glass": GlassFlow,
}
