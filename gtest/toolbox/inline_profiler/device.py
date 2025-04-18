from typing import Callable, Any

import gtest.libgtest_toolbox as _C_toolbox


# TODO(zobin): we need to support rocm later


class GWDevice:
    def __init__(self, gw_device : _C_toolbox.GWDevice_CUDA) -> None:
        self._gw_device = gw_device

        
    def export_metric_properties(self, metric_properties_cache_path : str):
        """
        Export the metric properties of the device.
        """
        self._gw_device.export_metric_properties(metric_properties_cache_path)


__all__ = [ 'GWDevice' ]
