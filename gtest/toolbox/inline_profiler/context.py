from typing import List, Dict, Any

import gtest.libgtest_toolbox as _C_toolbox
from gtest.toolbox.inline_profiler.profiler import *
from gtest.toolbox.inline_profiler.device import *



class GWContext:
    """
    GWContext class is a singleton that provides access to the GWContext API. It allows you to create and manage profiling sessions for NVIDIA GPUs.
    """
    _instance = None
    _si_gtest = None

    def __new__(self, *args, **kwargs):
        if not self._instance:
            self._instance = super(GWContext, self).__new__(self)
            assert(self._si_gtest == None)

            if "lazy_init_device" not in kwargs:
                kwargs["lazy_init_device"] = True

            self._si_gtest = _C_toolbox.GWContext_CUDA(kwargs["lazy_init_device"])
        return self._instance


    def create_profiler(self, deviceId : int, metricNames : List[str], profiler_mode : str = "") -> GWProfiler:
        """
        Creates a new profiling session for the specified NVIDIA GPU device.
        
        Parameters:
            deviceId (int): The ID of the NVIDIA GPU device to profile.
            metricNames (list[str]): A list of metric names to collect during profiling.
        """
        _gw_profiler = self._si_gtest.create_profiler(deviceId, metricNames, profiler_mode)
        return GWProfiler(_gw_profiler, profiler_mode)


    def destory_profiler(self, profiler : GWProfiler):
        """
        Destroys the specified profiling session.
        
        Parameters:
            profiler (GWProfiler): The profiler to destroy.
        """
        self._si_gtest.destory_profiler(profiler._gw_profiler)


    def get_clock_freq(self, device_id : int) -> Dict[str,int]:
        """
        Obtain the clock id of different domain

        Parameters:
            device_id (int): index of the device to be monitored
        """
        return self._si_gtest.get_clock_freq(device_id)
    

    def get_devices(self) -> Dict[int, GWDevice]:
        """
        Returns a list of all NVIDIA GPU devices available on the system.
        """
        _gw_devices : Dict[int, Any] = self._si_gtest.get_devices()
        map_gw_device : Dict[int, GWDevice] = {} 
        for deivce_id, _gw_device in _gw_devices.items():
            map_gw_device[deivce_id] = GWDevice(_gw_device)
        return map_gw_device


__all__ = [ 'GWContext' ]
