import numpy as np
from typing import Literal, Dict, List

import gtest.libgtest_toolbox as _C_toolbox


GW_CUPTI_REPLAY_MODE_AUTO = 0
GW_CUPTI_REPLAY_MODE_USER = 1
GW_CUPTI_RANGE_MODE_KERNEL = 0
GW_CUPTI_RANGE_MODE_USER = 1


class GWProfiler:
    def __init__(self, gw_profiler : _C_toolbox.GWProfiler_CUDA, profiler_mode : Literal['range', 'pm']) -> None:
        self._gw_profiler = gw_profiler
        self._profiler_mode = profiler_mode
        self.range_profile : 'GWProfiler'._range_profile = None
        self.pm_sampling : 'GWProfiler'._pm_sampling = None

        if profiler_mode == "range" or profiler_mode == "":
            self._profiler_mode = "range"
            self.range_profile = GWProfiler._range_profile(gw_profiler)
        elif profiler_mode == "pm":
            self.pm_sampling = GWProfiler._pm_sampling(gw_profiler)


    def is_range_profiling(self):
        return self._profiler_mode == "range"


    def is_pm_sampling(self):
        return self._profiler_mode == "pm"


    """
    Range Profile APIs
    """
    class _range_profile:
        def __init__(self, gw_profiler : _C_toolbox.GWProfiler_CUDA) -> None:
            self._gw_profiler = gw_profiler

            # range info
            self._dict_range_latencies : Dict[str, List] = {}   # {range_name, [latency]}

            # profile aux info
            self._nb_passes : int = 0
            self._list_ckpt_latencies : List[float] = []
            self._list_restore_latencies : List[float] = []


        def start_session(
            self,
            max_launches_per_pass : int = 512,
            max_ranges_per_pass : int = 64,
            cupti_profile_range_mode : int = GW_CUPTI_REPLAY_MODE_USER,
            cupti_profile_reply_mode : int = GW_CUPTI_RANGE_MODE_USER,
            cupti_profile_min_nesting_level : int = 1,
            cupti_profile_num_nesting_levels : int = 1,
            cupti_profile_target_nesting_levels : int = 1
        ):
            """
            Start a session with the given parameters.

            Parameters:
                max_launches_per_pass (int): Maximum number of launches per pass.
                max_ranges_per_pass (int): Maximum number of ranges per pass.
                cupti_profile_range_mode (int): CUPTI profile range mode.
                cupti_profile_reply_mode (int): CUPTI profile reply mode.
                cupti_profile_min_nesting_level (int): Minimum nesting level.
                cupti_profile_num_nesting_levels (int): Number of nesting levels.
                cupti_profile_target_nesting_levels (int): Target nesting levels.
        
            Note: this api would switch to target device context
            """

            assert(
                max_launches_per_pass > 0 and
                max_ranges_per_pass > 0 and
                cupti_profile_min_nesting_level > 0 and
                cupti_profile_num_nesting_levels > 0 and
                cupti_profile_target_nesting_levels > 0
            )

            assert(
                cupti_profile_range_mode == GW_CUPTI_REPLAY_MODE_AUTO or
                cupti_profile_range_mode == GW_CUPTI_REPLAY_MODE_USER
            )

            assert(
                cupti_profile_reply_mode == GW_CUPTI_RANGE_MODE_KERNEL or
                cupti_profile_reply_mode == GW_CUPTI_RANGE_MODE_USER
            )

            self._gw_profiler.RangeProfile_start_session(
                max_launches_per_pass,
                max_ranges_per_pass,
                cupti_profile_range_mode,
                cupti_profile_reply_mode,
                cupti_profile_min_nesting_level,
                cupti_profile_num_nesting_levels,
                cupti_profile_target_nesting_levels
            )


        def destory_session(self):
            """
            Destroys the session.
            """
            self._gw_profiler.RangeProfile_destory_session()

        
        def is_session_created(self):
            """
            Returns whether a session is created
            """
            return self._gw_profiler.RangeProfile_is_session_created()


        def begin_pass(self):
            """
            Begins a new pass.
            """
            self._gw_profiler.RangeProfile_begin_pass()


        def end_pass(self) -> bool:
            """
            Ends the current pass and returns True if all passes have been completed
            """
            return self._gw_profiler.RangeProfile_end_pass()
        

        def enable_profiling(self):
            """
            Enables profiling
            """
            self._gw_profiler.RangeProfile_enable_profiling()


        def disable_profiling(self):
            """
            Disables profiling
            """
            self._gw_profiler.RangeProfile_disable_profiling()


        def push_range(self, range_name : str):
            """
            Pushes a new range
            """
            self._gw_profiler.RangeProfile_push_range(range_name)


        def pop_range(self):
            """
            Pop a range
            """
            self._gw_profiler.RangeProfile_pop_range()


        def flush_data(self):
            """
            Flushes the counter data to image        
            """
            self._gw_profiler.RangeProfile_flush_data()


        def flush_data(self):
            """
            Flushes the counter data to image        
            """
            self._gw_profiler.RangeProfile_flush_data()


        def set_profile_aux_info(self, nb_passes : int = 1, ckpt_latencies : List[float] = [], restore_latencies : List[float] = []):
            self._nb_passes = nb_passes
            self._ckpt_latencies = ckpt_latencies
            self._restore_latencies = restore_latencies


        def set_range_latency(self, range_name : str, latency : float):
            if range_name not in self._dict_range_latencies.keys():
                self._dict_range_latencies[range_name] = [latency]
            else:
                self._dict_range_latencies[range_name].append(latency)


        def get_metrics(self):
            return self._gw_profiler.RangeProfile_get_metrics()


    """
    PM Sampling APIs
    """
    class _pm_sampling:
        def __init__(self, gw_profiler : _C_toolbox.GWProfiler_CUDA) -> None:
            self._gw_profiler = gw_profiler

        def enable_profiling(self):
            self._gw_profiler.PmSampling_enable_profiling()

        
        def disable_profiling(self):
            self._gw_profiler.PmSampling_disable_profiling()


        def set_config(
            self,
            hw_buf_size : int = 512 * 1024 * 1024,
            sampling_interval : int = 100000,
            max_samples : int = 10000
        ):
            self._gw_profiler.PmSampling_set_config(hw_buf_size, sampling_interval, max_samples)


        def start_profiling(self):
            self._gw_profiler.PmSampling_start_profiling()

        
        def stop_profiling(self):
            self._gw_profiler.PmSampling_stop_profiling()


        def get_metrics(self):
            return self._gw_profiler.PmSampling_get_metrics()


    """
    Common APIs
    """
    def checkpoint(self):
        """
        Checkpoint the memory state of GPU
        """
        self._gw_profiler.checkpoint()


    def restore(self, do_pop : bool = False):
        """
        Restore the memory state of GPU
        """
        self._gw_profiler.restore(do_pop)


    def free_checkpoint(self):
        """
        Free the latest checkpoint
        """
        self._gw_profiler.free_checkpoint()


    def reset_counter_data(self):
        """
        Reset the counter data
        """
        self._gw_profiler.reset_counter_data()

        self._nb_passes = 0
        self._dict_range_latencies.clear()
        self._list_ckpt_latencies.clear()
        self._list_restore_latencies.clear()


__all__ = [
    'GWProfiler',
    'GW_CUPTI_REPLAY_MODE_AUTO',
    'GW_CUPTI_REPLAY_MODE_USER',
    'GW_CUPTI_RANGE_MODE_KERNEL',
    'GW_CUPTI_RANGE_MODE_USER'
]
