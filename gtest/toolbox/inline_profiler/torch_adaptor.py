import torch
import time
import inspect
from typing import Callable, List, Any
from contextlib import contextmanager

from gtest.toolbox.inline_profiler.profiler import *

class torch_adapt:
    @staticmethod
    def profile(gw_profiler : GWProfiler = None, allow_multipass : bool = False):
        """
        [Decorator] Declare a computation graph to be profiled

        Parameters:
            gw_profiler: the GWProfiler instance to conduct profiling on the device
        """
        def decorator_wrapper(func : Callable):
            def wrapper(*args, **kwargs):
                first_pass : bool = True
                last_pass : bool = False
                nb_pass : int = 0
                ckpt_latencies : List[float] = []
                restore_latencies : List[float] = []
                gw_profiler_instance : GWProfiler = None

                # try to get gw_profiler from class intance
                # (speculate the decorated function is a member function)
                if gw_profiler is None:
                    class_instance = args[0]
                    if hasattr(class_instance, 'gw_profiler'):
                        gw_profiler_instance = getattr(class_instance, 'gw_profiler')
                else:
                    gw_profiler_instance = gw_profiler

                # profile if gwprofiler is founded
                if gw_profiler_instance != None and gw_profiler_instance.range_profile.is_session_created():
                    assert(gw_profiler_instance.is_range_profiling())

                    if allow_multipass:
                        ckpt_start = time.time()
                        gw_profiler_instance.checkpoint()
                        ckpt_latencies.append(time.time() - ckpt_start)
                    while not last_pass:
                        # we use first_pass to avoid unnecessary
                        # memory restore at the first pass
                        if not first_pass and allow_multipass:
                            restore_start = time.time()
                            gw_profiler_instance.restore()
                            restore_latencies.append(time.time() - restore_start)
                        else:
                            first_pass = False
                        gw_profiler_instance.range_profile.begin_pass()
                        gw_profiler_instance.range_profile.enable_profiling()
                        ret = func(*args, **kwargs)
                        gw_profiler_instance.range_profile.disable_profiling()
                        last_pass = gw_profiler_instance.range_profile.end_pass()
                        nb_pass += 1
                        if not allow_multipass:
                            if not last_pass:
                                print(f"warn: multipass profiling when multipass is not allowed")
                            break
                    if allow_multipass:
                        gw_profiler_instance.free_checkpoint()
                    gw_profiler_instance.range_profile.flush_data()
                    gw_profiler_instance.range_profile.set_profile_aux_info(nb_passes=nb_pass, ckpt_latencies=ckpt_latencies, restore_latencies=restore_latencies)
                else:
                    ret = func(*args, **kwargs)
                return ret

            return wrapper
        
        return decorator_wrapper


    @staticmethod
    @contextmanager
    def declare_profile_range_inline(
        class_instance : Any,
        range_name : str = "",
        do_measure_latency : bool = False
    ):
        gw_profiler_instance : GWProfiler = None

        # step 1: check GWContext profiler and range_name
        if hasattr(class_instance, 'gw_profiler'):
            gw_profiler_instance = getattr(class_instance, 'gw_profiler')
        if not gw_profiler_instance or range_name == "" or not gw_profiler_instance.range_profile.is_session_created():
            yield
            return

        assert(gw_profiler_instance.is_range_profiling())

        try:
            # NOTE(zhuobin):
            # we don't support warmup here,
            # since we can't yield multiple times

            # create cuda event if measuring latency
            if do_measure_latency:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

            # step 2: execute and profile
            gw_profiler_instance.range_profile.push_range(range_name)
            if do_measure_latency:
                start_event.record()
            yield
            if do_measure_latency:
                end_event.record()
            gw_profiler_instance.range_profile.pop_range()
            if do_measure_latency:
                end_event.synchronize()
                total_time = start_event.elapsed_time(end_event)
                gw_profiler_instance.range_profile.set_range_latency(range_name, total_time)

        finally:
            pass


    @staticmethod
    def declare_profile_range(
        gw_profiler : GWProfiler = None,
        range_name : str = "",
        do_warpup : bool = False,
        do_measure_latency : bool = False
    ):
        """
        [Decorator] Declare a profiling range

        Parameters:
            gw_profiler: the GWProfiler instance to conduct profiling on the device
            range_name: the name of the profiling range
        """

        def decorator_wrapper(*o_args, **o_kwargs):
            # first level of wrapper: accept one callable
            assert(len(o_args) == 1)
            func = o_args[0]
            assert(callable(func))

            def wrapper(*i_args, **i_kwargs):
                nonlocal do_warpup
                current_range_name = range_name

                # step 1: obtain GWContext profiler
                # try to get gw_profiler from class instance
                # (speculate the decorated function is a member function)
                gw_profiler_instance : GWProfiler = None
                if gw_profiler is None:
                    class_instance = i_args[0]
                    if hasattr(class_instance, 'gw_profiler'):
                        gw_profiler_instance = getattr(class_instance, 'gw_profiler')
                else:
                    gw_profiler_instance = gw_profiler

                # step 2: obtain profile range name
                if current_range_name == "":
                    class_instance = i_args[0]
                    if hasattr(class_instance, 'gw_profile_range_name'):
                        current_range_name = getattr(class_instance, 'gw_profile_range_name')

                # check whether to execute profiling
                do_insert_range : bool = (
                    gw_profiler_instance != None and current_range_name != "" and gw_profiler_instance.range_profile.is_session_created()
                )

                # check the mode of the profiler
                if do_insert_range:
                    assert(gw_profiler_instance.is_range_profiling())

                # create cuda event if measuring latency
                if do_insert_range and do_measure_latency:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                # step 3: warm-up
                if do_insert_range and do_warpup:
                    gw_profiler_instance.checkpoint()
                    func(*i_args, **i_kwargs)
                    gw_profiler_instance.restore(do_pop=True)

                # step 4: pre-execute
                if do_insert_range:
                    # push range
                    gw_profiler_instance.range_profile.push_range(current_range_name)
                    if do_measure_latency:
                        start_event.record()

                # step 5: execute
                list_func_ret = func(*i_args, **i_kwargs)

                # step 6: post-execute
                if do_insert_range:
                    if do_measure_latency:
                        end_event.record()

                    gw_profiler_instance.range_profile.pop_range()

                    if do_measure_latency:
                        end_event.synchronize()
                        total_time = start_event.elapsed_time(end_event)
                        gw_profiler_instance.range_profile.set_range_latency(current_range_name, total_time)

                return list_func_ret

            return wrapper
        
        return decorator_wrapper


__all__ = [ "torch_adapt" ]
