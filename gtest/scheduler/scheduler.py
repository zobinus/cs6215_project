import os
import sys
import ast
import abc
import subprocess
import importlib.util
from typing import Literal, Callable, List, Any
from loguru import logger
from gtest.watchscript import *
import gtest.config as config


import gtest.libgtest_scheduler as _C_scheduler


current_file_path = os.path.abspath(__file__)
gtest_dir_path = os.path.dirname(os.path.dirname(current_file_path))


class GWScheduler:

    _gtest_scheduler = None

    def __new__(
        cls,
        backend : Literal["cuda", "rocm"],
        watchscript_path : str = "",
        world_size : int = 1,
        visual : bool = False,
        command : List[str] = ""
    ):
        # create single instance if not created
        if config.SI_gw_scheduler == None:
            config.SI_gw_scheduler = super(GWScheduler, cls).__new__(cls)
            assert(cls._gtest_scheduler == None)

            # start scheduler backend
            if backend == "cuda":
                cls._gtest_scheduler = _C_scheduler.GWScheduler(gtest_dir_path)
            else:
                raise NotImplementedError(
                    f"{backend} backend is not implemented yet"
                )

        # return single instance
        return config.SI_gw_scheduler


    def __init__(
        self, 
        backend : Literal["cuda", "rocm"],
        watchscript_path : str = "",
        world_size : int = 1,
        visual : bool = False,
        command : List[str] = ""
    ):
        self._backend : str = backend
        self._watchscript_path : str = watchscript_path
        self._world_size : int = world_size
        self._visual : bool = visual
        self._command : str = command
        self._watchscript : Callable = None

        # should be created in __new__
        assert(self._gtest_scheduler != None)

        # import watchscript
        if self._watchscript_path == "":
            self._watchscript_path = f"{os.getcwd()}/WatchScript.py"
        if os.path.exists(self._watchscript_path):
            abs_path = os.path.abspath(self._watchscript_path)
            spec = importlib.util.spec_from_file_location("watchscript_module", abs_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["watchscript_module"] = module
            spec.loader.exec_module(module)
            if hasattr(module, "WatchScript"):
                self._watchscript = module.WatchScript
                logger.info(
                    f"WatchScript found in {self._watchscript_path}"
                )
            else:
                self._watchscript = WatchScriptDefault
                logger.warning(
                    f"Callable 'WatchScript' not defined in {self._watchscript_path}, "
                    f"using default watchscript"
                )
                
        else:
            self._watchscript = WatchScriptDefault
            logger.warning(
                f"no WatchScript given, using default watchscript"
            )


    def serve(self):
        # start gtest scheduler
        self._gtest_scheduler.serve()
        if self._visual:
            # TODO: start gTrace
            pass


    def start_capsule(self):
        self._gtest_scheduler.start_capsule(self._command)

        # block until world size is match!
        while(self._gtest_scheduler.get_capsule_world_size() < self._world_size):
            continue


    def execute_step(self, step_name : Literal["profile_range", "record_range"], **kwargs) -> Any:
        if step_name == "record_range":
            if ("start_ms" in kwargs and "end_ms" in kwargs):
                return self._gtest_scheduler.step_record_event_1(kwargs["start_ms"], kwargs["end_ms"])
            elif "max_num_events" in kwargs:
                return self._gtest_scheduler.step_record_event_2(kwargs["max_num_events"])
            else:
                raise ValueError("Invalid arguments for record_range step")
        
        elif step_name == "profile_range":
            if "list_events" in kwargs and "list_metric_names" in kwargs:
                return self._gtest_scheduler.step_record_counter(
                    kwargs["list_events"], kwargs["list_metric_names"]
                )
            else:
                raise ValueError("Invalid arguments for profile_range step")


__all__ = [ "GWScheduler" ]
