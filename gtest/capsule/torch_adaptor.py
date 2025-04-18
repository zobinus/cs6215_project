from typing import Any, List

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

import gtest.libgtest_capsule as _C_capsule

# take these refs:
# https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557
# https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505
# https://pastebin.com/V3wATa7w
# https://pastebin.com/AkvAyJBw
# https://dev-discuss.pytorch.org/t/torchdispatchmode-for-debugging-testing-and-more/717


class _GWEvent_App_Range:
    def __init__(self, name : str):
        self._C_instance = _C_capsule.GWEvent_App_Range(name)


    def record_begin_tick(self):
        self._C_instance.record_begin_tick()


    def record_end_tick(self):
        self._C_instance.record_end_tick()


    def set_input_tensor_info(self, input_tensor_info : List):
        self._C_instance.set_input_tensor_info(input_tensor_info)


    def set_output_tensor_info(self, output_tensor_info : List):
        self._C_instance.set_output_tensor_info(output_tensor_info)


class GWModelAnlyser(TorchDispatchMode):
    def __init__(self, module : nn.Module):
        super().__init__()
        self._module = module
        self._module_app_range_event : List[_GWEvent_App_Range] = []

        # we maintain this list to prevent the app range event from being garbage collected
        self._event_keepalive : List[_GWEvent_App_Range] = []
        
        self.__parse_module(module)


    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """
        op-level hijack
        """
        input_tensor_info : List = []
        output_tensor_info : List = []
        kwargs = kwargs if kwargs else {}

        # create new app range event
        app_range_event : _GWEvent_App_Range = _GWEvent_App_Range(func.__name__)
        _C_capsule.add_app_range_event(app_range_event._C_instance)
        self._event_keepalive.append(app_range_event)

        # record input tensor info
        input_tensor_info = GWModelAnlyser.__collect_tensor_info(args)
        input_tensor_info += GWModelAnlyser.__collect_tensor_info(kwargs.values())
        app_range_event.set_input_tensor_info(input_tensor_info)

        # execute the operator
        app_range_event.record_begin_tick()
        out = func(*args, **kwargs)
        app_range_event.record_end_tick()

        # record output tensor info
        output_tensor_info = GWModelAnlyser.__collect_tensor_info(out)
        app_range_event.set_output_tensor_info(output_tensor_info)

        return out


    def __parse_module(self, module : nn.Module):
        """
        module-level hijack
        """
        module.register_forward_pre_hook(GWModelAnlyser.__pre_nn_module_forward(self))
        module.register_forward_hook(GWModelAnlyser.__post_nn_module_forward(self))
        for child in module.children():
            self.__parse_module(child)


    @staticmethod
    def __pre_nn_module_forward(self : 'GWModelAnlyser'):
        def _func(module : nn.Module, input : Any):
            # create new app range event
            app_range_event : _GWEvent_App_Range = _GWEvent_App_Range(module.__class__.__name__)
            _C_capsule.add_app_range_event(app_range_event._C_instance)
            self._event_keepalive.append(app_range_event)
    
            # record input tensor info
            input_tensor_info = GWModelAnlyser.__collect_tensor_info(input)
            app_range_event.set_input_tensor_info(input_tensor_info)

            # save the app range event to stack for later used
            self._module_app_range_event.append(app_range_event)

            # start ticking
            app_range_event.record_begin_tick()
            
        return _func


    @staticmethod
    def __post_nn_module_forward(self : 'GWModelAnlyser'):
        def _func(module : nn.Module, input : Any, output : Any):
            # obtain the app range event from stack
            try:
                app_range_event = self._module_app_range_event.pop()
            except:
                raise RuntimeError("no app range event found")

            # end ticking
            app_range_event.record_end_tick()

            # record output tensor info
            output_tensor_info = GWModelAnlyser.__collect_tensor_info(output)
            app_range_event.set_output_tensor_info(output_tensor_info)

        return _func


    @staticmethod
    def __collect_tensor_info(tensors : Any):
        info : List = []
        def _recursive_collect(elem):
            if isinstance(elem, torch.Tensor):
                info.append({
                    "shape": str(list(elem.shape)),
                    "dtype": str(elem.dtype).split(".")[-1],
                    "ptr": str(elem.data_ptr()),
                    "device": str(elem.device)
                })
            elif isinstance(elem, (list, tuple)):
                for e in elem:
                    _recursive_collect(e)
            elif isinstance(elem, dict):
                for v in elem.values():
                    _recursive_collect(v)
        _recursive_collect(tensors)
        return info


__all__ = [ "GWModelAnlyser" ]
