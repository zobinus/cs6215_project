import os
import sys

# make sure current directory exist in sys.path
module_dir = os.path.dirname(os.path.abspath(__file__))
if module_dir not in sys.path:
    sys.path.append(module_dir)


# load close-source portion of gWatch
import ctypes
from ctypes.util import find_library
lib_name = "gtest_dark"
lib_path = find_library(lib_name)
if lib_path is None:
    raise RuntimeError(f"cannot find the shared library '{lib_name}'")
try:
    ctypes.CDLL(lib_path)
except OSError as e:
    raise RuntimeError(f"failed to load '{lib_path}': {e}")

from gtest.toolbox.inline_profiler.context import *
from gtest.toolbox.inline_profiler.profiler import *
from gtest.toolbox.inline_profiler.device import *


# import torch adaptor
try:
    import torch
    from gtest.toolbox.inline_profiler.torch_adaptor import *
except ImportError:
    pass
