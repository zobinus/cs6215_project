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


from gtest.toolbox.binary_utilities.cuda import *
