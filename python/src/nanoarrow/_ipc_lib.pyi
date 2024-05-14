import _cython_3_0_10
from _typeshed import Incomplete

__reduce_cython__: _cython_3_0_10.cython_function_or_method
__setstate_cython__: _cython_3_0_10.cython_function_or_method
__test__: dict
init_array_stream: _cython_3_0_10.cython_function_or_method

class CIpcInputStream:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    @staticmethod
    def from_readable(*args, **kwargs): ...
    def is_valid(self, *args, **kwargs): ...
    def release(self, *args, **kwargs): ...
    def __reduce__(self): ...

class PyInputStreamPrivate:
    close_obj: Incomplete
    obj: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def set_buffer(self, *args, **kwargs): ...
    def __buffer__(self, *args, **kwargs):
        """Return a buffer object that exposes the underlying memory of the object."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...
