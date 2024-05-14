import _cython_3_0_10
import enum
import types
from _typeshed import Incomplete
from typing import Callable, ClassVar

DEVICE_CPU: Device
__reduce_cython__: _cython_3_0_10.cython_function_or_method
__setstate_cython__: _cython_3_0_10.cython_function_or_method
__test__: dict
assert_type_equal: _cython_3_0_10.cython_function_or_method
c_version: _cython_3_0_10.cython_function_or_method
get_pyobject_buffer_count: _cython_3_0_10.cython_function_or_method
sys_byteorder: str

class CArray:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    buffers: Incomplete
    children: Incomplete
    device_id: Incomplete
    device_type: Incomplete
    device_type_id: Incomplete
    dictionary: Incomplete
    length: Incomplete
    n_buffers: Incomplete
    n_children: Incomplete
    null_count: Incomplete
    offset: Incomplete
    schema: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    @staticmethod
    def allocate(*args, **kwargs): ...
    def child(self, *args, **kwargs): ...
    def is_valid(self, *args, **kwargs): ...
    def view(self, *args, **kwargs): ...
    def __arrow_c_array__(self, *args, **kwargs):
        """
        Get a pair of PyCapsules containing a C ArrowArray representation of the object.

        Parameters
        ----------
        requested_schema : PyCapsule | None
            A PyCapsule containing a C ArrowSchema representation of a requested
            schema. Not supported.

        Returns
        -------
        Tuple[PyCapsule, PyCapsule]
            A pair of PyCapsules containing a C ArrowSchema and ArrowArray,
            respectively.
        """
    def __getitem__(self, index):
        """Return self[key]."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class CArrayBuilder:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    @staticmethod
    def allocate(*args, **kwargs): ...
    def append_bytes(self, *args, **kwargs): ...
    def append_strings(self, *args, **kwargs): ...
    def finish(self, *args, **kwargs): ...
    def init_from_schema(self, *args, **kwargs): ...
    def init_from_type(self, *args, **kwargs): ...
    def is_empty(self, *args, **kwargs): ...
    def resolve_null_count(self, *args, **kwargs): ...
    def set_buffer(self, *args, **kwargs):
        """Sets a buffer of this ArrowArray such the pointer at array->buffers[i] is
                equal to buffer->data and such that the buffer's lifcycle is managed by
                the array. If move is True, the input Python object that previously wrapped
                the ArrowBuffer will be invalidated, which is usually the desired behaviour
                if you built or imported a buffer specifically to build this array. If move
                is False (the default), this function will a make a shallow copy via another
                layer of Python object wrapping."""
    def set_child(self, *args, **kwargs): ...
    def set_length(self, *args, **kwargs): ...
    def set_null_count(self, *args, **kwargs): ...
    def set_offset(self, *args, **kwargs): ...
    def start_appending(self, *args, **kwargs): ...
    def __reduce__(self): ...

class CArrayStream:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    @staticmethod
    def allocate(*args, **kwargs): ...
    @staticmethod
    def from_c_arrays(*args, **kwargs): ...
    def get_next(self, *args, **kwargs):
        """Get the next Array from this stream

                Raises StopIteration when there are no more arrays in this stream.
        """
    def get_schema(self, *args, **kwargs):
        """Get the schema associated with this stream
        """
    def is_valid(self, *args, **kwargs): ...
    def release(self, *args, **kwargs): ...
    def __arrow_c_stream__(self, *args, **kwargs):
        """
        Export the stream as an Arrow C stream PyCapsule.

        Parameters
        ----------
        requested_schema : PyCapsule | None
            A PyCapsule containing a C ArrowSchema representation of a requested
            schema. Not supported.

        Returns
        -------
        PyCapsule
        """
    def __enter__(self): ...
    def __exit__(self, type: type[BaseException] | None, value: BaseException | None, traceback: types.TracebackType | None): ...
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self): ...

class CArrayView:
    buffers: Incomplete
    children: Incomplete
    dictionary: Incomplete
    layout: Incomplete
    length: Incomplete
    n_buffers: Incomplete
    n_children: Incomplete
    null_count: Incomplete
    offset: Incomplete
    storage_type: Incomplete
    storage_type_id: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def buffer(self, *args, **kwargs): ...
    def buffer_type(self, *args, **kwargs): ...
    def child(self, *args, **kwargs): ...
    @staticmethod
    def from_array(*args, **kwargs): ...
    @staticmethod
    def from_schema(*args, **kwargs): ...
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class CArrowTimeUnit:
    MICRO: ClassVar[int] = ...
    MILLI: ClassVar[int] = ...
    NANO: ClassVar[int] = ...
    SECOND: ClassVar[int] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...

class CArrowType:
    BINARY: ClassVar[int] = ...
    BOOL: ClassVar[int] = ...
    DATE32: ClassVar[int] = ...
    DATE64: ClassVar[int] = ...
    DECIMAL128: ClassVar[int] = ...
    DECIMAL256: ClassVar[int] = ...
    DENSE_UNION: ClassVar[int] = ...
    DICTIONARY: ClassVar[int] = ...
    DOUBLE: ClassVar[int] = ...
    DURATION: ClassVar[int] = ...
    EXTENSION: ClassVar[int] = ...
    FIXED_SIZE_BINARY: ClassVar[int] = ...
    FIXED_SIZE_LIST: ClassVar[int] = ...
    FLOAT: ClassVar[int] = ...
    HALF_FLOAT: ClassVar[int] = ...
    INT16: ClassVar[int] = ...
    INT32: ClassVar[int] = ...
    INT64: ClassVar[int] = ...
    INT8: ClassVar[int] = ...
    INTERVAL_DAY_TIME: ClassVar[int] = ...
    INTERVAL_MONTHS: ClassVar[int] = ...
    INTERVAL_MONTH_DAY_NANO: ClassVar[int] = ...
    LARGE_BINARY: ClassVar[int] = ...
    LARGE_LIST: ClassVar[int] = ...
    LARGE_STRING: ClassVar[int] = ...
    LIST: ClassVar[int] = ...
    MAP: ClassVar[int] = ...
    NA: ClassVar[int] = ...
    SPARSE_UNION: ClassVar[int] = ...
    STRING: ClassVar[int] = ...
    STRUCT: ClassVar[int] = ...
    TIME32: ClassVar[int] = ...
    TIME64: ClassVar[int] = ...
    TIMESTAMP: ClassVar[int] = ...
    UINT16: ClassVar[int] = ...
    UINT32: ClassVar[int] = ...
    UINT64: ClassVar[int] = ...
    UINT8: ClassVar[int] = ...
    UNINITIALIZED: ClassVar[int] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...

class CBuffer:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    data_type: Incomplete
    data_type_id: Incomplete
    element_size_bits: Incomplete
    format: Incomplete
    item_size: Incomplete
    n_elements: Incomplete
    size_bytes: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def element(self, *args, **kwargs): ...
    def elements(self, *args, **kwargs): ...
    @staticmethod
    def empty(*args, **kwargs): ...
    @staticmethod
    def from_pybuffer(*args, **kwargs): ...
    def __getitem__(self, index):
        """Return self[key]."""
    def __iter__(self):
        """Implement iter(self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class CBufferBuilder:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    capacity_bytes: Incomplete
    format: Incomplete
    size_bytes: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def advance(self, *args, **kwargs):
        """Manually increase :attr:`size_bytes` by ``additional_bytes``

                This can be used after writing to the buffer using the buffer protocol
                to ensure that :attr:`size_bytes` accurately reflects the number of
                bytes written to the buffer.
        """
    def finish(self, *args, **kwargs):
        """Finish building this buffer

                Performs any steps required to finish building this buffer and
                returns the result. Any behaviour resulting from calling methods
                on this object after it has been finished is not currently
                defined (but should not crash).
        """
    def reserve_bytes(self, *args, **kwargs):
        """Ensure that the underlying buffer has space for ``additional_bytes``
                more bytes to be written"""
    def set_data_type(self, *args, **kwargs):
        """Set the data type used to interpret elements in :meth:`write_elements`."""
    def set_format(self, *args, **kwargs):
        """Set the Python buffer format used to interpret elements in
                :meth:`write_elements`.
        """
    def write(self, *args, **kwargs):
        """Write bytes to this buffer

                Writes the bytes of ``content`` without considering the element type of
                ``content`` or the element type of this buffer.

                This method returns the number of bytes that were written.
        """
    def write_elements(self, *args, **kwargs):
        '''"Write an iterable of elements to this buffer

                Writes the elements of iterable ``obj`` according to the binary
                representation specified by :attr:`format`. This is currently
                powered by ``struct.pack_into()`` except when building bitmaps
                where an internal implementation is used.

                This method returns the number of elements that were written.
        '''
    def __reduce__(self): ...

class CBufferView:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    data_type: Incomplete
    data_type_id: Incomplete
    device: Incomplete
    element_size_bits: Incomplete
    format: Incomplete
    item_size: Incomplete
    n_elements: Incomplete
    size_bytes: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def copy(self, *args, **kwargs): ...
    def copy_into(self, *args, **kwargs): ...
    def element(self, *args, **kwargs): ...
    def elements(self, *args, **kwargs): ...
    def unpack_bits(self, *args, **kwargs): ...
    def unpack_bits_into(self, *args, **kwargs): ...
    def __getitem__(self, index):
        """Return self[key]."""
    def __iter__(self):
        """Implement iter(self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class CDeviceArray:
    array: Incomplete
    device_id: Incomplete
    device_type: Incomplete
    device_type_id: Incomplete
    schema: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def view(self, *args, **kwargs): ...
    def __arrow_c_array__(self, *args, **kwargs): ...
    def __arrow_c_device_array__(self, *args, **kwargs): ...
    def __reduce__(self): ...

class CLayout:
    buffer_data_type_id: Incomplete
    child_size_elements: Incomplete
    element_size_bits: Incomplete
    n_buffers: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self): ...

class CMaterializedArrayStream:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    arrays: Incomplete
    n_arrays: Incomplete
    schema: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def array(self, *args, **kwargs): ...
    def child(self, *args, **kwargs): ...
    @staticmethod
    def from_c_array(*args, **kwargs): ...
    @staticmethod
    def from_c_array_stream(*args, **kwargs): ...
    @staticmethod
    def from_c_arrays(*args, **kwargs): ...
    def __arrow_c_stream__(self, *args, **kwargs): ...
    def __getitem__(self, index):
        """Return self[key]."""
    def __iter__(self):
        """Implement iter(self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class CSchema:
    children: Incomplete
    dictionary: Incomplete
    flags: Incomplete
    format: Incomplete
    metadata: Incomplete
    n_children: Incomplete
    name: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    @staticmethod
    def allocate(*args, **kwargs): ...
    def child(self, *args, **kwargs): ...
    def is_valid(self, *args, **kwargs): ...
    def modify(self, *args, **kwargs): ...
    def type_equals(self, *args, **kwargs): ...
    def __arrow_c_schema__(self, *args, **kwargs):
        """
        Export to a ArrowSchema PyCapsule
        """
    def __deepcopy__(self): ...
    def __reduce__(self): ...

class CSchemaBuilder:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    @staticmethod
    def allocate(*args, **kwargs): ...
    def allocate_children(self, *args, **kwargs): ...
    def append_metadata(self, *args, **kwargs): ...
    def child(self, *args, **kwargs): ...
    def finish(self, *args, **kwargs): ...
    def set_child(self, *args, **kwargs): ...
    def set_dictionary(self, *args, **kwargs): ...
    def set_dictionary_ordered(self, *args, **kwargs): ...
    def set_flags(self, *args, **kwargs): ...
    def set_format(self, *args, **kwargs): ...
    def set_name(self, *args, **kwargs): ...
    def set_nullable(self, *args, **kwargs): ...
    def set_type(self, *args, **kwargs): ...
    def set_type_date_time(self, *args, **kwargs): ...
    def set_type_decimal(self, *args, **kwargs): ...
    def set_type_fixed_size(self, *args, **kwargs): ...
    def validate(self, *args, **kwargs): ...
    def __reduce__(self): ...

class CSchemaView:
    _decimal_types: ClassVar[tuple] = ...
    _fixed_size_types: ClassVar[tuple] = ...
    _time_unit_types: ClassVar[tuple] = ...
    _union_types: ClassVar[tuple] = ...
    buffer_format: Incomplete
    decimal_bitwidth: Incomplete
    decimal_precision: Incomplete
    decimal_scale: Incomplete
    dictionary_ordered: Incomplete
    extension_metadata: Incomplete
    extension_name: Incomplete
    fixed_size: Incomplete
    layout: Incomplete
    map_keys_sorted: Incomplete
    nullable: Incomplete
    storage_type: Incomplete
    storage_type_id: Incomplete
    time_unit: Incomplete
    time_unit_id: Incomplete
    timezone: Incomplete
    type: Incomplete
    type_id: Incomplete
    union_type_ids: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self): ...

class Device:
    device_id: Incomplete
    device_type: Incomplete
    device_type_id: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    @staticmethod
    def resolve(*args, **kwargs): ...
    def __reduce__(self): ...

class DeviceType(enum.Enum):
    __new__: ClassVar[Callable] = ...
    CPU: ClassVar[DeviceType] = ...
    CUDA: ClassVar[DeviceType] = ...
    CUDA_HOST: ClassVar[DeviceType] = ...
    CUDA_MANAGED: ClassVar[DeviceType] = ...
    EXT_DEV: ClassVar[DeviceType] = ...
    HEXAGON: ClassVar[DeviceType] = ...
    METAL: ClassVar[DeviceType] = ...
    ONEAPI: ClassVar[DeviceType] = ...
    OPENCL: ClassVar[DeviceType] = ...
    ROCM: ClassVar[DeviceType] = ...
    ROCM_HOST: ClassVar[DeviceType] = ...
    VPI: ClassVar[DeviceType] = ...
    VULKAN: ClassVar[DeviceType] = ...
    WEBGPU: ClassVar[DeviceType] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[object]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    _value_repr_: ClassVar[None] = ...

class Error:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    @staticmethod
    def raise_error(*args, **kwargs):
        """Raise a NanoarrowException without a message
        """
    @staticmethod
    def raise_error_not_ok(*args, **kwargs): ...
    def raise_message(self, *args, **kwargs):
        """Raise a NanoarrowException from this message
        """
    def raise_message_not_ok(self, *args, **kwargs): ...
    def __reduce__(self): ...

class NanoarrowException(RuntimeError):
    def __init__(self, *args, **kwargs) -> None: ...

class NoneAwareWrapperIterator:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def finish(self, *args, **kwargs):
        """Obtain the total count, null count, and validity bitmap after
                consuming this iterable."""
    def reserve(self, *args, **kwargs): ...
    def __iter__(self):
        """Implement iter(self)."""
    def __reduce__(self): ...

class SchemaMetadata:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    @staticmethod
    def empty(*args, **kwargs): ...
    def items(self, *args, **kwargs): ...
    def keys(self, *args, **kwargs): ...
    def values(self, *args, **kwargs): ...
    def __contains__(self, other) -> bool:
        """Return key in self."""
    def __getitem__(self, index):
        """Return self[key]."""
    def __iter__(self):
        """Implement iter(self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...
