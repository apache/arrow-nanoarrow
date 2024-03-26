
import nanoarrow as na


class CArraySuite:
    """
    Benchmarks for creating Arrays and CArrays
    """
    def setup(self):
        self.py_integers = list(range(int(1e6)))
        self.py_bools = [False, True, True, False] * int(1e6 // 4)
        self.c_integers = na.c_array(self.py_integers, na.int32())
        self.c_bools = na.c_array(self.py_bools, na.bool())

    def time_build_c_array_int32(self):
        na.c_array(self.py_integers, na.int32())

    def time_build_c_array_bool(self):
        na.c_array(self.py_bools, na.bool())
