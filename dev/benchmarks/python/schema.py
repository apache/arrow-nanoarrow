
import nanoarrow as na


class SchemaSuite:
    """
    Benchmarks of some Schema/CSchema operations
    """
    def setup(self):
        self.children = [na.int32()] * 10000
        self.c_children = [na.c_schema(child) for child in self.children]
        self.c_wide_struct = na.c_schema(na.struct(self.children))

    def time_create_wide_struct_from_schemas(self):
        """Create a struct Schema with 10000 columns from a list of Schema"""
        na.struct(self.children)

    def time_create_wide_struct_from_c_schemas(self):
        """Create a struct Schema with 10000 columns from a list of CSchema"""
        na.struct(self.c_children)

    def time_c_schema_protocol_wide_struct(self):
        """Export a struct Schema with 10000 columns via the PyCapsule protocol"""
        self.c_wide_struct.__arrow_c_schema__()
