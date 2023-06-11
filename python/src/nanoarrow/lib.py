from ._lib import Schema, Array


def schema(obj):
    if isinstance(obj, Schema):
        return obj

    # Not entirely safe but will have to do until there's a dunder method
    if hasattr(obj, "_export_to_c"):
        out = Schema.empty()
        obj._export_to_c(out._addr())
        return out
    else:
        raise TypeError(
            f"Can't convert object of type {type(obj).__name__} to nanoarrow.Schema"
        )


def array(obj):
    if isinstance(obj, Array):
        return obj

    # Not entirely safe but will have to do until there's a dunder method
    if hasattr(obj, "_export_to_c"):
        out = Array.empty(Schema.empty())
        obj._export_to_c(out._addr(), out.schema._addr())
        return out
    else:
        raise TypeError(
            f"Can't convert object of type {type(obj).__name__} to nanoarrow.Array"
        )
