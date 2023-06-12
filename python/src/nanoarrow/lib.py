from ._lib import Schema, Array, ArrayStream


def schema(obj):
    if isinstance(obj, Schema):
        return obj

    # Not particularly safe because _export_to_c() could be exporting an
    # array, schema, or array_stream. The ideal
    # solution here would be something like __arrow_c_schema__()
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

    # Somewhat safe because calling _export_to_c() with two arguments will
    # not fail with a crash (but will fail with a confusing error). The ideal
    # solution here would be something like __arrow_c_array__()
    if hasattr(obj, "_export_to_c"):
        out = Array.empty(Schema.empty())
        obj._export_to_c(out._addr(), out.schema._addr())
        return out
    else:
        raise TypeError(
            f"Can't convert object of type {type(obj).__name__} to nanoarrow.Array"
        )


def array_stream(obj):
    if isinstance(obj, Schema):
        return obj

    # Not particularly safe because _export_to_c() could be exporting an
    # array, schema, or array_stream. The ideal
    # solution here would be something like __arrow_c_array_stream__()
    if hasattr(obj, "_export_to_c"):
        out = ArrayStream.empty()
        obj._export_to_c(out._addr())
        return out
    else:
        raise TypeError(
            f"Can't convert object of type {type(obj).__name__} to nanoarrow.Schema"
        )
