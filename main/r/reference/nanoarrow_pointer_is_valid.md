# Danger zone: low-level pointer operations

The [nanoarrow_schema](as_nanoarrow_schema.md),
[nanoarrow_array](as_nanoarrow_array.md), and
[nanoarrow_array_stream](as_nanoarrow_array_stream.md) classes are
represented in R as external pointers (`EXTPTRSXP`). When these objects
go out of scope (i.e., when they are garbage collected or shortly
thereafter), the underlying object's `release()` callback is called if
the underlying pointer is non-null and if the `release()` callback is
non-null.

## Usage

``` r
nanoarrow_pointer_is_valid(ptr)

nanoarrow_pointer_addr_dbl(ptr)

nanoarrow_pointer_addr_chr(ptr)

nanoarrow_pointer_addr_pretty(ptr)

nanoarrow_pointer_release(ptr)

nanoarrow_pointer_move(ptr_src, ptr_dst)

nanoarrow_pointer_export(ptr_src, ptr_dst)

nanoarrow_allocate_schema()

nanoarrow_allocate_array()

nanoarrow_allocate_array_stream()

nanoarrow_pointer_set_protected(ptr_src, protected)
```

## Arguments

- ptr, ptr_src, ptr_dst:

  An external pointer to a `struct ArrowSchema`, `struct ArrowArray`, or
  `struct ArrowArrayStream`.

- protected:

  An object whose scope must outlive that of `ptr`. This is useful for
  array streams since at least two specifications involving the array
  stream specify that the stream is only valid for the lifecycle of
  another object (e.g., an AdbcStatement or OGRDataset).

## Value

- `nanoarrow_pointer_is_valid()` returns TRUE if the pointer is non-null
  and has a non-null release callback.

- `nanoarrow_pointer_addr_dbl()` and `nanoarrow_pointer_addr_chr()`
  return pointer representations that may be helpful to facilitate
  moving or exporting nanoarrow objects to other libraries.

- `nanoarrow_pointer_addr_pretty()` gives a pointer representation
  suitable for printing or error messages.

- `nanoarrow_pointer_release()` returns `ptr`, invisibly.

- `nanoarrow_pointer_move()` and `nanoarrow_pointer_export()` reeturn
  `ptr_dst`, invisibly.

- `nanoarrow_allocate_array()`, `nanoarrow_allocate_schema()`, and
  `nanoarrow_allocate_array_stream()` return an
  [array](as_nanoarrow_array.md), a [schema](as_nanoarrow_schema.md),
  and an [array stream](as_nanoarrow_array_stream.md), respectively.

## Details

When interacting with other C Data Interface implementations, it is
important to keep in mind that the R object wrapping these pointers is
always passed by reference (because it is an external pointer) and may
be referred to by another R object (e.g., an element in a
[`list()`](https://rdrr.io/r/base/list.html) or as a variable assigned
in a user's environment). When importing a schema, array, or array
stream into nanoarrow this is not a problem: the R object takes
ownership of the lifecycle and memory is released when the R object is
garbage collected. In this case, one can use `nanoarrow_pointer_move()`
where `ptr_dst` was created using `nanoarrow_allocate_*()`.

The case of exporting is more complicated and as such has a dedicated
function, `nanoarrow_pointer_export()`, that implements different logic
schemas, arrays, and array streams:

- Schema objects are (deep) copied such that a fresh copy of the schema
  is exported and made the responsibility of some other C data interface
  implementation.

- Array objects are exported as a shell around the original array that
  preserves a reference to the R object. This ensures that the buffers
  and children pointed to by the array are not copied and that any
  references to the original array are not invalidated.

- Array stream objects are moved: the responsibility for the object is
  transferred to the other C data interface implementation and any
  references to the original R object are invalidated. Because these
  objects are mutable, this is typically what you want (i.e., you should
  not be pulling arrays from a stream accidentally from two places).

If you know the lifecycle of your object (i.e., you created the R object
yourself and never passed references to it elsewhere), you can slightly
more efficiently call `nanoarrow_pointer_move()` for all three pointer
types.
