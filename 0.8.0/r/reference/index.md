# Package index

## All functions

- [`array_stream_set_finalizer()`](array_stream_set_finalizer.md) :
  Register an array stream finalizer
- [`as_nanoarrow_array()`](as_nanoarrow_array.md) : Convert an object to
  a nanoarrow array
- [`as_nanoarrow_array_stream()`](as_nanoarrow_array_stream.md) :
  Convert an object to a nanoarrow array_stream
- [`as_nanoarrow_buffer()`](as_nanoarrow_buffer.md) : Convert an object
  to a nanoarrow buffer
- [`as_nanoarrow_schema()`](as_nanoarrow_schema.md)
  [`infer_nanoarrow_schema()`](as_nanoarrow_schema.md)
  [`nanoarrow_schema_parse()`](as_nanoarrow_schema.md)
  [`nanoarrow_schema_modify()`](as_nanoarrow_schema.md) : Convert an
  object to a nanoarrow schema
- [`as_nanoarrow_schema(`*`<python.builtin.object>`*`)`](as_nanoarrow_schema.python.builtin.object.md)
  [`as_nanoarrow_array(`*`<python.builtin.object>`*`)`](as_nanoarrow_schema.python.builtin.object.md)
  [`as_nanoarrow_array_stream(`*`<python.builtin.object>`*`)`](as_nanoarrow_schema.python.builtin.object.md)
  [`test_reticulate_with_nanoarrow()`](as_nanoarrow_schema.python.builtin.object.md)
  : Python integration via reticulate
- [`as_nanoarrow_vctr()`](as_nanoarrow_vctr.md)
  [`nanoarrow_vctr()`](as_nanoarrow_vctr.md) : Experimental Arrow
  encoded arrays as R vectors
- [`basic_array_stream()`](basic_array_stream.md) : Create ArrayStreams
  from batches
- [`convert_array()`](convert_array.md) : Convert an Array into an R
  vector
- [`convert_array_stream()`](convert_array_stream.md)
  [`collect_array_stream()`](convert_array_stream.md) : Convert an Array
  Stream into an R vector
- [`example_ipc_stream()`](example_ipc_stream.md) : Example Arrow IPC
  Data
- [`infer_nanoarrow_ptype()`](infer_nanoarrow_ptype.md) : Infer an R
  vector prototype
- [`infer_nanoarrow_ptype_extension()`](infer_nanoarrow_ptype_extension.md)
  [`convert_array_extension()`](infer_nanoarrow_ptype_extension.md)
  [`as_nanoarrow_array_extension()`](infer_nanoarrow_ptype_extension.md)
  : Implement Arrow extension types
- [`na_type()`](na_type.md) [`na_na()`](na_type.md)
  [`na_bool()`](na_type.md) [`na_int8()`](na_type.md)
  [`na_uint8()`](na_type.md) [`na_int16()`](na_type.md)
  [`na_uint16()`](na_type.md) [`na_int32()`](na_type.md)
  [`na_uint32()`](na_type.md) [`na_int64()`](na_type.md)
  [`na_uint64()`](na_type.md) [`na_half_float()`](na_type.md)
  [`na_float()`](na_type.md) [`na_double()`](na_type.md)
  [`na_string()`](na_type.md) [`na_large_string()`](na_type.md)
  [`na_string_view()`](na_type.md) [`na_binary()`](na_type.md)
  [`na_large_binary()`](na_type.md)
  [`na_fixed_size_binary()`](na_type.md)
  [`na_binary_view()`](na_type.md) [`na_date32()`](na_type.md)
  [`na_date64()`](na_type.md) [`na_time32()`](na_type.md)
  [`na_time64()`](na_type.md) [`na_duration()`](na_type.md)
  [`na_interval_months()`](na_type.md)
  [`na_interval_day_time()`](na_type.md)
  [`na_interval_month_day_nano()`](na_type.md)
  [`na_timestamp()`](na_type.md) [`na_decimal32()`](na_type.md)
  [`na_decimal64()`](na_type.md) [`na_decimal128()`](na_type.md)
  [`na_decimal256()`](na_type.md) [`na_struct()`](na_type.md)
  [`na_sparse_union()`](na_type.md) [`na_dense_union()`](na_type.md)
  [`na_list()`](na_type.md) [`na_large_list()`](na_type.md)
  [`na_list_view()`](na_type.md) [`na_large_list_view()`](na_type.md)
  [`na_fixed_size_list()`](na_type.md) [`na_map()`](na_type.md)
  [`na_dictionary()`](na_type.md) [`na_extension()`](na_type.md) :
  Create type objects
- [`na_vctrs()`](na_vctrs.md) : Vctrs extension type
- [`nanoarrow_array_init()`](nanoarrow_array_init.md)
  [`nanoarrow_array_set_schema()`](nanoarrow_array_init.md)
  [`nanoarrow_array_modify()`](nanoarrow_array_init.md) : Modify
  nanoarrow arrays
- [`nanoarrow_buffer_init()`](nanoarrow_buffer_init.md)
  [`nanoarrow_buffer_append()`](nanoarrow_buffer_init.md)
  [`convert_buffer()`](nanoarrow_buffer_init.md) : Create and modify
  nanoarrow buffers
- [`nanoarrow_extension_array()`](nanoarrow_extension_array.md) : Create
  Arrow extension arrays
- [`nanoarrow_extension_spec()`](nanoarrow_extension_spec.md)
  [`register_nanoarrow_extension()`](nanoarrow_extension_spec.md)
  [`unregister_nanoarrow_extension()`](nanoarrow_extension_spec.md)
  [`resolve_nanoarrow_extension()`](nanoarrow_extension_spec.md) :
  Register Arrow extension types
- [`nanoarrow_pointer_is_valid()`](nanoarrow_pointer_is_valid.md)
  [`nanoarrow_pointer_addr_dbl()`](nanoarrow_pointer_is_valid.md)
  [`nanoarrow_pointer_addr_chr()`](nanoarrow_pointer_is_valid.md)
  [`nanoarrow_pointer_addr_pretty()`](nanoarrow_pointer_is_valid.md)
  [`nanoarrow_pointer_release()`](nanoarrow_pointer_is_valid.md)
  [`nanoarrow_pointer_move()`](nanoarrow_pointer_is_valid.md)
  [`nanoarrow_pointer_export()`](nanoarrow_pointer_is_valid.md)
  [`nanoarrow_allocate_schema()`](nanoarrow_pointer_is_valid.md)
  [`nanoarrow_allocate_array()`](nanoarrow_pointer_is_valid.md)
  [`nanoarrow_allocate_array_stream()`](nanoarrow_pointer_is_valid.md)
  [`nanoarrow_pointer_set_protected()`](nanoarrow_pointer_is_valid.md) :
  Danger zone: low-level pointer operations
- [`nanoarrow_version()`](nanoarrow_version.md)
  [`nanoarrow_with_zstd()`](nanoarrow_version.md) : Underlying
  'nanoarrow' C library build
- [`read_nanoarrow()`](read_nanoarrow.md)
  [`write_nanoarrow()`](read_nanoarrow.md) : Read/write serialized
  streams of Arrow data
