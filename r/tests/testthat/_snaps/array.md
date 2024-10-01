# string/binary view nanoarrow_array buffers print correctly

    Code
      print(view_array_all_inlined)
    Output
      <nanoarrow_array string_view[26]>
       $ length    : int 26
       $ null_count: int 0
       $ offset    : int 0
       $ buffers   :List of 3
        ..$ :<nanoarrow_buffer validity<bool>[null] ``
        ..$ :<nanoarrow_buffer data<string_view>[26][416 b]>`
        ..$ :<nanoarrow_buffer variadic_size<int64>[null] ``
       $ dictionary: NULL
       $ children  : list()

---

    Code
      print(view_array_not_all_inlined)
    Output
      <nanoarrow_array string_view[1]>
       $ length    : int 1
       $ null_count: int 0
       $ offset    : int 0
       $ buffers   :List of 4
        ..$ :<nanoarrow_buffer validity<bool>[null] ``
        ..$ :<nanoarrow_buffer data<string_view>[1][16 b]>`
        ..$ :<nanoarrow_buffer variadic_data<string>[35 b]> `this string is longer...`
        ..$ :<nanoarrow_buffer variadic_size<int64>[1][8 b]> `35`
       $ dictionary: NULL
       $ children  : list()

