# buffers can be printed

    Code
      str(as_nanoarrow_buffer(1:10))
    Output
      <nanoarrow_buffer data<int32>[10][40 b]> `1 2 3 4 5 6 7 8 9 10`

---

    Code
      str(as_nanoarrow_buffer(1:10000))
    Output
      <nanoarrow_buffer data<int32>[10000][40000 b]> `1 2 3 4 5 6 7 8 9 10 11 12 1...`

---

    Code
      str(as_nanoarrow_buffer(strrep("abcdefg", 100)))
    Output
      <nanoarrow_buffer data<string>[700 b]> `abcdefgabcdefgabcdefgabcdefgabcdefga...`

---

    Code
      str(as_nanoarrow_buffer(charToRaw(strrep("abcdefg", 100))))
    Output
      <nanoarrow_buffer data<binary>[700 b]> `61 62 63 64 65 66 67 61 62 63 64 65 ...`

---

    Code
      str(array$buffers[[2]])
    Output
      <nanoarrow_buffer unknown<unknown>>

