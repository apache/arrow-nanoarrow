
test_that("nanoarrow_altrep() errors for unsupported types", {
  expect_error(nanoarrow_altrep(as_nanoarrow_array(1:10)), "Can't make ALTREP")
})

test_that("nanoarrow_altrep() works for string", {
  x <- as_nanoarrow_array(letters, schema = arrow::utf8())
  x_altrep <- nanoarrow_altrep(x)
  expect_output(.Internal(inspect(x_altrep)), "<nanoarrow::array_string\\[26\\]>")
  expect_identical(x_altrep, letters)
  expect_false(anyNA(x_altrep))

  for (i in seq_along(x_altrep)) {
    expect_identical(x_altrep[i], letters[i])
  }

  expect_output(.Internal(inspect(x_altrep)), "<nanoarrow::array_string\\[26\\]>")
})

test_that("nanoarrow_altrep() works for large string", {
  x <- as_nanoarrow_array(letters, schema = arrow::large_utf8())
  x_altrep <- nanoarrow_altrep(x)
  expect_output(.Internal(inspect(x_altrep)), "<nanoarrow::array_string\\[26\\]>")
  expect_identical(x_altrep, letters)
  expect_false(anyNA(x_altrep))

  for (i in seq_along(x_altrep)) {
    expect_identical(x_altrep[i], letters[i])
  }

  expect_output(.Internal(inspect(x_altrep)), "<nanoarrow::array_string\\[26\\]>")
})
