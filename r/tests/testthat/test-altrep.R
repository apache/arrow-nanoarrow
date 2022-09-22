
test_that("nanoarrow_altrep() returns NULL for unsupported types", {
  expect_null(nanoarrow_altrep(as_nanoarrow_array(1:10), character()))
  expect_null(nanoarrow_altrep(as_nanoarrow_array(1:10)))
})

test_that("nanoarrow_altrep() works for string", {
  x <- as_nanoarrow_array(letters, schema = arrow::utf8())
  x_altrep <- nanoarrow_altrep(x, character())
  expect_output(.Internal(inspect(x_altrep)), "<nanoarrow::array_string\\[26\\]>")
  expect_identical(x_altrep, letters)
  expect_false(anyNA(x_altrep))

  for (i in seq_along(x_altrep)) {
    expect_identical(x_altrep[i], letters[i])
  }

  expect_output(.Internal(inspect(x_altrep)), "<nanoarrow::array_string\\[26\\]>")

  x_altrep[1] <- "not a letter"
  expect_identical(x_altrep, c("not a letter", letters[-1]))
})

test_that("nanoarrow_altrep() works for large string", {
  x <- as_nanoarrow_array(letters, schema = arrow::large_utf8())
  x_altrep <- nanoarrow_altrep(x, character())
  expect_output(.Internal(inspect(x_altrep)), "<nanoarrow::array_string\\[26\\]>")
  expect_identical(x_altrep, letters)
  expect_false(anyNA(x_altrep))

  for (i in seq_along(x_altrep)) {
    expect_identical(x_altrep[i], letters[i])
  }

  expect_output(.Internal(inspect(x_altrep)), "<nanoarrow::array_string\\[26\\]>")

  x_altrep[1] <- "not a letter"
  expect_identical(x_altrep, c("not a letter", letters[-1]))
})

test_that("is_nanoarrow_altrep() returns true for nanoarrow altrep objects", {
  expect_false(is_nanoarrow_altrep("not altrep"))
  expect_false(is_nanoarrow_altrep(1:10))
  expect_true(is_nanoarrow_altrep(nanoarrow_altrep(as_nanoarrow_array("whee"), character())))
})

test_that("nanoarrow_altrep_force_materialize() forces materialization", {
  x <- as_nanoarrow_array(letters, schema = arrow::utf8())
  x_altrep <- nanoarrow_altrep(x, character())

  expect_identical(nanoarrow_altrep_force_materialize("not altrep"), 0L)
  expect_identical(nanoarrow_altrep_force_materialize(x_altrep), 1L)

  x <- as_nanoarrow_array(letters, schema = arrow::utf8())
  x_altrep_df <- data.frame(x = nanoarrow_altrep(x, character()))
  expect_identical(
    nanoarrow_altrep_force_materialize(x_altrep_df, recursive = FALSE),
    0L
  )
  expect_identical(
    nanoarrow_altrep_force_materialize(x_altrep_df, recursive = TRUE),
    1L
  )
  expect_identical(
    nanoarrow_altrep_force_materialize(x_altrep_df, recursive = TRUE),
    0L
  )
})

test_that("is_nanoarrow_altrep_materialized() checks for materialization", {
  expect_identical(is_nanoarrow_altrep_materialized("not altrep"), NA)
  expect_identical(is_nanoarrow_altrep_materialized(1:10), NA)

  x <- as_nanoarrow_array(letters, schema = arrow::utf8())
  x_altrep <- nanoarrow_altrep(x, character())
  expect_false(is_nanoarrow_altrep_materialized(x_altrep))
  expect_identical(nanoarrow_altrep_force_materialize(x_altrep), 1L)
  expect_true(is_nanoarrow_altrep_materialized(x_altrep))
})
