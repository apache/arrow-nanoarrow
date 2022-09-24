
test_that("nanoarrow_altrep() returns NULL for unsupported types", {
  expect_null(nanoarrow_altrep(as_nanoarrow_array(1:10), character()))
  expect_null(nanoarrow_altrep(as_nanoarrow_array(1:10)))
})

test_that("nanoarrow_altrep() works for string", {
  x <- as_nanoarrow_array(c(NA, letters), schema = arrow::utf8())
  x_altrep <- nanoarrow_altrep(x, character())

  expect_output(.Internal(inspect(x_altrep)), "<nanoarrow::altrep_string\\[27\\]>")

  # Check that some common operations that call the string elt method
  # don't materialize the vector
  expect_identical(x_altrep, c(NA, letters))
  expect_length(x_altrep, 27)
  expect_false(is_nanoarrow_altrep_materialized(x_altrep))

  # Setting an element will materialize, duplicate, then modify
  x_altrep2 <- x_altrep
  x_altrep2[1] <- "not a letter"
  expect_identical(x_altrep2, c("not a letter", letters))
  expect_true(is_nanoarrow_altrep_materialized(x_altrep))

  # Check the same operations on the materialized output
  expect_identical(x_altrep, c(NA, letters))
  expect_length(x_altrep, 27)

  # Materialization should get printed in inspect()
  expect_output(.Internal(inspect(x_altrep)), "<materialized nanoarrow::altrep_string\\[27\\]>")

  # For good measure, force materialization again and check
  nanoarrow_altrep_force_materialize(x_altrep)
  expect_identical(x_altrep, c(NA, letters))
  expect_length(x_altrep, 27)
})

test_that("nanoarrow_altrep() works for large string", {
  x <- as_nanoarrow_array(letters, schema = arrow::large_utf8())
  x_altrep <- nanoarrow_altrep(x, character())
  expect_identical(x_altrep, letters)
})

test_that("is_nanoarrow_altrep() returns true for nanoarrow altrep objects", {
  expect_false(is_nanoarrow_altrep("not altrep"))
  expect_false(is_nanoarrow_altrep(1:10))
  expect_true(is_nanoarrow_altrep(nanoarrow_altrep(as_nanoarrow_array("whee"), character())))
})

test_that("nanoarrow_altrep() works for valid integer()", {
  # TODO: Not altrep yet, just materialized
  arrow_int_types <- list(
    int8 = arrow::int8(),
    uint8 = arrow::uint8(),
    int16 = arrow::int16(),
    uint16 = arrow::uint16(),
    int32 = arrow::int32(),
    uint32 = arrow::uint32(),
    int64 = arrow::int64(),
    uint64 = arrow::uint64(),
    float32 = arrow::float32(),
    float64 = arrow::float64()
  )

  ints <- c(NA, 0:10)
  for (nm in names(arrow_int_types)) {
    expect_identical(
      nanoarrow_altrep(
        as_nanoarrow_array(ints, schema = arrow_int_types[[!!nm]]),
        integer()
      ),
      ints
    )
  }

  ints_no_na <- 0:10
  for (nm in names(arrow_int_types)) {
    expect_identical(
      nanoarrow_altrep(
        as_nanoarrow_array(ints_no_na, schema = arrow_int_types[[!!nm]]),
        integer()
      ),
      ints_no_na
    )
  }

  # Boolean array to integer
  expect_identical(
    nanoarrow_altrep(
      as_nanoarrow_array(c(NA, TRUE, FALSE), schema = arrow::boolean()),
      integer()
    ),
    c(NA, 1L, 0L)
  )

  expect_identical(
    nanoarrow_altrep(
      as_nanoarrow_array(c(TRUE, FALSE), schema = arrow::boolean()),
      integer()
    ),
    c(1L, 0L)
  )
})

test_that("nanoarrow_altrep() warns for invalid integer()", {
  array <- as_nanoarrow_array(arrow::as_arrow_array(.Machine$double.xmax))
  expect_warning(
    expect_identical(nanoarrow_altrep(array, integer()), NA_integer_),
    "1 value\\(s\\) outside integer range set to NA"
  )

  array <- as_nanoarrow_array(arrow::as_arrow_array(c(NA, .Machine$double.xmax)))
  expect_warning(
    expect_identical(nanoarrow_altrep(array, integer()), c(NA_integer_, NA_integer_)),
    "1 value\\(s\\) outside integer range set to NA"
  )
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
