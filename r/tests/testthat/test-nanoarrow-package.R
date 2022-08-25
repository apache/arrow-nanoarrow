
test_that("nanoarrow_build_id() works", {
  expect_identical(
    nanoarrow_build_id(runtime = TRUE),
    nanoarrow_build_id(runtime = FALSE)
  )
})
