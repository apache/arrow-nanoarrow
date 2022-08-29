
test_that("new_data_frame() works", {
  expect_identical(
    new_data_frame(list(x = 1, y = 2), nrow = 1),
    data.frame(x = 1, y = 2)
  )
})
