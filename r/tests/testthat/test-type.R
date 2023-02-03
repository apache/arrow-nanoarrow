
test_that("type constructors for parameter-free types work", {
  # Some of these types have parameters but also have default values
  parameter_free_types <- c(
    "na", "bool", "uint8", "int8", "uint16", "int16",
    "uint32", "int32", "uint64", "int64", "half_float", "float",
    "double", "string", "binary", "date32",
    "date64", "timestamp", "time32", "time64", "interval_months",
    "interval_day_time", "struct",
    "duration", "large_string", "large_binary",
    "interval_month_day_nano"
  )

  for (type_name in parameter_free_types) {
    # Check that the right type gets created
    expect_identical(
      nanoarrow_schema_parse(na_type(!!type_name))$type,
      !!type_name
    )

    # Check that the default schema is nullable
    expect_identical(na_type(!!type_name)$flags, 2L)

    # Check that non-nullable schemas are non-nullable
    expect_identical(na_type(!!type_name, nullable = FALSE)$flags, 0L)
  }
})
