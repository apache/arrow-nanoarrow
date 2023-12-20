
fetchcontent_declare(nanoarrow_zlib
    URL https://github.com/madler/zlib/releases/download/v1.3/zlib13.zip
    URL_HASH SHA256=c561d09347f674f0d72692e7c75d9898919326c532aab7f8c07bb43b07efeb38
    )
fetchcontent_makeavailable(nanoarrow_zlib)

add_library(ZLIB::ZLIB ALIAS zlibstatic)
target_include_directories(zlibstatic INTERFACE ${zlib_BINARY_DIR} ${zlib_SOURCE_DIR})
