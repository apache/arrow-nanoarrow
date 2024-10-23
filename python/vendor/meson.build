vendored_files = custom_target(
    'generate-pyx',
    output: [
        'flatcc.c',
        'nanoarrow.c',
        'nanoarrow_c.pxd',
        'nanoarrow_device.c',
        'nanoarrow_device_c.pxd',
        'nanoarrow_device.h',
        'nanoarrow_device.hpp',
        'nanoarrow.h',
        'nanoarrow.hpp',
        'nanoarrow_ipc.c',
        'nanoarrow_ipc.h',
        'nanoarrow_ipc.hpp',
    ],
    command: [
        py,
        meson.current_source_dir() + '/../bootstrap.py',
        '--output-dir', meson.current_build_dir()
    ],
)

nanoarrow_lib = static_library(
    'nanoarrow',
    sources: vendored_files[1],
)

nanoarrow_pyx_dep = declare_dependency(
    sources: vendored_files[2],
    link_with: nanoarrow_lib
)

nanoarrow_ipc_lib = static_library(
    'nanoarrow_ipc',
    sources: [vendored_files[0], vendored_files[9]],
    link_with: nanoarrow_lib,
)

nanoarrow_ipc_dep = declare_dependency(
    link_with: nanoarrow_ipc_lib
)

nanoarrow_device_lib = static_library(
    'nanoarrow_device',
    sources: vendored_files[3],
    link_with: nanoarrow_lib,
)

nanoarrow_device_pyx_dep = declare_dependency(
    sources: vendored_files[4],
    link_with: nanoarrow_device_lib,
)
