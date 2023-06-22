
# nanoarrow device extension

This extension provides a similar set of tools as the core nanoarrow C API
extended to the
[Arrow C Device](https://arrow.apache.org/docs/dev/format/CDeviceDataInterface.html)
interfaces in the Arrow specification.

Currently, this extension provides an implementation fof CUDA devices
and an implementation for the default Apple Metal device on MacOS/M1.
These implementation are preliminary/experimental and are under active
development.

## Example

```c
struct ArrowDevice* gpu = ArrowDeviceMetalDefaultDevice();
// Alternatively, ArrowDeviceCuda(ARROW_DEVICE_CUDA, 0)
// or  ArrowDeviceCuda(ARROW_DEVICE_CUDA_HOST, 0)
struct ArrowDevice* cpu = ArrowDeviceCpu();
struct ArrowArray array;
struct ArrowDeviceArray device_array;
struct ArrowDeviceArrayView device_array_view;

// Build a CPU array
ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_STRING), NANOARROW_OK);
ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
ASSERT_EQ(ArrowArrayAppendString(&array, ArrowCharView("abc")), NANOARROW_OK);
ASSERT_EQ(ArrowArrayAppendString(&array, ArrowCharView("defg")), NANOARROW_OK);
ASSERT_EQ(ArrowArrayAppendNull(&array, 1), NANOARROW_OK);
ASSERT_EQ(ArrowArrayFinishBuildingDefault(&array, nullptr), NANOARROW_OK);

// Convert to a DeviceArray, still on the CPU
ASSERT_EQ(ArrowDeviceArrayInit(cpu, &device_array, &array), NANOARROW_OK);

// Parse contents into a view that can be copied to another device
ArrowDeviceArrayViewInit(&device_array_view);
ArrowArrayViewInitFromType(&device_array_view.array_view, string_type);
ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array, nullptr),
          NANOARROW_OK);

// Copy to another device. For some devices, ArrowDeviceArrayMoveToDevice() is
// possible without an explicit copy (although this sometimes triggers an implicit
// copy by the driver).
struct ArrowDeviceArray device_array2;
device_array2.array.release = nullptr;
ASSERT_EQ(
    ArrowDeviceArrayViewCopy(&device_array, &device_array_view, gpu, &device_array2),
    NANOARROW_OK);
```
