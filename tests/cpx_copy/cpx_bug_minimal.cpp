#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t err = cmd;                                                      \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error: %s at %s:%d\n", hipGetErrorString(err),      \
              __FILE__, __LINE__);                                             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  int src_dev = 0, dst_dev = 1;
  if (argc >= 3) {
    src_dev = atoi(argv[1]);
    dst_dev = atoi(argv[2]);
  }

  char bus_src[64], bus_dst[64];
  HIP_CHECK(hipDeviceGetPCIBusId(bus_src, sizeof(bus_src), src_dev));
  HIP_CHECK(hipDeviceGetPCIBusId(bus_dst, sizeof(bus_dst), dst_dev));

  printf("CPX D2D test: dev %d (%s) -> dev %d (%s)\n", src_dev, bus_src,
         dst_dev, bus_dst);

  size_t size = 64 * 1024 * 1024;
  float *src_buf, *dst_buf;

  int canAccess = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, src_dev, dst_dev));
  if (!canAccess) {
    fprintf(stderr, "Device %d cannot access device %d\n", src_dev, dst_dev);
    return 1;
  }

  HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, dst_dev, src_dev));
  if (!canAccess) {
    fprintf(stderr, "Device %d cannot access device %d\n", dst_dev, src_dev);
    return 1;
  }

  HIP_CHECK(hipSetDevice(src_dev));
  HIP_CHECK(hipMalloc(&src_buf, size));
  HIP_CHECK(hipMemset(src_buf, 0xAB, size));

  HIP_CHECK(hipSetDevice(dst_dev));
  HIP_CHECK(hipMalloc(&dst_buf, size));
  HIP_CHECK(hipDeviceEnablePeerAccess(src_dev, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  printf("[Copy 1] hipMemcpyAsync D2D... ");
  fflush(stdout);
  HIP_CHECK(
      hipMemcpyAsync(dst_buf, src_buf, size, hipMemcpyDeviceToDevice, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  printf("OK\n");

  printf("[Copy 2] hipMemcpyAsync D2D... ");
  fflush(stdout);
  HIP_CHECK(
      hipMemcpyAsync(dst_buf, src_buf, size, hipMemcpyDeviceToDevice, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  printf("OK\n");

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipSetDevice(src_dev));
  HIP_CHECK(hipFree(src_buf));
  HIP_CHECK(hipSetDevice(dst_dev));
  HIP_CHECK(hipFree(dst_buf));

  printf("\nPASSED\n");
  return 0;
}
