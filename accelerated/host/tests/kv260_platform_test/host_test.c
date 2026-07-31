/**
 * host_test.c - minimal sanity check that a v++-2025.2-built xclbin loads and
 * runs on this board's XRT (2.18.0). Throwaway - mirrors the XRT call
 * sequence in accelerated/host/fpga.c so it exercises the same API surface
 * (xrtDeviceLoadXclbinFile, xrtPLKernelOpenExclusive, xrtRunSetArg/Start/Wait,
 * xrtBOSync) without depending on the real matmul kernel.
 *
 * Build ON THE BOARD (arm64, needs XRT headers there):
 *   gcc -O2 -Wall -Wextra $(pkg-config --cflags xrt) \
 *       -o host_test host_test.c $(pkg-config --libs xrt) -luuid
 *
 * Run from the same directory as test_kernel.xclbin:
 *   ./host_test
 */
#include <stdio.h>

#include <xrt.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#define XCLBIN_PATH "test_kernel.xclbin"
#define KERNEL_NAME "test_kernel"
#define ARG_IN      0
#define ARG_OUT     1

int main(void) {
    xrtDeviceHandle dev = xrtDeviceOpen(0);
    if (!dev) {
        fprintf(stderr, "failed to open FPGA device 0\n");
        return 1;
    }

    xuid_t uuid;
    if (xrtDeviceLoadXclbinFile(dev, XCLBIN_PATH)) {
        fprintf(stderr, "failed to load xclbin '%s'\n", XCLBIN_PATH);
        return 1;
    }
    xrtDeviceGetXclbinUUID(dev, uuid);

    // exclusive open matches the real host code's convention, even though
    // this kernel has no error register to read back
    xrtKernelHandle krnl = xrtPLKernelOpenExclusive(dev, uuid, KERNEL_NAME);
    if (!krnl) {
        fprintf(stderr, "failed to open kernel '%s'\n", KERNEL_NAME);
        return 1;
    }

    int grp = xrtKernelArgGroupId(krnl, ARG_OUT);
    if (grp < 0) {
        fprintf(stderr, "kernel argument %d has no memory group\n", ARG_OUT);
        return 1;
    }

    xrtBufferHandle bo = xrtBOAlloc(dev, sizeof(int), 0, grp);
    if (!bo) {
        fprintf(stderr, "failed to allocate output buffer\n");
        return 1;
    }

    int *map = xrtBOMap(bo);
    if (!map) {
        fprintf(stderr, "failed to map output buffer\n");
        return 1;
    }
    *map = -1;

    xrtRunHandle run = xrtRunOpen(krnl);
    if (!run) {
        fprintf(stderr, "failed to open a run object\n");
        return 1;
    }

    xrtRunSetArg(run, ARG_IN, 41);
    xrtRunSetArg(run, ARG_OUT, bo);

    xrtRunStart(run);
    xrtRunWait(run);

    xrtBOSync(bo, XCL_BO_SYNC_BO_FROM_DEVICE, sizeof(int), 0);

    printf("result = %d (expected 42)\n", *map);
    int ok = (*map == 42);

    xrtRunClose(run);
    xrtBOFree(bo);
    xrtKernelClose(krnl);
    xrtDeviceClose(dev);

    return ok ? 0 : 1;
}
