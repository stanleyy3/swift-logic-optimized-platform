/**
 * fpga.c - XRT device/kernel lifecycle for the FPGA matmul accelerator
 */

#include "fpga.h"

#include <stdio.h>
#include <stdlib.h>

#include "config.h"

xrtDeviceHandle fpga_dev = NULL;
xrtKernelHandle fpga_matmul_krnl = NULL;

xrtBufferHandle fpga_bo_a = NULL;
xrtBufferHandle fpga_bo_b = NULL;
xrtBufferHandle fpga_bo_c = NULL;

void fpga_init(void) {
    fpga_dev = xrtDeviceOpen(0);
    if (!fpga_dev) {
        fprintf(stderr, "fpga_init: failed to open FPGA device 0\n");
        exit(1);
    }

    xuid_t uuid;
    if (xrtDeviceLoadXclbin(fpga_dev, FPGA_XCLBIN_PATH)) {
        fprintf(stderr, "fpga_init: failed to load xclbin '%s'\n", FPGA_XCLBIN_PATH);
        exit(1);
    }
    xrtDeviceGetXclbinUUID(fpga_dev, uuid);

    fpga_matmul_krnl = xrtPLKernelOpen(fpga_dev, uuid, FPGA_KERNEL_NAME);
    if (!fpga_matmul_krnl) {
        fprintf(stderr, "fpga_init: failed to open kernel '%s'\n", FPGA_KERNEL_NAME);
        exit(1);
    }

    size_t tile_bytes = (size_t)TILE_DIM * TILE_DIM * sizeof(float);
    fpga_bo_a = xrtBOAlloc(fpga_dev, tile_bytes, 0, xrtKernelArgGroupId(fpga_matmul_krnl, 0));
    fpga_bo_b = xrtBOAlloc(fpga_dev, tile_bytes, 0, xrtKernelArgGroupId(fpga_matmul_krnl, 1));
    fpga_bo_c = xrtBOAlloc(fpga_dev, tile_bytes, 0, xrtKernelArgGroupId(fpga_matmul_krnl, 2));
}

void fpga_cleanup(void) {
    xrtBOFree(fpga_bo_a);
    xrtBOFree(fpga_bo_b);
    xrtBOFree(fpga_bo_c);

    xrtKernelClose(fpga_matmul_krnl);
    xrtDeviceClose(fpga_dev);
}
