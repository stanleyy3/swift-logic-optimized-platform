/**
 * fpga.c - XRT device/kernel lifecycle for the FPGA matmul accelerator
 */

#include "fpga.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "config.h"
#include "quant.h"

xrtDeviceHandle fpga_dev = NULL;
xrtKernelHandle fpga_matmul_krnl = NULL;
xrtRunHandle fpga_matmul_run = NULL;

xrtBufferHandle fpga_bo_a = NULL;
xrtBufferHandle fpga_bo_b = NULL;
xrtBufferHandle fpga_bo_c = NULL;

/**
 * @brief Allocates one tile buffer, failing loudly rather than handing back a
 *        null handle for xrtBOMap to trip over later
 *
 * @param[in] bytes Size of the buffer
 * @param[in] argno Index of the kernel argument the buffer is bound to
 * @return          The buffer handle
 */
static xrtBufferHandle alloc_tile_bo(size_t bytes, int argno) {
    // only global memory arguments have a memory group; a negative result
    // means argno is not a pointer argument of this kernel
    int grp = xrtKernelArgGroupId(fpga_matmul_krnl, argno);
    if (grp < 0) {
        fprintf(stderr, "fpga_init: kernel argument %d has no memory group "
                        "(is it a global memory pointer?)\n", argno);
        exit(1);
    }

    xrtBufferHandle bo = xrtBOAlloc(fpga_dev, bytes, 0, grp);
    if (!bo) {
        fprintf(stderr, "fpga_init: failed to allocate %zu bytes for kernel "
                        "argument %d in memory group %d\n", bytes, argno, grp);
        exit(1);
    }

    return bo;
}

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

    fpga_matmul_run = xrtRunOpen(fpga_matmul_krnl);
    if (!fpga_matmul_run) {
        fprintf(stderr, "fpga_init: failed to open a run object for kernel '%s'\n",
                FPGA_KERNEL_NAME);
        exit(1);
    }

    // operands are float16 (the array's MUL_WIDTH); each accumulator is
    // ACC_WIDTH bits of fixed point, read back out of a 64-bit slot
    size_t operand_bytes = (size_t)TILE_DIM * TILE_DIM * sizeof(f16_t);
    size_t acc_bytes = (size_t)TILE_DIM * TILE_DIM * sizeof(uint64_t);

    fpga_bo_a = alloc_tile_bo(operand_bytes, 0);
    fpga_bo_b = alloc_tile_bo(operand_bytes, 1);
    fpga_bo_c = alloc_tile_bo(acc_bytes, 2);

    // a short k tile leaves part of the operand buffers unwritten, and the
    // array multiplies the whole tile regardless, so start from zeros
    memset(xrtBOMap(fpga_bo_a), 0, operand_bytes);
    memset(xrtBOMap(fpga_bo_b), 0, operand_bytes);
}

void fpga_cleanup(void) {
    xrtBOFree(fpga_bo_a);
    xrtBOFree(fpga_bo_b);
    xrtBOFree(fpga_bo_c);

    xrtRunClose(fpga_matmul_run);
    xrtKernelClose(fpga_matmul_krnl);
    xrtDeviceClose(fpga_dev);
}
