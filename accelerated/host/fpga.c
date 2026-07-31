/**
 * fpga.c - Device lifecycle and block launches for the FPGA matmul accelerator
 */

#include "fpga.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "config.h"
#include "quant.h"

// a block buffer holds FPGA_BLOCK_DIM x FPGA_BLOCK_DIM float16 elements; the
// result comes back float16 too, since the array converts on the way out
#define BLOCK_ELEMS ((size_t)FPGA_BLOCK_DIM * FPGA_BLOCK_DIM)
#define BLOCK_BYTES (BLOCK_ELEMS * sizeof(f16_t))

#if FPGA_MODEL

////////////////////////////////////////////////////////////////////////////////
// SOFTWARE MODEL
////////////////////////////////////////////////////////////////////////////////

// Stands in for the device so the host's blocking, padding and de-tiling can be
// exercised without a board. Plain heap buffers replace the XRT ones; there is
// no device to sync to, and the model cannot fail, so `load_block_en` only has
// to be honoured to the extent that a resident block really does persist -
// which it does, because the buffers are never cleared between launches.

#include "device_model.h"

static f16_t *model_a = NULL;
static f16_t *model_b = NULL;
static f16_t *model_c = NULL;

void fpga_init(void) {
    model_a = calloc(BLOCK_ELEMS, sizeof(f16_t));
    model_b = calloc(BLOCK_ELEMS, sizeof(f16_t));
    model_c = calloc(BLOCK_ELEMS, sizeof(f16_t));

    if (!model_a || !model_b || !model_c) {
        fprintf(stderr, "fpga_init: failed to allocate the model's block buffers\n");
        exit(1);
    }
}

void fpga_cleanup(void) {
    free(model_a);
    free(model_b);
    free(model_c);
}

f16_t *fpga_block_a(void) { return model_a; }
f16_t *fpga_block_b(void) { return model_b; }
const f16_t *fpga_block_c(void) { return model_c; }

void fpga_launch_block(int blk_m, int blk_k, int blk_n, int load_block_en) {
    (void)load_block_en;

    device_model_block_matmul(model_a, model_b, blk_m, blk_k, blk_n, model_c);
}

#else

////////////////////////////////////////////////////////////////////////////////
// XRT
////////////////////////////////////////////////////////////////////////////////

#include <xrt.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

// Kernel argument indices; these must match the order in the packaged kernel's
// kernel.xml. Arguments 0-2 are global memory pointers, so XRT writes each
// buffer's *device* address into them - which is what the DataMover needs,
// since it masters memory itself and cannot see host virtual addresses.
#define ARG_A_BLOCK      0
#define ARG_B_BLOCK      1
#define ARG_C_BLOCK      2
#define ARG_BLK_M        3
#define ARG_BLK_K        4
#define ARG_BLK_N        5
#define ARG_LOAD_BLK_EN  6

static xrtDeviceHandle fpga_dev = NULL;
static xrtKernelHandle fpga_matmul_krnl = NULL;
static xrtRunHandle fpga_matmul_run = NULL;

static xrtBufferHandle fpga_bo_a = NULL;
static xrtBufferHandle fpga_bo_b = NULL;
static xrtBufferHandle fpga_bo_c = NULL;

static f16_t *map_a = NULL;
static f16_t *map_b = NULL;
static f16_t *map_c = NULL;

/**
 * @brief Allocates one block buffer, failing loudly rather than handing back a
 *        null handle for xrtBOMap to trip over later
 *
 * @param[in] argno Index of the kernel argument the buffer is bound to
 * @return          The buffer handle
 */
static xrtBufferHandle alloc_block_bo(int argno) {
    // only global memory arguments have a memory group; a negative result
    // means argno is not a pointer argument of this kernel
    int grp = xrtKernelArgGroupId(fpga_matmul_krnl, argno);
    if (grp < 0) {
        fprintf(stderr, "fpga_init: kernel argument %d has no memory group "
                        "(is it a global memory pointer?)\n", argno);
        exit(1);
    }

    xrtBufferHandle bo = xrtBOAlloc(fpga_dev, BLOCK_BYTES, 0, grp);
    if (!bo) {
        fprintf(stderr, "fpga_init: failed to allocate %zu bytes for kernel "
                        "argument %d in memory group %d\n", BLOCK_BYTES, argno, grp);
        exit(1);
    }

    return bo;
}

/**
 * @brief Maps a block buffer, failing loudly on a null mapping
 *
 * @param[in] bo    Buffer to map
 * @param[in] label Name of the buffer, for the error message
 * @return          Pointer to the mapping
 */
static f16_t *map_block_bo(xrtBufferHandle bo, const char *label) {
    f16_t *p = xrtBOMap(bo);
    if (!p) {
        fprintf(stderr, "fpga_init: failed to map the %s block buffer\n", label);
        exit(1);
    }

    return p;
}

void fpga_init(void) {
    fpga_dev = xrtDeviceOpen(0);
    if (!fpga_dev) {
        fprintf(stderr, "fpga_init: failed to open FPGA device 0\n");
        exit(1);
    }

    xuid_t uuid;
    // ...File(), not xrtDeviceLoadXclbin(), which takes an already-read axlf
    // image rather than a path
    if (xrtDeviceLoadXclbinFile(fpga_dev, FPGA_XCLBIN_PATH)) {
        fprintf(stderr, "fpga_init: failed to load xclbin '%s'\n", FPGA_XCLBIN_PATH);
        exit(1);
    }
    xrtDeviceGetXclbinUUID(fpga_dev, uuid);

    // exclusive, because xrtKernelReadRegister refuses to read a kernel that is
    // shared; that read is how the device's error flag gets back here
    fpga_matmul_krnl = xrtPLKernelOpenExclusive(fpga_dev, uuid, FPGA_KERNEL_NAME);
    if (!fpga_matmul_krnl) {
        fprintf(stderr, "fpga_init: failed to open kernel '%s'\n", FPGA_KERNEL_NAME);
        exit(1);
    }

    // a run object is reusable, so one is opened up front and re-armed for each
    // block rather than opened and closed per launch
    fpga_matmul_run = xrtRunOpen(fpga_matmul_krnl);
    if (!fpga_matmul_run) {
        fprintf(stderr, "fpga_init: failed to open a run object for kernel '%s'\n",
                FPGA_KERNEL_NAME);
        exit(1);
    }

    fpga_bo_a = alloc_block_bo(ARG_A_BLOCK);
    fpga_bo_b = alloc_block_bo(ARG_B_BLOCK);
    fpga_bo_c = alloc_block_bo(ARG_C_BLOCK);

    map_a = map_block_bo(fpga_bo_a, "first operand");
    map_b = map_block_bo(fpga_bo_b, "second operand");
    map_c = map_block_bo(fpga_bo_c, "result");

    // a ragged block leaves part of the operand buffers unwritten, and the
    // array multiplies the whole padded block regardless, so start from zeros
    memset(map_a, 0, BLOCK_BYTES);
    memset(map_b, 0, BLOCK_BYTES);

    // the buffer handles never change, so bind them once
    xrtRunSetArg(fpga_matmul_run, ARG_A_BLOCK, fpga_bo_a);
    xrtRunSetArg(fpga_matmul_run, ARG_B_BLOCK, fpga_bo_b);
    xrtRunSetArg(fpga_matmul_run, ARG_C_BLOCK, fpga_bo_c);
}

void fpga_cleanup(void) {
    xrtBOFree(fpga_bo_a);
    xrtBOFree(fpga_bo_b);
    xrtBOFree(fpga_bo_c);

    xrtRunClose(fpga_matmul_run);
    xrtKernelClose(fpga_matmul_krnl);
    xrtDeviceClose(fpga_dev);
}

f16_t *fpga_block_a(void) { return map_a; }
f16_t *fpga_block_b(void) { return map_b; }
const f16_t *fpga_block_c(void) { return map_c; }

void fpga_launch_block(int blk_m, int blk_k, int blk_n, int load_block_en) {
    // The DataMover reads and writes DDR directly and does not snoop the CPU's
    // caches, so these syncs are what actually make the staged operands visible
    // to it and the result visible here. Only push a block the array is going
    // to reload; a resident block is already in its large buffer.
    if (load_block_en & FPGA_LOAD_A) {
        xrtBOSync(fpga_bo_a, XCL_BO_SYNC_BO_TO_DEVICE,
                  (size_t)blk_m * blk_k * sizeof(f16_t), 0);
    }
    if (load_block_en & FPGA_LOAD_B) {
        xrtBOSync(fpga_bo_b, XCL_BO_SYNC_BO_TO_DEVICE,
                  (size_t)blk_k * blk_n * sizeof(f16_t), 0);
    }

    xrtRunSetArg(fpga_matmul_run, ARG_BLK_M, blk_m);
    xrtRunSetArg(fpga_matmul_run, ARG_BLK_K, blk_k);
    xrtRunSetArg(fpga_matmul_run, ARG_BLK_N, blk_n);
    xrtRunSetArg(fpga_matmul_run, ARG_LOAD_BLK_EN, load_block_en);

    xrtRunStart(fpga_matmul_run);
    xrtRunWait(fpga_matmul_run);

    // control.sv latches a DataMover error but keeps running to done_blk_comp
    // anyway, so a launch that reports one has written a partial result and
    // whatever training run is in flight is no longer meaningful
    uint32_t error = 0;
    if (xrtKernelReadRegister(fpga_matmul_krnl, FPGA_ERROR_REG_OFFSET, &error) == 0
        && error != 0) {
        fprintf(stderr, "fpga_launch_block: device reported an error on a "
                        "%dx%dx%d block (error register = 0x%x)\n",
                blk_m, blk_k, blk_n, error);
        exit(1);
    }

    xrtBOSync(fpga_bo_c, XCL_BO_SYNC_BO_FROM_DEVICE,
              (size_t)blk_m * blk_n * sizeof(f16_t), 0);
}

#endif
