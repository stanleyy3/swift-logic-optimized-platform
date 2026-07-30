/**
 * fpga.h - XRT device/kernel lifecycle for the FPGA matmul accelerator
 */

#ifndef _FPGA_H_
#define _FPGA_H_

#include <xrt.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

extern xrtDeviceHandle fpga_dev;
extern xrtKernelHandle fpga_matmul_krnl;

// a run object is reusable, so one is opened up front and re-armed for each
// tile rather than opened and closed per launch
extern xrtRunHandle fpga_matmul_run;

// persistent TILE_DIM x TILE_DIM tile buffers, reused across every matmul call
extern xrtBufferHandle fpga_bo_a;
extern xrtBufferHandle fpga_bo_b;
extern xrtBufferHandle fpga_bo_c;

/**
 * @brief Opens the FPGA device, loads the xclbin, opens the matmul kernel,
 *        and allocates the persistent tile buffers
 *
 * Call once at program startup, before any call to tiled_mat_mat_mul_fpga
 */
void fpga_init(void);

/**
 * @brief Frees the tile buffers and closes the kernel/device
 *
 * Call once at program shutdown
 */
void fpga_cleanup(void);

#endif
