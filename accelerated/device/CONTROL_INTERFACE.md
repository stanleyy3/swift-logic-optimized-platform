# Control interface contract

`systolic_array.sv` is the datapath only. It has no AXI4-Lite slave and does not
instantiate the AXI DataMover it drives, so it cannot be packaged as an XRT
kernel on its own. This file pins down what the wrapper around it has to
present, because the host codes against exactly this and nothing in either half
can detect a mismatch at run time.

The host side lives in `accelerated/host/fpga.c`; the argument indices and the
error register offset there must match the tables below.

## Block diagram

```
              AXI4-Lite (from PS)
                     |
              [ control regs ]  <- ap_ctrl_hs + argument shadow registers
                     |
              systolic_array.sv
                 |         |
     mm2s cmd/data/sts   s2mm cmd/data/sts
                 |         |
            [ AXI DataMover (PG022) ]
                     |
                AXI4 master -> HP port -> DDR
```

## Register map

Standard `ap_ctrl_hs` layout, so that `xrtRunSetArg` / `xrtRunStart` /
`xrtRunWait` work unmodified.

| Offset | Arg | Name | Access | Drives |
|---|---|---|---|---|
| 0x00 | — | `CTRL` | R/W | `ap_start` (bit 0), `ap_done` (1), `ap_idle` (2), `ap_ready` (3) |
| 0x04 | — | `GIER` | R/W | global interrupt enable (unused) |
| 0x08 | — | `IP_IER` | R/W | interrupt enable (unused) |
| 0x0C | — | `IP_ISR` | R/W | interrupt status (unused) |
| 0x10 | 0 | `a_block` [31:0] | W | `a_block_addr` |
| 0x14 | 0 | `a_block` [63:32] | W | `a_block_addr` |
| 0x1C | 1 | `b_block` [31:0] | W | `b_block_addr` |
| 0x20 | 1 | `b_block` [63:32] | W | `b_block_addr` |
| 0x28 | 2 | `c_block` [31:0] | W | `c_block_addr` |
| 0x2C | 2 | `c_block` [63:32] | W | `c_block_addr` |
| 0x34 | 3 | `blk_m` | W | `blk_m` |
| 0x3C | 4 | `blk_k` | W | `blk_k` |
| 0x44 | 5 | `blk_n` | W | `blk_n` |
| 0x4C | 6 | `load_block_en` | W | `load_block_en[1:0]` |
| 0x54 | 7 | `error` | R | `error_blk_comp`, zero-extended |

`load_block_en` bit 0 loads the A block, bit 1 the B block. Clearing bit 0
reuses whatever A block is already in the large buffers, which is only valid
while `blk_m` and `blk_k` are unchanged from the launch that loaded it.

## Two things that will otherwise hang or corrupt silently

**`ap_start` is a self-clearing pulse; `start_blk_comp` is a level.**
`control.sv` requires `start_blk_comp` held high from launch until
`done_blk_comp`, and `S_DONE` only returns to `S_IDLE` once it deasserts
(`control.sv:455`). The wrapper needs a `busy` flop:

```
busy            <- set by ap_start, cleared by done_blk_comp
start_blk_comp  =  busy
ap_done         <- sticky, set by done_blk_comp, cleared on read
ap_idle         =  ~busy
```

Driving `start_blk_comp` straight from a one-cycle `ap_start` leaves the FSM in
`S_IDLE` and the run never completes.

**The block descriptors must be stable from start to done** (`control.sv:293`).
Capture all seven argument registers into shadow flops on the `ap_start` edge
and feed `systolic_array` from the shadows, so a write that lands mid-run
cannot move the block geometry under a running FSM.

## DataMover configuration

| Setting | Value | Why |
|---|---|---|
| Memory map data width | 64 (or wider) | HP port width |
| Stream data width | 16 | `MUL_WIDTH`; `mm2s_tdata`/`s2mm_tdata` are this wide |
| Address width | 64 | `` `PHYSICAL_ADDR_WIDTH `` |
| BTT width | 23 | the `mm2s_cmd_bytes`/`s2mm_cmd_bytes` field |
| Command/status interface | AXI4-Stream | what the command word packing expects |
| Data Realignment Engine | **disabled** | `systolic_array.sv:165-181` hardcodes `DRR=0`, `DSA=0` |
| Store-and-forward | off | not needed; the design backpressures |

Disabling the DRE means addresses and byte counts must be aligned to the
memory-map data width. That holds for free here and is worth preserving:

- XRT buffer objects are page-aligned, and the host always passes a buffer's
  base address rather than an offset into it
- every launch dimension is a multiple of `ARRAY_DIM` = 8, so an operand block
  is a multiple of `8 * 8 * 2` = 128 bytes
- the writeback issues one command per output tile, always `TILE_BYTES` = 128

If anyone later stages several blocks into one buffer at element offsets, that
invariant breaks and the DRE has to be turned on.

## Packaging

The three pointer arguments must be declared as `global` memory pointers bound
to the DataMover's `m_axi` bundle in `kernel.xml`. That is what makes
`xrtRunSetArg(run, i, bo)` write the buffer's **device** address into the
register - the DataMover masters DDR itself and cannot see host virtual
addresses. It is also what gives `xrtKernelArgGroupId` a memory bank to report,
which is how `fpga_init` places the buffers in the bank the DataMover reaches.

```xml
<arg name="a_block"       addressQualifier="1" id="0" port="m_axi_gmem" size="0x8" offset="0x10" hostOffset="0x0" hostSize="0x8" type="ushort*"/>
<arg name="b_block"       addressQualifier="1" id="1" port="m_axi_gmem" size="0x8" offset="0x1C" hostOffset="0x0" hostSize="0x8" type="ushort*"/>
<arg name="c_block"       addressQualifier="1" id="2" port="m_axi_gmem" size="0x8" offset="0x28" hostOffset="0x0" hostSize="0x8" type="ushort*"/>
<arg name="blk_m"         addressQualifier="0" id="3" port="s_axi_control" size="0x4" offset="0x34" hostOffset="0x0" hostSize="0x4" type="uint"/>
<arg name="blk_k"         addressQualifier="0" id="4" port="s_axi_control" size="0x4" offset="0x3C" hostOffset="0x0" hostSize="0x4" type="uint"/>
<arg name="blk_n"         addressQualifier="0" id="5" port="s_axi_control" size="0x4" offset="0x44" hostOffset="0x0" hostSize="0x4" type="uint"/>
<arg name="load_block_en" addressQualifier="0" id="6" port="s_axi_control" size="0x4" offset="0x4C" hostOffset="0x0" hostSize="0x4" type="uint"/>
<arg name="error"         addressQualifier="0" id="7" port="s_axi_control" size="0x4" offset="0x54" hostOffset="0x0" hostSize="0x4" type="uint"/>
```

The kernel must be openable exclusively: `fpga_init` uses
`xrtPLKernelOpenExclusive`, because `xrtKernelReadRegister` refuses to read a
shared kernel and that read is the only way the error flag gets back to the
host.

## Elaboration parameters

The host mirrors these in `accelerated/host/config.h` (`FPGA_ARRAY_DIM`,
`FPGA_LARGE_BUFFER_DIM`) and in `ACC_WIDTH`/`ACC_LSB` in `quant.h`. Changing one
side without the other produces wrong results with no diagnostic.

| Parameter | fp16 (current) | fp32 |
|---|---|---|
| `LARGE_BUFFER_DIM` | 256 | 256 |
| `ARRAY_DIM` | 8 | 8 |
| `MUL_WIDTH` | 16 | 32 |
| `EXP_WIDTH` | 5 | 8 |
| `ACC_WIDTH` | 48 | 64 |
| `ACC_LSB` | -24 | -52 |
