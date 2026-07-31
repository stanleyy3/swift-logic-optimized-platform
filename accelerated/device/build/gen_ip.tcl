# gen_ip.tcl -- create the kernel project and generate the AXI DataMover IP
#
# Part/board match the KV260 platform this kernel will be linked against
# (xilinx_kv260_base_202520_1.xpfm), confirmed via `platforminfo`.

create_project -force matmul_krnl [file dirname [info script]]/matmul_krnl_proj
set_property part xck26-sfvc784-2LV-c [current_project]
set_property board_part xilinx.com:kv260_som:part0:1.4 [current_project]

create_ip -name axi_datamover -vendor xilinx.com -library ip -module_name axi_datamover_0

# Note: PG022 forces "Enable Store and Forward" on whenever the memory-map
# data width differs from the stream data width (64 vs. 16 here), regardless
# of c_*_include_sf -- confirmed via AMD doc search (PG022 "Advanced Options").
# CONTROL_INTERFACE.md's "store-and-forward: off" line was written before
# this constraint was known; SF is mandatory here and is what performs the
# 16<->64 bit width conversion, so this does not change any of the addressing
# invariants the doc describes. c_single_interface does not merge the mm2s/
# s2mm AXI ports into one bus (confirmed by inspecting the generated .veo) --
# the kernel keeps two separate m_axi masters (a read-only mm2s port, a
# write-only s2mm port), so it is left at its default (0).
set_property -dict [list \
    CONFIG.c_addr_width              {64} \
    CONFIG.c_m_axi_mm2s_addr_width   {64} \
    CONFIG.c_m_axi_s2mm_addr_width   {64} \
    CONFIG.c_m_axi_mm2s_data_width   {64} \
    CONFIG.c_m_axi_s2mm_data_width   {64} \
    CONFIG.c_m_axis_mm2s_tdata_width {16} \
    CONFIG.c_s_axis_s2mm_tdata_width {16} \
    CONFIG.c_mm2s_btt_used           {23} \
    CONFIG.c_s2mm_btt_used           {23} \
    CONFIG.c_include_mm2s_dre        {false} \
    CONFIG.c_include_s2mm_dre        {false} \
] [get_ips axi_datamover_0]

generate_target all [get_files axi_datamover_0.xci]
