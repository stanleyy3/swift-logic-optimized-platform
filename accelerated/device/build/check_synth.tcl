# check_synth.tcl -- cheap connectivity check: elaborate/synthesize
# matmul_krnl standalone before paying for a full package_xo/v++ link.
#
# Run after gen_ip.tcl has created matmul_krnl_proj with the DataMover IP.

set script_dir [file dirname [info script]]
set rtl_dir [file normalize $script_dir/../rtl]

open_project $script_dir/matmul_krnl_proj/matmul_krnl.xpr

add_files -norecurse [glob $rtl_dir/*.sv]
update_compile_order -fileset sources_1
set_property top matmul_krnl [current_fileset]
update_compile_order -fileset sources_1

synth_design -top matmul_krnl -mode out_of_context
