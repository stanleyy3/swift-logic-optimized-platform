# package_kernel.tcl -- package matmul_krnl as a Vitis RTL kernel (.xo)
#
# Run after gen_ip.tcl (DataMover IP + project) and check_synth.tcl (adds RTL
# sources, confirms matmul_krnl elaborates/synthesizes cleanly).

set script_dir [file dirname [info script]]
open_project $script_dir/matmul_krnl_proj/matmul_krnl.xpr

set_property top matmul_krnl [current_fileset]
update_compile_order -fileset sources_1

ipx::package_project -root_dir $script_dir/matmul_krnl_ip -vendor xilinx.com \
    -library user -taxonomy /UserIP -import_files -set_current true

set core [ipx::current_core]

puts "===INFERRED BUS INTERFACES==="
foreach bif [ipx::get_bus_interfaces -of_objects $core] {
    puts "  [get_property NAME $bif] : [get_property ABSTRACTION_TYPE_VLNV $bif]"
}

# associate each inferred AXI interface with ap_clk, and drop FREQ_HZ so v++
# (not this IP) drives the clock frequency association during linking
foreach bif_name {s_axi_control m_axi_gmem m_axi_gmem1} {
    if {[llength [ipx::get_bus_interfaces $bif_name -of_objects $core]]} {
        ipx::associate_bus_interfaces -busif $bif_name -clock ap_clk $core
        ipx::remove_bus_parameter FREQ_HZ [ipx::get_bus_interfaces $bif_name -of_objects $core]
    } else {
        puts "WARNING: expected bus interface $bif_name not found after packaging"
    }
}

set_property core_revision 1 $core
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core

package_xo -force \
    -kernel_name matmul_krnl \
    -ip_directory $script_dir/matmul_krnl_ip \
    -kernel_xml $script_dir/../kernel.xml \
    -xo_path $script_dir/matmul_krnl.xo
