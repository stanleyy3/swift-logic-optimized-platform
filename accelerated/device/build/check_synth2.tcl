open_project matmul_krnl_proj/matmul_krnl.xpr
reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
puts "===SYNTH1 STATUS==="
puts [get_property STATUS [get_runs synth_1]]
puts [get_property PROGRESS [get_runs synth_1]]
