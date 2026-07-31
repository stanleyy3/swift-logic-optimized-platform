open_project matmul_krnl_proj/matmul_krnl.xpr
puts "===ALL SOURCES==="
foreach f [get_files] { puts $f }
puts "===IPs==="
foreach ip [get_ips] { puts "$ip -> [get_property IP_FILE $ip]" }
