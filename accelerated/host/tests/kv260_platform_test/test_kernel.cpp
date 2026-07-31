// Throwaway sanity-check kernel: does a v++ 2025.2 xclbin, built against
// xilinx_kv260_base_202520_1, actually load and run on the board's XRT 2.18.0?
// Not part of the real design - HLS generates the AXI4-Lite control plane and
// m_axi master so there's no hand-written RTL to debug here, only the
// toolchain/platform/board compatibility question.
extern "C" void test_kernel(int in, int *out) {
#pragma HLS INTERFACE s_axilite port=in     bundle=control
#pragma HLS INTERFACE m_axi     port=out    bundle=gmem offset=slave
#pragma HLS INTERFACE s_axilite port=out    bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    out[0] = in + 1;
}
