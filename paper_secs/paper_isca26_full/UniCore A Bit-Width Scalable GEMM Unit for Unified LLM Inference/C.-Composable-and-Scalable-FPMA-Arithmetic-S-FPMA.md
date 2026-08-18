# C. Composable and Scalable FPMA Arithmetic (S-FPMA)

To support multiple quantization formats within one architecture, the arithmetic unit must scale its bit-width without duplicating full datapaths. Existing FPMA designs [50], however, are rigidly tied to a single operand width and cannot expand to wider precisions. UNICORE addresses this by decomposing FPMA into a uniform slice (4-bit as the example), enabling the construction of wider FPMA operators simply by chaining slices through their carry interfaces. Because FPMA fundamentally performs an integer addition on the concatenated exponent—mantissa field, its datapath naturally decomposes into fixed-width adders, unlike multipliers whose partial-product structures do not compose linearly.

Figure 6a illustrates one such slice operating as a complete W4×A4 FPMA. Each 4-bit slice independently processes a local exponent–mantissa segment, subtracts the bias, and incorporates both fine-grained (FG) and coarse-grained (CG) compensation. FG compensation is injected at the LSB side of the slice to restore the missing lower-order bits of the product.

Wider precisions are obtained by fusing slices. In W8×A8 mode (Figure 6b), the lower slice processes the low 4 bits and

<span id="page-5-0"></span>![](_page_5_Figure_0.jpeg)

Fig. 7: UNICORE systolic array architecture.

the upper slice processes the next 4 bits, with the carry from the first feeding the second. FG compensation remains in the lowest slice so that the restored fine detail propagates through the fused adder chain, while CG compensation uses the empty bits of B to participate in addition to coarse mantissa bits. This extends naturally to W16×A16, where four slices form a 16-bit FPMA operator (Figure 6c). Carries ripple through all slices, and compensation remains correctly aligned. Because the fused datapath is still a single addition, area and delay scale linearly with bit-width.

S-FPMA therefore allows the same hardware to operate efficiently in W4A4, W8A8, and W16A16 modes, providing full utilization and multi-precision capability without duplicating FPUs or widening multiplier arrays.

### VI. UNICORE ARCHITECTURE

