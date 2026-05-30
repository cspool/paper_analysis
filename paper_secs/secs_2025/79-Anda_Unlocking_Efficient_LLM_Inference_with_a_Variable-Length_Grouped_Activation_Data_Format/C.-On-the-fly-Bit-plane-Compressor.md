# C. On-the-fly Bit-plane Compressor

The bit-plane compressor (BPC) is a critical component of the Anda architecture, enabling on-the-fly conversion of FP16 activation values into the compressed Anda format. It efficiently addresses the challenges of variable-length Anda activation storage and transfer in LLM inference by processing a large number of activation values in parallel and outputting them in a bit-serial manner.

Fig. 12 illustrates the architecture of the proposed BPC. It consists of 16 parallel lanes, each capable of processing 64 grouped FP16 values simultaneously. Within each lane, the FP field extractor decomposes the FP16 inputs into their sign,

![](_page_7_Figure_0.jpeg)

Fig. 12. The architecture of the on-the-fly bit plane compressor and the mantissa alignment process performed in the parallel-to-serial mantissa aligner.

exponent, and mantissa components. The maximum exponent catcher identifies the maximum exponent within a grouped lane, and then calculates the difference of each exponent to the shared maximum exponent.

The core of the compression process lies in the mantissa alignment performed by the parallel-to-serial mantissa aligner. As shown in Fig. 12, in each cycle, each element's exponent difference decreases by one until it reaches zero. When the exponent difference is zero, the most significant mantissa bit of that element should be shifted out each cycle; otherwise, it remains unchanged and output zero. The shifted-out bits among each element in the lane are packed into the bit-plane aligned mantissa. This process continues for multiple cycles until the number of output bit-planes reaches the configurable mantissa length. The parallel-to-serial mantissa alignment process generates compressed bit-planes directly. The resulting bit-serial output, along with the sign bits and shared maximum exponents from all lanes, is passed to the data packager unit. This unit assembles the final compressed output in a format compatible with the proposed bit-plane compression scheme. The proposed bit-serial mantissa aligner is more areaefficient compared to existing bit-parallel aligners [32], [85], requiring only a comparator and shifter. In contrast, bit-parallel designs need multiple shifters and comparators for single-cycle dynamic shifting [15]. While our bit-serial aligner introduces some latency, it can largely overlap with APU computations, with little impact on overall system performance.

