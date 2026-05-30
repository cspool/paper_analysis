# <span id="page-7-1"></span>5.5 Forward compatible: custom datatypes

We note that future designs may use different precision levels and non-standard formats for storage and/or intermediaries. TAIDL provides three mechanisms for handling custom datatypes.

(1) Manual precision and rounding control. XLA-HLO operators like reduce\_precision<sup>8</sup>, round\_nearest\_afz, clamp can be used before and after other operators to control numeric precision and rounding modes of tensor computations. Figure 13 shows an example of precisely representing floating-point conversion with an arbitrary number of exponent ( $E \ge 1$ ) and mantissa ( $M \ge 0$ ) bits, where an FP16 value of 0.395264 is converted to a less precise FP8 (E4M3) value of 0.40625. The FP8 (E4M3) is stored on FP16 registers with zero padding for precise functional simulation on CPU and GPU.

```
1 # %T0: FP16 [exponent='01101', mantissa='1001010011']
2 # %T1: FP8 E4M3 [exponent='0101', mantissa='101']
3 %T1:f16 = reduce_precision(%T0, E=4, M=3)
4 # Stored as [exponent='00101', mantissa='0000000101']
```

Figure 13: Usage of reduce\_precision to control numeric precision of floating-point data (LHS bits = E + M + 1 (for sign)).

- (2) Custom quantized formats. Architects can define their custom quantized formats using StableHLO type definition !quant.uniform with quantization parameters like storage type, zero-point, scale.
- (3) Custom bit-precise implementation. Since XLA-HLO is Turing-complete (§5.1) and also supports bit-vector semantics (§5.4), architects can define custom XLA-HLO functions and access them via XLA-HLO operator call. Alternatively, architects can define external C functions that perform bit-precise computation and access them via XLA-HLO operator custom\_call within a TAIDL definition.

<sup>4</sup>https://openxla.org/xla/operation\_semantics#select

<sup>&</sup>lt;sup>5</sup>https://openxla.org/xla/operation\_semantics#while

<sup>&</sup>lt;sup>6</sup>https://openxla.org/xla/operation\_semantics#convertelementtype

 $<sup>^7</sup> https://openxla.org/stablehlo/spec\#dot\_general$ 

<sup>8</sup>https://openxla.org/xla/operation\_semantics#reduceprecision

<span id="page-8-1"></span>![](_page_8_Figure_2.jpeg)

Figure 14: An ISA-specific test oracle library TAIDL-TO is generated from architect-provided ISA defined in TAIDL. A kernel programmer uses this library to write low-level kernels, which are then compiled and simulated using the generated test oracle.

