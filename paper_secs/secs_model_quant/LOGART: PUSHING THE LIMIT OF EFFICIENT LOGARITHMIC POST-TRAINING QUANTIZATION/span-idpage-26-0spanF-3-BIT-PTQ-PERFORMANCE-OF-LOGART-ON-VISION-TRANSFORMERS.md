# <span id="page-26-0"></span>F 3-BIT PTQ PERFORMANCE OF LOGART ON VISION TRANSFORMERS

In the main body of the paper (Table [5\)](#page-8-1), we present the 4-bit weight quantization performance of LogART on various vision transformer models. To further evaluate the robustness and effectiveness of our method under more aggressive compression settings, this section provides the corresponding results for 3-bit per-channel weight quantization. The experimental setup follows that of 4-bit experiments in the main text, with only the target bitwidth changed. The comparison results are presented in Table [15.](#page-26-1)

Table 15: Comparison of top-1 accuracy and GPU runtime (in minutes) for 3-bit and 4-bit per-channel weight quantization with LogART on vision transformers.

<span id="page-26-1"></span>

| W     | ViT-Small |         | ViT-Base |         | DeiT-Tiny |         | DeiT-Base |         |
|-------|-----------|---------|----------|---------|-----------|---------|-----------|---------|
|       | Acc(%)    | Runtime | Accc(%)  | Runtime | Accc(%)   | Runtime | Accc(%)   | Runtime |
| FP16  | 81.39     | -       | 85.10    | -       | 72.16     | -       | 81.98     | -       |
| 4-bit | 81.06     | 6.7     | 85.02    | 10.9    | 71.62     | 5.6     | 81.92     | 10.9    |
| 3-bit | 79.56     | 5.9     | 84.54    | 9.9     | 70.21     | 5.0     | 81.51     | 10.1    |

#### <span id="page-27-0"></span>G DETAILS OF LOGART AE

#### G.1 LOGART AE DESIGN

While our proposed multi-base logarithmic quantizer enhances accuracy compared with fixed Log2 quantizer, it indeed introduces implementation challenges for the AE. Our LogART AE, shown in Figure 4, is customized to overcome these issues. It handles the computation involving  $\sqrt{2}$  through the integrated HAF, and also performs efficient on-chip decoding of the dynamic multi-base values through a novel encoding and decoding scheme. The functionality is realized through a design consisting of four primary modules: decoder, signed arithmetic logic, approximate computing logic, and shifter.

The decoder and its associated encoding scheme efficiently manage the complexity introduced by the multi-base logarithmic quantizer. In a 4-bit example, each weight is encoded with a 1-bit sign and a 3-bit value code  $(w\_code)$  that maps to one of the  $n_1$  base- $\sqrt{2}$  larger-valued codes and  $n_2$  smaller-valued base-2 codes. This is accompanied by 4-bit per-channel metadata that stores  $n_2$  and a parity flag  $(chk\_even)$ . The  $chk\_even$  is decided by the parity of the maximum base- $\sqrt{2}$  exponent. During computation, the decoder uses simple combinational logic to process this information and output control signals for the multiplier-free AE: a 1-bit sign, 3 shift bits  $(Shift\_bits)$ , and a 1-bit approximation flag  $(Approx\_flag)$ . Table 16 provides concrete examples of this process.

<span id="page-27-1"></span>Table 16: Encoding and decoding scheme examples.

| Weight     | Base                                                         | After Scaling | o. Encouring and              | Shift_bits | Approx_flag      |          |   |  |  |  |  |  |
|------------|--------------------------------------------------------------|---------------|-------------------------------|------------|------------------|----------|---|--|--|--|--|--|
|            | Example 1: $n_2$ =4, $chk_even$ =1, Scaling Factor= $2^{-8}$ |               |                               |            |                  |          |   |  |  |  |  |  |
| $2^{-8}$   | 2                                                            | 20            |                               | 000        |                  | 0        | 0 |  |  |  |  |  |
| $2^{-7}$   | 2                                                            | $2^1$         |                               | 001        |                  | 1        | 0 |  |  |  |  |  |
| $2^{-6}$   | 2                                                            | $2^2$         |                               | 010        |                  | 2        | 0 |  |  |  |  |  |
| $2^{-5}$   | 2                                                            | $2^3$         | Encoding ->                   | 011        | Decoding->       | 3        | 0 |  |  |  |  |  |
| $2^{-4.5}$ | $\sqrt{2}$                                                   | $2^{3.5}$     | Encoung >                     | 100        | Decouning >      | 3        | 1 |  |  |  |  |  |
| $2^{-4}$   | $\sqrt{2}$                                                   | $2^4$         |                               | 101        |                  | 4        | 0 |  |  |  |  |  |
| $2^{-3.5}$ | $\sqrt{2}$                                                   | $2^{4.5}$     |                               | 110        |                  | 4        | 1 |  |  |  |  |  |
| $2^{-3}$   | $\sqrt{2}$                                                   | $2^5$         |                               | 111        |                  | 5        | 0 |  |  |  |  |  |
|            |                                                              | Exam          | ple 2: n <sub>2</sub> =5, chk | _even=0,   | Scaling Factor=2 | $2^{-8}$ |   |  |  |  |  |  |
| $2^{-8}$   | 2                                                            | $2^{0}$       |                               | 000        |                  | 0        | 0 |  |  |  |  |  |
| $2^{-7}$   | 2                                                            | $2^1$         |                               | 001        |                  | 1        | 0 |  |  |  |  |  |
| $2^{-6}$   | 2                                                            | $2^2$         |                               | 010        |                  | 2        | 0 |  |  |  |  |  |
| $2^{-5.5}$ | $\sqrt{2}$                                                   | $2^{2.5}$     | Encoding ->                   | 011        | Decoding->       | 2        | 1 |  |  |  |  |  |
| $2^{-5}$   | $\sqrt{2}$                                                   | $2^3$         | Zincouning >                  | 100        | 2000amg >        | 3        | 0 |  |  |  |  |  |
| $2^{-4.5}$ | $\sqrt{2}$                                                   | $2^{3.5}$     |                               | 101        |                  | 3        | 1 |  |  |  |  |  |
| $2^{-4}$   | $\sqrt{2}$                                                   | $2^4$         |                               | 110        |                  | 4        | 0 |  |  |  |  |  |
| $2^{-3.5}$ | $\sqrt{2}$                                                   | $2^{4.5}$     |                               | 111        |                  | 4        | 1 |  |  |  |  |  |

The signed arithmetic logic operates based on the sign bit of the weight. For a positive weight, the activation is used directly. For a negative weight, the module computes the two's complement of the activation to perform the negation. This module is a crucial part of the arithmetic logic that ensures calculations involving negative weights are performed correctly and efficiently in hardware. The output of the signed arithmetic logic is then passed to the approximate computing logic.

The approximate computing logic and the shifter operate together to execute the final stage of the multiplier-free computation, supporting the multi-base feature of the LogART quantizer. The approximate computing logic is specifically designed to handle computations involving the  $\sqrt{2}$  base. With K=2 selected for our HAF due to its strong balance between hardware efficiency and accuracy, the multiplication by  $\sqrt{2}\approx 1.5$  is implemented as X+X/2. The approximate computing logic takes an enabling signal,  $Approx_-flag$ , from the decoder. If disabled, the activation passes through unchanged; otherwise, it adds the activation to half of its value. The shifter then performs the final multiplication-like step. It takes the output from the approximate computing logic and applies a left

bit-shift, with the number of shift positions determined by Shif t bits from the decoder. This single shift operation is equivalent to a multiplication by a power of 2, completing the computation and producing the final output value.

### G.2 LOGART AE BREAKDOWN

With all four modules working in concert, the LogART AE correctly computes the product for any weight in its multi-base system, thereby realizing a fully multiplier-free design. The overall area and power consumption for this complete AE are presented in Table [6.](#page-9-0) To provide a more granular analysis, Table [17](#page-28-0) further breaks down the hardware cost into the key functional modules, as illustrated in the block diagram in Figure [4\(](#page-5-0)e).

<span id="page-28-0"></span>Table 17: Area and power breakdown of the LogART AE.

|                                  | LogART AE                  | Signed         | Decoder        | Approx(Add)    | Shift          |
|----------------------------------|----------------------------|----------------|----------------|----------------|----------------|
| 2<br>Area (µm<br>)<br>Power (µW) | 53.2 (100%)<br>3.45 (100%) | 19.6%<br>20.7% | 17.5%<br>21.1% | 25.6%<br>20.0% | 37.3%<br>38.2% |

The hardware breakdown reveals that the shifter is the dominant contributor to hardware cost, accounting for 37.3% of the total area and 38.2% of the power. This result is expected, as the variable shifter serves as the core computational unit in a LogART AE. The next largest contributors are the modules that enable our multi-base scheme: the approximate computing logic, which implements the hardware-friendly operation for the <sup>√</sup> 2 base, is the second-largest by area (25.6%), while the decoder is the second-largest consumer of power (21.1%). Together, these two modules represent the necessary hardware investment to support the flexibility and high accuracy of our multi-base design. This breakdown validates our design choices, showing that even with this essential overhead, the LogART AE remains compact and power-efficient overall.

