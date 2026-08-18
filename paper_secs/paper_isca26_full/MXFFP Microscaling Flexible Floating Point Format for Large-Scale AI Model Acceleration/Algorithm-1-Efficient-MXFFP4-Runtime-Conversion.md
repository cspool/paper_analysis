# Algorithm 1 Efficient MXFFP4 Runtime Conversion

```
Require: Block (or sub-block) \mathbf{x} = \{x_1, \dots, x_N\}
Ensure: Shared exponent E_b^{\text{shared}}, config cfg, quantized values \hat{\mathbf{x}}
  1: Get exponents: E_i \leftarrow \text{exponent}(x_i)
  2: E_b^{\text{MAX}^1} \leftarrow \max_i E_i
  3: Compute relative exponents: E_i^r \leftarrow E_i - E_b^{\text{MAX}}
 4: count_{E1} \leftarrow |\{E_i^r = 0\}|
5: count_{E2} \leftarrow |\{E_i^r \in \{-2, -3\}\}|
 6: if count_{E1}^2 > count_{E2} then
          cfg \leftarrow E1M2, \quad E_{element}^{MAX} \leftarrow 1
  7:
 8: else
          cfg \leftarrow E2M1, \quad E_{element}^{MAX} \leftarrow 2
10: end if
11: Unified shared exponent: E_b^{\text{shared}} \leftarrow E_b^{\text{MAX}} - E_{element}^{\text{MAX}}
12: Quantize each element:
                                   \hat{x}_i \leftarrow \operatorname{quant}(x_i/2^{E_b^{\text{shared}}})
13: return (E_h^{\text{shared}}, \text{cfg}, \hat{\mathbf{x}})
```

Config = 
$$\begin{cases} E1Mx, & count_{E1}^2 > count_{E2}, \\ E2Mx, & \text{otherwise.} \end{cases}$$
 (5)

After selecting the configuration, MXFFP computes the shared exponent in the same manner as MXFP. However, large blocks, which are internally divided into multiple sub-blocks, require an additional consideration. When a block is split into sub-blocks, each sub-block may select a different configuration (E1Mx or E2Mx). Because these configurations use different  $E_{element}^{\rm MAX}$  values (1 for E1Mx and 2 for E2Mx), they naturally require different shared exponents.

To avoid storing multiple shared exponents, we adopt a unified approach. We first compute and store the shared exponent assuming the E2Mx configuration. When the shared exponent is used, sub-blocks configured as E2Mx simply use this stored value directly, while sub-blocks configured as E1Mx subtract one from it to obtain their correct shared exponent. This simple approach allows all sub-blocks to reuse the same hardware path for exponent computation with negligible overhead, while still respecting the correct bias of each configuration.

Algorithm 1 summarizes the complete MXFFP runtime conversion procedure, requiring only a few additional operations beyond MXFP and incurring minimal hardware overhead. Despite its simplicity, the proposed conversion method closely matches oracle behavior with negligible overhead, as validated in Fig. 20.

## V. MXFFP TENSOR CORE

To fully exploit the benefits of MXFFP in hardware, several design considerations must be addressed. Since the MXFFP format employs multiple bit configurations (E1Mx and E2Mx), the compute pipeline must inspect the configuration of each block, properly align operand bits, and route them to compatible low-bit execution lanes. Furthermore, after computation, the output values must be reorganized into blocks and converted back into the MXFFP representation by deriving a shared exponent and selecting the appropriate configuration

![](_page_6_Figure_8.jpeg)

Fig. 11: MXFFP data flow and threadgroup mapping. Each block within a block set is assigned to specific threadgroups (TGs) and steps, enabling step-wise execution while maintaining configuration and shared exponent information.

for each block. In this section, we describe the hardware design based on these considerations and describe its key components and dataflow in detail.

#### A. Overview

Modern GPU Tensor Cores, widely used for ML acceleration, now support multiple low-precision floating-point formats such as FP4, FP6, FP8, and MXFP formats [22], [46]. MXFP formats apply block-wise shared scaling, meaning that Tensor Cores already provide hardware primitives for shared-exponent operation and format-specific low-bit conversions [2], [54]. Building on these capabilities, we design the MXFFP Tensor Core as an extension of the conventional Tensor Core pipeline, enabling flexible configurations without disrupting existing compute or data paths.

Fig. 12(a) shows an overview of MXFFP Tensor Core. MXFFP blocks are fetched into the matrix buffer and retained in their configured format until entering the dot-product unit. Inside the dot-product unit, a lightweight mapper interprets the configuration bits and aligns the incoming elements to the low-precision arithmetic units. After multiplication and accumulation, the results are rescaled using each block's shared exponent and accumulated into high-precision block outputs. Finally, the MXFFP converter reconstructs new MXFFP blocks by determining the shared exponent, following the MX methodology, and selecting the most suitable configuration for each block or sub-block.

Through this design, the MXFFP Tensor Core enables accurate computation across multiple low-bit configurations, while preserving the memory-bandwidth benefits of uniformly low-bit operands. Note that the MXFFP Tensor Core natively supports both block-level and sub-block-level execution. In the following description, we first present the design for block-level processing and then detail the additional mechanisms required for sub-block support.

#### B. Data Flow

Tensor Cores execute matrix multiply and accumulate (MMA) operations by dividing a  $16 \times 16$  output tile into eight  $4 \times 8$  submatrices [29], [51], [55]. Each submatrix is computed iteratively by a dedicated *threadgroup* (TG) consisting of four dot-product units as shown in Fig. 12(a). The computation proceeds over four steps, where each step accumulates partial sums to form the final output matrix as shown in Fig. 11.

Fig. 11 highlights the data flow of Step 1. Each threadgroup fetches its assigned tile from the operand buses into its matrix

![](_page_7_Figure_0.jpeg)

Fig. 12: MXFFP Tensor Core design.

buffer and performs operations on the corresponding input and output tiles. For example, TG0 multiplies the first  $4\times4$  tile of matrix A with the corresponding  $4\times8$  tiles of matrices B and C to produce the first  $4\times8$  tile of matrix D.

Building upon this mapping, MXFFP assigns each block within a block set to specific threadgroups and steps, as illustrated in Fig. 11. For instance, different subsets of threadgroups are responsible for distinct blocks of matrices A and B during Step 1, ensuring that all TGs operate on separate MXFFP blocks while maintaining consistent data reuse. Steps 2–4 follow the same mapping pattern, with each threadgroup accessing the corresponding blocks for its designated step.

To support this mapping, each threadgroup maintains two small registers: a *configuration register* (2B) and a *shared exponent register* (2B) for matricies A and B. The configuration register stores the configuration set of the block set, while the shared exponent register holds the shared exponent value for each block. At the beginning of a block-set operation, the configuration register is loaded and retained across the four steps, enabling each TG to select the appropriate configuration bit for the block it processes in each step. The shared exponent register is updated at every step, loading the scale value corresponding to the assigned block for that step.

## C. Dot-product Unit

The dot-product unit performs the core fused multiply—accumulate operations using tiles fetched from the matrix buffer [17], [29], [51], [55]. Each unit receives a  $4\times1$  fragment from matrix A and a  $1\times4$  fragment from matrix B, producing a four-element dot product every cycle. Since operands are stored in MXFFP format, which supports multiple exponent—mantissa configurations under the same bit-width, the unit must first expand and align incoming elements before computation.

Fig. 12(b) illustrates the MXFFP dot-product datapath for MXFFP4. Each MXFFP operand is fed through a lightweight *bit mapper* prior to entering the arithmetic core. The mapper is implemented as a compact multiplexer array that reinterprets the incoming 4-bit word according to the configuration bit. When the configuration is E1M2, the mapper routes only the

most-significant exponent bit into the exponent field and fills the remaining positions using the most-significant mantissa bits. When the configuration is E2M1, two exponent bits are forwarded directly, and a single mantissa bit is appended to form the final aligned format.

To accommodate both mapped representations, the FP4 (E2M1) arithmetic core is minimally extended to support an internal E2M2 format. Specifically, one additional mantissa bit is added to the significand path. The corresponding normalization unit is widened to handle the increased precision, while exponent biasing and range handling remain unchanged. The area and power overhead to support this is negligible since only the significand pipeline is widened (see Section VI-D).

After computation, each element is multiplied by its corresponding block's shared exponent prior to accumulation, consistent with MX processing semantics. Accumulated results are stored in high-precision accumulators (e.g., FP16 or BF16), ensuring numerical stability across the full dot-product sequence. Although the example focuses on MXFFP4, the same mechanism applies to MXFFP6 and MXFFP8.

