# B. Challenges of Unified GEMM Design with FPMA

While FPMA offers a compelling path toward linearly scalable arithmetic, its direct application to GEMM architecture introduces several non-trivial challenges.

Challenge 1: Preserving accuracy under approximation error and low-bit quantization. FPMA introduces approximation errors by linearizing logarithmic relationships between operands. At the same time, achieving high accuracy at low-bit is intrinsically challenging: formats like FP4/FP3 suffer from subnormal distortion and narrow dynamic range. These difficulties compound with FPMA's approximation error, leading to severe accuracy degradation. Maintaining fidelity requires overcoming both the algorithmic limitations of low-bit floating-point and the approximation errors of FPMA.

Challenge 2: Efficient unified hardware architecture for bit-width scalability. Existing FPMA-based accelerators such as AxCore [50] use fixed datapaths optimized for a single mode (e.g., W4A16) and cannot be adapted to other precisions. Supporting multiple bit-widths within one GEMM engine requires a reconfigurable FPMA fabric that can scale exponent and mantissa handling while incorporating lightweight correction across modes. Without such architectural unification, multi-precision support would require duplicating hardware for each format, undermining the efficiency benefits of FPMA.

### IV. UNICORE OVERVIEW

UNICORE addresses these challenges by incorporating three key techniques that together enable a unified, multi-precision GEMM engine.

Technique 1: Lightweight Accuracy Preservation. To mitigate FPMA's approximation error and stabilize computation under low-bit quantization, UNICORE introduces a lightweight format-conversion and compensation pipeline. All inputs are first normalized into an expanded floating-point domain to eliminate subnormals and ensure consistent exponent alignment. Within the compute path, UNICORE applies dual-path compensation: fine-grained mantissa reconstruction restores low-order detail, while coarse-grained adjustment corrects high-order approximation error. These mechanisms enable FPMA-based GEMM to maintain high numerical fidelity even in aggressive 4-bit configurations.

Technique 2: Unified Bit-Width Scalable Architecture. UNICORE builds a unified GEMM engine on S-FPMA, a dynamically fusible FPMA primitive composed of uniform adder slices. Each slice handles a fixed low-bit segment (e.g., 4 bits), and wider precisions are formed by cascading slices with simple carry propagation. Because each additional bit contributes only one slice, the hardware cost and delay grow strictly linearly, O(n), with precision. This directly resolves the architectural unification challenge: a single datapath now supports diverse formats without redundant hardware.

Technique 3: Distribution-Adaptive Quantization with DynFP. UNICORE strengthens numerical fidelity in low-bit quantization through DynFP, a distribution-adaptive low-bit floating-point format. DynFP improves representational expressiveness by adjusting exponent—mantissa allocation and removing redundant encodings to match the asymmetric, heavy-tailed distributions in LLM layers. By improving the numerical quality of low-bit data, DynFP reinforces UNICORE's accuracy goals and complements the FPMA correction pipeline—further addressing Challenge 1, while supporting stable multi-bit-width execution (Challenge 2).

### V. ACCURACY-PRESERVED SCALABLE FPMA

### A. Eliminating Subnormals with Format Conversion

1) Subnormals break FPMA: In the IEEE floating-point standard, normal numbers (as shown in Equation 1) have a hidden leading 1 (i.e., 1+M) for the mantissa, which imposes a minimum representable magnitude:  $v_{\min,normal} = 2^{1-B}$ . As a result, values smaller than this threshold would jump directly

<span id="page-3-0"></span>![](_page_3_Figure_9.jpeg)

Fig. 4: Subnormal numbers from diverse FP4 variants are losslessly remapped to the normal range of a wider FP format.

to zero. To avoid this abrupt underflow, IEEE-754 introduces subnormal numbers, encoded with:

$$v_{\text{subnormal}} = (-1)^S \cdot 2^{1-B} \cdot (0+M)$$
 (5)

This gradually extends the representable range toward zero, providing smooth underflow. However, without the leading 1 for mantissa, their logarithmic behavior no longer satisfies FPMA's approximation (i.e.,  $\log_2(1+M_x)\approx M_x$ ). This is particularly problematic in low-bit formats (FP4/FP3), where up to 25–50% of representable values may be subnormals. As a result, FPMA can generate large, systematic inaccuracies for these inputs. Prior work, such as AxCore [50], mitigates this issue by remapping subnormals to nearby values in the normal-number domain. However, this conversion is itself approximate and inevitably introduces additional numerical noise.

2) Normalizing Subnormals via Bit-Width Expansion: To eliminate subnormal-induced FPMA errors, we convert low-bit floating-point operands into a wider format in which every representable value becomes a normal number. For example, a subnormal in any FP4 variant (E3M0, E2M1, E1M2) has a mantissa of the form (0+M). We normalize these values by left-shifting their mantissas until they fall within [1,2), while reducing the exponent by the same amount. This bit-width expansion preserves the exact numerical value but requires that the target format must keep at least as many mantissa bits as the source  $(M' \geq M)$  and provide enough exponent range to absorb up to M normalization shifts  $(B' \geq B + M)$ .

Figure 4 illustrates this process using FP4 variants (E3M0, E2M1, E1M2) and the target E3M2 format. The red crosses indicate subnormal values across several FP4 variants. Because these formats use short mantissas and narrow exponent ranges, many of their smallest values fall below the normal range. When projected into the expanded E3M2 format, each of these subnormals maps to a unique, exact normal representation, shown by the blue dots. E3M2 provides two additional mantissa bits and a larger exponent bias, ensuring that normalization never causes re-underflow and that all significant bits are preserved. As shown in the figure, the entire subnormal region of FP4 lies squarely within the normal range of E3M2. Even the smallest FP4 subnormal (0.5 in E1M2) shifts cleanly into E3M2's normal range after normalization.

This bit-expanded representation removes all subnormals before the FPMA computation begins, guaranteeing that every operand satisfies the assumptions of FPMA's logarithmic approximation. Importantly, all weights and activations remain stored and transferred in their original low-bit formats; the

<span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Fig. 5: While coarse-grained compensation [6] fails in low bit-width FPMA, fine-grained compensation can improve accuracy for both low and higher bit-width situations.

temporary E3M2 expansion is a precision-preserving internal re-encoding confined to the compute datapath, thereby retaining the system-level benefits of low-bit storage while enabling stable, subnormal-free arithmetic. Although this conversion will slightly expand the bit-width of the calculation datapaths, the resource overhead remains minimal for FPMA due to its addition nature.

