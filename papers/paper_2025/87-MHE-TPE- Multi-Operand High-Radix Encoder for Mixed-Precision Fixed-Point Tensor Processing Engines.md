## **MHE-TPE: Multi-Operand High-Radix Encoder for Mixed-Precision Fixed-Point Tensor Processing Engines** 

Qizhe Wu USTC Hefei, China wqz1998@mail.ustc.edu.cn 

Zhichen Zeng University of Washington 

Seattle, USA zczeng@uw.edu 

Linfeng Tao USTC Hefei, China tlf@mail.ustc.edu.cn 

Letian Zhao USTC Hefei, China zhaolt@mail.ustc.edu.cn 

Jinyi Zhou 

USTC Hefei, China zjy2017@mail.ustc.edu.cn 

Huawen Liang USTC Hefei, China lhw233@mail.ustc.edu.cn 

Xin Zhang 

Zbit Semiconductor Shanghai, China zhangxin19961129@163.com 

Wei Yuan USTC Hefei, China yuanwei501240@mail.ustc.edu.cn 

Zhanhe Hu 

USTC Hefei, China zhanhe_hu@mail.ustc.edu.cn 

Jiuru Zhu USTC Hefei, China zjr_030720@mail.ustc.edu.cn 

Zekang Cheng USTC Hefei, China chengzk@mail.ustc.edu.cn 

Xiaotian Wang Raytron Technology Suzhou, China wxtdsg@mail.ustc.edu.cn 

Xi Jin[∗] USTC Hefei, China jinxi@ustc.edu.cn 

## **Abstract** 

Fixed-point general matrix multiplication (GEMM) is pivotal in AI-accelerated computing for data centers and edge devices in GPU and NPU tensor processing engines (TPEs). This work exposes two critical limitations in typical spatial mixed-precision TPEs: ❶ Redundant partial products (PPs) reduction in PE multipliers across temporal and spatial domains in MAC arrays. ❷ Compute density imbalance: when the operand bit-width is reduced by one-half, the throughput of GEMM only doubles, which is half of the theoretical 4× improvement. To address these challenges. First, we design a multi-operand high-radix encoder based on vector inner products, which reduces PPs for vector reduction by half through decoding. Second, we establish a three-stage computational paradigm for TPE’s microarchitecture, comprising bit-slice encoding, PPs generation, and PPs reduction, which enables bit-width reconfiguration in unified hardware. Our approach decomposes the mixed-precision mapping process in TPEs into two components: temporal mapping of multi-precision multiplicands and spatial mapping of multipliers, 

∗Corresponding author. 

achieving balanced computational density. Implementation results based on the UMC 22nm process demonstrate that this architecture outperforms other solutions in critical metrics, including the mixed-precision support range (INT2 ∼ INT32 combinations), area efficiency, and energy efficiency. 

## **CCS Concepts** 

- **Hardware** → **Arithmetic and datapath circuits** . 

## **Keywords** 

High-Radix Encoder, Fixed-Point Tensor Processing Engine, MixedPrecision Computing 

## **ACM Reference Format:** 

Qizhe Wu, Jinyi Zhou, Zhanhe Hu, Zhichen Zeng, Huawen Liang, Jiuru Zhu, Linfeng Tao, Xin Zhang, Zekang Cheng, Letian Zhao, Wei Yuan, Xiaotian Wang, and Xi Jin. 2025. MHE-TPE: Multi-Operand High-Radix Encoder for Mixed-Precision Fixed-Point Tensor Processing Engines. In _58th IEEE/ACM International Symposium on Microarchitecture (MICRO ’25), October 18–22, 2025, Seoul, Republic of Korea._ ACM, New York, NY, USA, 15 pages. https: //doi.org/10.1145/3725843.3756101 

## **1 Introduction** 

This work is licensed under a Creative Commons Attribution 4.0 International License. _MICRO ’25, Seoul, Republic of Korea_ 

© 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1573-0/25/10 https://doi.org/10.1145/3725843.3756101 

In the field of edge computing AI deployment, fixed-point model inference has become the dominant paradigm in the technological ecosystem [13, 31, 32, 51, 52]. This work identifies systematic computational redundancy in current spatial GEMM architectures 

1 

1625 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea Q. Wu, J. Zhou, Z. Hu, Z. Zeng, H. Liang, J. Zhu, L. Tao, X. Zhang, Z. Cheng, L. Zhao, Y. Wei, X. Wang, and X. Jin 

**==> picture [505 x 162] intentionally omitted <==**

**----- Start of picture text -----**<br>
B1,T B2,T BN,T Temporal Dimension<br>105Multiplicand(A 0 1 1 0 1 0 0 1 1,T) Multiplier(B1,T) Multiplicand(A93 0 1 0 1 1 1 0 1, 1 T+1) Multiplier(B1,T+1) Essentially performing the same calculation<br>A1,T PE1 PE PE B1,T MBE Encoder MBE Encoderai+2 ai+1 MBE Encoderai ai-1 MBE Encoderai-2 Map MBE Encoder MBE Encoderai+2 ai+1 MBE Encoderai ai-1 MBE Encoderai-2 Map 2B MBE Encoder 1,T+B1,T+1 -B MBE Encoder 1,T+2B1,T+1-(2B MBE Encoder 1,T+B1,T+1) MBE Encoder B1,T+B1,T+1 Map<br>0 B -B 2B -2B 0 B -B 2B -2B 0 B -B 2B -2B 0 B -B 2B -2B 0 B -B 2B -2B 0 B -B 2B -2B 0 B -B 2B -2B 0 B -B 2B -2B 0 B -B 2B -2B 0 B -B 2B -2B 0 B -B 2B -2B 0 B -B 2B -2B<br>A2,T PE2 PE PE A1,T 2B1, ShiftMux T<<6 -B1, Compressor TreeShiftMux T<<4 -2B1, ShiftMux T<<2 B1, Shift T Mux <<0 B1, ShiftMux T+1<<6 2B1, Compressor TreeShiftMux T+1<<4 -B1, ShiftMux T+1<<2 B1,T+1 ShiftMux <<0 <<6 ShiftMux Compressor Tree <<4 ShiftMux <<2 ShiftMux <<0 ShiftMux<br>1 Full Adder 105B1,T Full Adder 93B1,T+1 Full Adder 105B1,T + 93B1,T+1<br>AN,T PE PE PE Cycle T Cycle T+1 Cycle T and Cycle T+1<br>N (a) A2,T 2 -101Multiplicand(A 1 0 0 1 1 0ai+21 N, 1 T+N ai+1 ) Multiplier(B ai ai-1 1,T a ) i-2 Multiplicand(A-41 1 1 0 1 0 1 N, a1i+2 T+N+1 1 ai+1 ) Multiplier(B ai ai-1 1, a T+1 i-2 ) -(2B1,T+B1,T+1) 2B1,T+B1,T+1 -B1,T+2B1,T+1 -(B1,T+B1,T+1)<br>a2i+10000 a11002i a01012i-1 Operation+2B+B+B0 AN,T -2B 0MBE EncoderB 1, ShiftMux-B T<<6 2B -2B 0MBE Encoder 2B B 1, Compressor TreeShiftMux-B T<<4 2B -2B 0 -B MBE EncoderB 1, ShiftMux-B T<<2 2B -2B 0 -B MBE EncoderB 1, ShiftMux-B T<<0 2B -2B Map 0MBE Encoder -B B 1, ShiftMux-B T+1 2B <<6 -2B 0MBE Encoder B B 1, Compressor TreeShiftMux-B T+1 2B <<4 -2B 0MBE Encoder 2B B Shift 1, Mux-B T+1 2B <<2 -2B 0MBE Encoder -B B 1, ShiftMux-B T+1 2B <<0 -2B Map 0MBE EncoderB <<6 ShiftMux-B 2B -2B 0MBE EncoderB Compressor Tree <<4 ShiftMux-B 2B -2B 0MBE EncoderB <<2 ShiftMux-B 2B -2B 0MBE EncoderB <<0 ShiftMux-B 2B -2B Map<br>111 100 001 -2B-B-B N Full Adder -101B1,T -41B Full Adder 1,T+1 Full Adder -101B1,T  - 41B1,T+1<br>1 1 1 0 Cycle T+N Cycle T+N+1 Cycle T+N and Cycle T+N+1<br>(b) (c)<br>Spatial Dimension<br>**----- End of picture text -----**<br>


**Figure 1: (a) OS systolic array,** _𝐴𝑖,𝑇_ **and** _𝐵𝑖,𝑇_ **denote the multiplicand and multiplier input to the** _𝑖[𝑡ℎ]_ **-row and** _𝑖[𝑡ℎ]_ **-column PE at cycle** _𝑇_ **. (b) MBE truth table, where** _𝑎𝑖_ **represents the** _𝑖_ **-th binary bit of multiplicand A, and B is the multiplier. (c) Analysis of redundant computations in the PE microarchitecture of the spatial accelerator from temporal and spatial dimensions.** 

[1, 3, 12, 18, 19, 26, 30, 33, 45] under the data-reuse-oriented design paradigm, manifesting as redundant PPs reduction across both temporal and spatial dimensions within and between processing elements (PEs). We adopt the output stationary (OS) systolic array (shown in Fig. 1(a))[53] as a case study for empirical analysis. 

In the design of modern multipliers, the modified Booth encoder (MBE)[35, 42] is typically employed to reduce the number of PPs by half in multiplication [11, 21], and the encoding table is shown in Fig. 1(b). **At the temporal dimension within a single PE microarchitecture** : In Fig. 1(c) during cycle _𝑇_ , _𝑃𝐸_ 1 receives _𝐴_ 1 _,𝑇_ (value 105) and generates 4 PPs coefficients {2, -1, -2, 1} with bitweight ( _𝑏𝑤_ ) factors (2[6] _,_ 2[4] _,_ 2[2] _,_ 2[0] ) through MBE. These are summed via compressor tree [46] and full adder to obtain 105 × _𝐵_ 1 _,𝑇_ . In cycle _𝑇_ + 1, _𝐴_ 1 _,𝑇_ +1 (value 93) produces coefficients {1, 2, -1, 1} through MBE conversion, ultimately yielding 93 × _𝐵_ 1 _,𝑇_ +1. The accumulator temporally sums these multiplication results across two clock cycles (105 × _𝐵_ 1 _,𝑇_ + 93 × _𝐵_ 1 _,𝑇_ +1). Notably, inherent redundancy emerges in the _𝑏𝑤_ coefficient reductions. As shown in Eq. 1 _𝑃𝐸_ 1, the _𝑏𝑤_ of 2[6] and 2[2] terms show redundancy through inverse-signed identical reductions (2 _𝐵_ 1 _,𝑇_ + _𝐵_ 1 _,𝑇_ +1), despite completely independent bit-slice distributions in _𝐴_ 1 _,𝑇_ and _𝐴_ 1 _,𝑇_ +1. 

**==> picture [241 x 60] intentionally omitted <==**

**At spatial dimension** : This redundant reduction exhibits significant prevalence among column PEs. In Fig. 1(c), within cycles _𝑇_ + _𝑁_ and _𝑇_ + _𝑁_ + 1, the _𝑃𝐸_ 1 ∼ _𝑃𝐸𝑁_ processes systolic-transmitted _𝐵_ 1 _,𝑇_ and _𝐵_ 1 _,𝑇_ +1, and all PE generating 4 PPs groups showing high similarity with the reduction patterns in the _𝑃𝐸_ 1 show in Eq. 1, as indicated by identically colored dashed regions in Fig. 1(c). 

Crucially, this redundancy at the temporal and spatial dimensions persists systematically across all PE columns, independent of multiplicand _𝐴_ variations, arising from dual mechanisms: 

> ❶ **Temporal causation within PEs:** (1) Discrete mapping in MBE allows equivalent coefficient generation from different bit-slice combinations (e.g., {0 _, 𝐵,_ − _𝐵_ } in Fig. 1(b) correspond to two different bit-slice sets). (2) Symmetric MBE coefficient set {−2 _𝐵,_ − _𝐵,_ 0 _, 𝐵,_ 2 _𝐵_ } constrains linear combination space, enabling identical computations from different encoding combinations (e.g., 2 _𝐵_ 1 _,𝑇_ + _𝐵_ 1 _,𝑇_ +1 vs. −(2 _𝐵_ 1 _,𝑇_ + _𝐵_ 1 _,𝑇_ +1) in _𝑃𝐸_ 1). 

❷ **Spatial causation across PEs:** Systolic propagation of _𝐵_ 1 _,𝑇_ and _𝐵_ 1 _,𝑇_ +1 through all column PEs enforces fixed linear combinations from coefficient set {−2 _𝐵,_ − _𝐵,_ 0 _, 𝐵,_ 2 _𝐵_ } under different _𝑏𝑤_ factors. This process, decoupled from _𝐴_ ’s bit distribution, inherently causes cross-PE redundant reductions. 

**==> picture [217 x 215] intentionally omitted <==**

**----- Start of picture text -----**<br>
P1,T<br>P1,T P2,T PN,T P1,T+1<br>A1,T<br>A1,T PE1 PE PE A1,T B1<br>1<br>B1<br>A2,T PE PE PE 1<br>2 A2,T<br>A2,T B2<br>2<br>AN,T PE PE PE 2 B2<br>P1,T+ A1,T-1B1+A2,TB2<br>(a) <<<br>Ai,T B1 B2 B3 B4 BN-1 BN <<<br>Ai,T+K 0 10110110 A 010 1 010 1 B1B1<br>0 A 1 1,T+K 0 1 1 1 0 1 B2<br>0 1 0 1 1 A 1 0 2 1<br>1 01 A 001 2,T+K 011 A 011 3 011 1 B3<br>1 11 A 011 3,T+K 001 A 101 N 111 1 BN<br>AN,T+K<br>(b) (c)<br>+<br>+<br>Spatial<br>Spatial<br>Spatial<br>**----- End of picture text -----**<br>


**Figure 2: (a) Weight stationary (WS) systolic array (where** _𝑃𝑖,𝑇_ **denotes the partial sum input to the** _𝑖_ **-th column PE at cycle** _𝑇_ **). (b) Multiplier-adder tree. (c) Bit-serial architecture.** 

2 

1626 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

MHE-TPE: Multi-Operand High-Radix Encoder for Mixed-Precision Fixed-Point Tensor Processing Engines 

|**_(a)_ IBM NPU FX-Point PE Engine**|**_(b)_ Samsung NPU FX-Point PE Engine**|
|---|---|
|**8 × INT4 Booth**<br>**Multipliers**<br>**16 × INT2**<br>**Multipliers**<br>**Select Network**<br>**Reduction Tree**|**Data Re-mapper**<br>**8 × 8**<br>**Mul**<br>**8 × 4**<br>**Mul**<br>**8 × 4**<br>**Mul**<br>**4 × 4**<br>**Mul**<br>**Select Network**<br>**Reduction Tree**|



**Figure 3: (a) IBM 7-nm NPU fixed-point PE engine [23]. (b) Samsung 4-nm fixed-point PE engine in mobile SoC [36].** 

**Table 1: Area of fixed-point multipliers across precision levels. 4-RT (8): An 8-bit input reduction tree with 4 inport.** 

|**Component**|**Area(**_𝜇𝑚_2**)**|**Area(**_𝜇𝑚_2**)**|**Area(**_𝜇𝑚_2**)**|**MOPS/**_𝜇𝑚_2|**MOPS/**_𝜇𝑚_2|**MOPS/**_𝜇𝑚_2|
|---|---|---|---|---|---|---|
||**Logic**|**DFF**|**Total**|**INT4**|**INT8**|**INT16**|
|**INT 4 MUL**|23.32|23.52|46.84|21.25|/|/|
|**4×INT 4 MUL**|93.28|94.08|247.63|16.15|4.03|<1.01|
|**4-RT(8)**|36.75|23.52|||||
|**INT8 MUL**|94.86|47.04|141.9|/|7.04|/|
|**4×INT8 MUL**|379.44|188.16|693.43|/|5.76|1.44|
|**4-RT(16)**|78.79|47.04|||||
|**INT16 MUL**|394.54|94.57|489.11|/|/|2.04|



vs INT8: 624 TOPS). This computational density degradation fundamentally constrains mixed-precision acceleration potential. The main contributions of this paper are reflected as follows: 

- (1) **In computational principles level** , this work proposes a cross-PE dual-operand encoding algorithm suitable for GEMM spatial accelerators. This enables PEs to share vector partial product lookup tables, thereby halving the PPs reduction computation within GEMM. 

- (2) **In mixed-precision TPE architecture level** , we propose a tensor computing framework structured as bit-sliced encoding → vector partial product generation → cross-dimensional reduction. By merging the bit-sliced reduction dimension inherent in the multiplier microarchitecture with the spatial reduction dimension of vector inner products, this unified compute engine achieves operand precision scaling from INT2 to INT32. 

- (3) **In TPE array scheduling level** , we decompose mixedprecision GEMM operand mapping into temporal and spatial domains. And we achieve a balanced compute density scaling for variable-precision inputs. Specifically, halving the bit-width of both input operands yields a 4-fold throughput increase, whereas halving the bit-width of only one operand results in a 2-fold throughput increase. 

**Table 2: The notation used in the subsequent sections.** 

Extended analysis reveals this as a fundamental limitation in typical spatial architectures: WS systolic arrays in Fig. 2(a) execute fixed linear combinations like _𝑃_ 1 _,𝑇_ + ( _𝐴_ 1 _,𝑇_ −1 _𝐵_ 1 + _𝐴_ 2 _,𝑇 𝐵_ 2) due to operand stationarity. Multiplier-adder tree in Fig. 2(b) and bit-serial architectures in Fig. 2(c) inherit redundant PPs generation from broadcast/stationary operand mechanisms. The root cause lies in isolated computation inherent to traditional spatial architectures; hardware designs constrained by scalar dataflow paradigms (broadcast, systolic, stationary) fail to implement cross-PE collaboration mechanisms. Under GEMM’s high data-reuse scenarios, this leads to tensor-level redundant PPs reduction with corresponding hardware area/energy efficiency losses. 

From the perspective of multi-precision computing, existing architectures face dual challenges. First, dynamic reconfiguration inefficiency persists in low-precision multiplier implementations. To support multi-precision (INT4/8/16/32) GEMM, current designs employ low-bit-width multiplier combinations like IBM’s NPU in Fig. 3(a), but incur significant hardware overhead. As Table 1 test component on UMC 22 nm under 1GHz constraint shows, INT8 multiplication using four INT4 multipliers with a reduction tree achieves only 57% computational density versus dedicated INT8 multipliers, while INT16 implementations suffer >50% area efficiency loss. Samsung’s hybrid design in Fig. 3(b) partially addresses INT8 efficiency through mixed-width multipliers (one INT8 × INT8, two INT8 × INT4, one INT4 × INT4), yet requires full activation of 4 multipliers for INT4 operations and three multipliers with reduction trees for INT8 operations, resulting in suboptimal hardware utilization and energy efficiency. Second, imbalanced compute density scaling undermines mixed-precision throughput matching. Theoretically, INT4×INT4 should deliver 4× density over INT8×INT8 under equivalent area, yet real-world implementations show sublinear scaling: NVIDIA A100 [2] achieves only 2× (INT4: 1248 TOPS 

||**NOTATION**<br>_𝐴𝑀_×_𝐾_<br>_𝐴𝑚,𝑘_<br>_𝐿𝐴_<br>(_𝑎𝑖_)_𝑚,𝑘_<br>_𝐴𝑚,𝑘_⟨_𝑖_⟩<br>_𝐴𝑚,𝑘_[_ℎ_:_𝑙_]<br>M(·)|**DESCRIPTION**<br>Matrix A of dimension_𝑀_×_𝐾_.<br>The _𝑚_-th row and _𝑘_-th column element of<br>_𝐴𝑀_×_𝐾_.<br>Bit-width of each element in matrix_𝐴_.<br>The_𝑖_-th bit of_𝐴𝑚,𝑘_, where_𝑎𝑖_∈{0_,_1},_𝑎_−1 =0.<br>The_𝑖_-th 3-bitgroup: (_𝑎_2_𝑖_+1_,𝑎_2_𝑖,𝑎_2_𝑖_−1)_𝑚,𝑘_.<br>Bit slice from_𝑙_-th bit to_ℎ_-bit of_𝐴𝑚,𝑘_.<br>MBE calculation rule for 3-bit input.|
|---|---|---|



## **2 Motivation** 

## **2.1 Computational Redundancy in GEMM for Spatial Architectures** 

Given a GEMM: _𝐶𝑀_ × _𝑁_ = _𝐴𝑀_ × _𝐾_ · _𝐵𝐾_ × _𝑁_ , when the _𝐴𝑀_ × _𝐾_ adopts the MBE multiplication principle, the computation can be decomposed as: 

**==> picture [212 x 28] intentionally omitted <==**

where _𝐼_ indicates the parallel reduction dimension within the multiplier. The MBE encoding function M(·) strictly satisfies M(·) ∈ {−2 _,_ −1 _,_ 0 _,_ 1 _,_ 2} in its codomain. Subsequently, we merge the even and odd terms along the reduction dimension _𝐾_ and restructure Eq. 2 through summation order exchange: 

**==> picture [248 x 39] intentionally omitted <==**

3 

1627 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea Q. Wu, J. Zhou, Z. Hu, Z. Zeng, H. Liang, J. Zhu, L. Tao, X. Zhang, Z. Cheng, L. Zhao, Y. Wei, X. Wang, and X. Jin 

|**_(a)_ INT4 Multiplier**|**_(b)_ INT8 Multiplier**|
|---|---|
|**Low area**<br>**efficiency**<br>**Encoder**<br>**Encoder**<br>**Full Adder**<br>**CPPG**<br>**CPPG**|**Encoder**<br>**4-2 Compressor Tree**<br>**Encoder**<br>**Encoder**<br>**Encoder**<br>**Full Adder**<br>**sum**<br>**carry**<br>**High area**<br>**efficiency**<br>**CPPG**<br>**CPPG**<br>**CPPG**<br>**CPPG**|



**Figure 4: (a) INT4 multiplier. (b) INT8 multiplier.** 

In Eq. 3, we can analyze the conditions under which redundant partial products occur: since _𝐵𝑘,𝑛_ is independent of dimension _𝑀_ and solely determined by _𝐾_ , when calculating the _𝑛_ -th column of the output matrix _𝐶𝑚,𝑛_ , it is possible for different rows or bitslice groups to compute identical or linearly related expressions. Specifically, as illustrated in Eq. 1, even though the multiplicand slices vary across cycles or PEs, the encoded partial products may still reduce to the same value due to the constrained output range of the MBE function M(·) and the symmetric structure of the coefficient set {−2 _,_ −1 _,_ 0 _,_ 1 _,_ 2}. This observation substantiates Eq. 4, which formally states that for certain pairs ( _𝑚,𝑖_ ) and ( _𝑝, 𝑗_ ), their vector partial products may satisfy a linear relationship: 

**==> picture [244 x 42] intentionally omitted <==**

This equivalence is likely to occur under two scenarios: within the same row across different bit-slice indices (( _𝑚_ = _𝑝_ ) ∧( _𝑖_ ≠ _𝑗_ )), or across different rows for all bit-slice indices (( _𝑚_ ≠ _𝑝_ ) ∧(∀ _𝑗_ )). 

This induces redundant computation of partial products in Eq. 3, establishing the theoretical foundation for subsequent hardware optimization of partial product compression. Therefore, for matrix elements residing in consecutive pairs of columns across all rows, redundant partial product accumulation will inevitably occur under different bit-weight slices if their encoded values satisfy the linear relationship in Eq. 4. 

## **2.2 Computational Density Imbalance in Mixed-Precision TPE** 

The construction of multi-precision TPEs using low-precision multipliers inherently creates computational density imbalance. This phenomenon arises from significant microarchitectural differences between low bit-width and high bit-width multipliers. 

In the INT4 multiplier shown in Fig. 4(a), the multiplicand is processed through MBE, and the candidate partial product generator (CPPG) generates 2 PPs, which are summed via a single full adder to produce the final multiplication result. In higher-precision implementations like the INT8 multiplier depicted in Fig. 4(b), CPPG produces 4 PPs that undergo reduction through a 4-2 compressor tree [38, 46] or Wallace tree [48] (merging PPs into final sum and carry terms) before getting the multiply result through final summation via a full adder [5, 9, 15]. The compressor tree utilizes 3-input half-adders for multi-operand summation, as half-adders demonstrate lower resource consumption and superior timing characteristics compared to full adders, and enable more efficient area utilization when reducing more than two operands. 

|**_-2(B2k+B2k+1)_**<br>**_-(B2k+B2k+1)_**<br>**_B2k+B2k+1_**<br>**_2(B2k+B2k+1)_**|**_-2(B2k+B2k+1)_**<br>**_-(B2k+B2k+1)_**<br>**_B2k+B2k+1_**<br>**_2(B2k+B2k+1)_**||**_B2k+B2k+1_**<br>**_B2k-B2k+1_**<br>**_2B2k+B2k+1_**<br>**_B2k-2B2k+1_**<br>**_B2k+2B2k+1_**<br>**_2B2k-B2k+1_**<br>**_B2k_**<br>**_B2k+1_**|
|---|---|---|---|
|**_-2(B2k-B2k+1)_**<br>**_-(B2k-B2k+1)_**<br>**_B2k-B2k+1_**<br>**_2(B2k-B2k+1)_**||||
||**_-(2B2k+B2k+1)_**<br>**_2B2k+B2k+1_**|||
||**_-(B2k-2B2k+1)_**<br>**_B2k-2B2k+1_**|||
||**_-(B2k+2B2k+1)_**<br>**_B2k+2B2k+1_**|||
||**_-(2B2k-B2k+1)_**<br>**_2B2k-B2k+1_**|||
|**_-2B2k_**<br>**_-B2k_**<br>**_B2k_**<br>**_2B2k_**||||
|**_-2B2k+1_**<br>**_-B2k+1_**<br>**_B2k+1_**<br>**_2B2k+1_**||||



**Figure 5: Vector Partial Product Lookup Table (VPP LUT). Here, both** _𝐵_ 2 _𝑘_ **and** _𝐵_ 2 _𝑘_ +1 **omit the same second dimension** _𝑛_ **.** 

**==> picture [239 x 93] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)  Multioperand High-Radix Encoder (b)  VPP Generator<br>Multiplicand(Am,2k) Multiplicand(Am,2k+1) VPP LUT<br>x x x x x x x x x x x x x x x x B2k+B2k+1 S[4:2] S[1:0]<br>a2i+1 a2i a2i+1 a2i 2BB2k2k-B+B2k+12k+1 -2x<br>MBE Encoder Bit Cache MBE Encoder BB2k2k+-2B2B2k+12k+1 x 2x-xx<br>Neg1 One1 Two1 Neg2 One2 Two2 2B2k-B2k+1<br>VPP Select Encoder B2k CE<br>B2k+1<br>CE,S[4:0] MH Decoder<br>Mux Map Mux<br>**----- End of picture text -----**<br>


**Figure 6: (a) Multi-operand High-radix Encoder (MHE). (b) Vector Partial Product (VPP) generator, composed of VPP LUT and MHD. Here, both** _𝐵_ 2 _𝑘_ **and** _𝐵_ 2 _𝑘_ +1 **omit the same second dimension** _𝑛_ **.** 

However, employing INT4 multipliers as foundational elements in multi-precision TPEs inevitably leads to computational density mismatch at higher precisions. The fundamental limitation stems from insufficient reduction dimensionality in INT4 multiplier implementations, which prevents the deployment of high-efficiency reduction components, thereby degrading both computational density and energy efficiency in high-precision computations. 

## **3 Proposed Architecture 3.1 Multi-Operand High-Radix Encoder** 

The essence of multiplication operations lies in weighted reduction of multipliers with distinct bit weights. Through numerical domain transformation of solution sets, MBE encoding reduces the number of PPs requiring a reduction in multiplication to half the bit length (as shown in Eq. 2), thereby halving the overhead of internal reduction components in multipliers. In spatial computing architectures, the hardware complexity of vector reduction units directly impacts system energy efficiency [6, 28]. This characteristic motivates our fundamental research proposition: **in vector reduction, can we achieve a further half compression of reduction elements in vector inner product by establishing a multi-dimensional joint operand encoding strategy ?** 

As derived from Eq. 3, the core mechanism for halving the vector inner product reduction number _𝐾_ in GEMM resides in directly obtaining the calculation results of Vector Partial Products (VPP). The mathematical definition of VPP can be expressed as: 

**==> picture [231 x 12] intentionally omitted <==**

It should be noted that VPP also serves as the source of redundant computation in GEMM. In Eq. 5, due to the limited codomain 

4 

1628 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

MHE-TPE: Multi-Operand High-Radix Encoder for Mixed-Precision Fixed-Point Tensor Processing Engines 

of the 3-bit window MBE encoding function M(·), its linear combination with _𝐵_ 2 _𝑘,𝑛_ and _𝐵_ 2 _𝑘_ +1 _,𝑛_ exhibits finite-state characteristics. Through algebraic simplification and common term extraction of 24 non-zero states (as illustrated in Fig. 5), we ultimately simplify them to 8 reduced linear combinations: { _𝐵_ 2 _𝑘,𝑛_ + _𝐵_ 2 _𝑘_ +1 _,𝑛, 𝐵_ 2 _𝑘,𝑛_ − _𝐵_ 2 _𝑘_ +1 _,𝑛,_ 2 _𝐵_ 2 _𝑘,𝑛_ + _𝐵_ 2 _𝑘_ +1 _,𝑛, 𝐵_ 2 _𝑘,𝑛_ −2 _𝐵_ 2 _𝑘_ +1 _,𝑛, 𝐵_ 2 _𝑘,𝑛_ +2 _𝐵_ 2 _𝑘_ +1 _,𝑛,_ 2 _𝐵_ 2 _𝑘,𝑛_ − _𝐵_ 2 _𝑘_ +1 _,𝑛, 𝐵_ 2 _𝑘,𝑛, 𝐵_ 2 _𝑘_ +1 _,𝑛_ } _._ This set of linear combinations can be naturally organized into a Vector Partial Product Lookup Table (VPP LUT) for ease of reference and reuse in subsequent computation, and we have arrived at the following matrix multiplication algorithm from Eq. 3: 

**==> picture [240 x 39] intentionally omitted <==**

Here, _𝛾_ ( _𝑚,𝑘,𝑖_ ) ∈{−2 _,_ −1 _,_ 0 _,_ 1 _,_ 2} denotes a scalar coefficient that selects and scales the corresponding linear combination from VPP LUT. Therefore, from a computational perspective in Eq. 6, for a matrix ∀ _𝑚_ ∈ _𝑀_ and all ∀ _𝑖_ ∈ _𝐼_ , these 8 linear combinations in VPP LUT can be pre-computed and reused to directly generate VPP _[𝑛] 𝑚,𝑘_[⟨] _[𝑖]_[⟩][, thereby halving the reduction dimension from] _[ 𝐾]_[to] _[ 𝐾]_[/][2][.] 

At the hardware level, the VPP LUT is implemented as a dedicated unit. First it reads _𝐵_ 2 _𝑘,𝑛_ and _𝐵_ 2 _𝑘_ +1 _,𝑛_ from memory as primitive inputs, and then use a single adder to serially generate the rest 6 derived terms, and employs data flip-flops (DFFs) for state latching as VPP LUT. The VPP LUT exhibits multi-dimensional sharing characteristics: Reusable across different bit-weights in _𝐴𝑚,𝑘_ ⟨ _𝑖_ ⟩ and shareable among all row indices _𝑚_ of the _𝐶𝑚,𝑛_ . 

As shown in Fig. 6(b), the VPP LUT and Multi-operand Highradix Decoder (MHD) collectively constitute the VPP generator for dual-operand inner products. Key signals and modules include: 5-bit selection signal _𝑆_ [4 : 0], zero-value generation chip selection signal _𝐶𝐸_ , and a mapping unit (Map) that expands 8 basic states into 24 valid states through simple signal bit extension and complementary number computation. 

Fig. 6(a) illustrates the structural design of the Multi-operand High-radix Encoder (MHE), whose workflow includes: ❶ synchronous reading of 4-bit slice from dual operands with identical bit weights; ❷ pre-encoding using dual MBE with 3-bit windows (including 1-bit in previous bit-cache); ❸ generation of combinational logic signals (Neg, One, Two); ❹ final selection signal _𝐶𝐸,𝑆_ [4 : 0] are generating through VPP Select Encoder. In the tensor computing phase, only selection signals are required for MHD in Fig. 6(b) to generate VPP and used for reduction in GEMM, without the need for a multiplier. Throughout this process, both the MHE and VPP LUT components can be shared in computations along the matrix’s _𝑀_ -dimension, thereby minimizing hardware overhead. 

The MHE, MHD, and VPP LUT collectively form the foundational module of the MHE-TPE. This design demonstrates notable hardware efficiency advantages: ❶ enhanced reuse rate of partial computation results; ❷ encoder-computation decoupling enables control signal broadcasting or pipelining across multiple computing modules, effectively amortizing MHE’s logic area overhead to negligible levels; ❸ subsequent reduction in hardware overhead through halving the partial product generation. 

**Table 3: The notation used in 3.2 and 3.3.** 

||**Table 3: The notation used in 3.2 and 3.3.**|
|---|---|
||**NOTATION**<br>**DESCRIPTION**<br>LUT_𝑛_<br>_𝑘_⟨_𝑏𝑠_⟩<br>Vector Partial Product Lookup Table, storing 8<br>reduced linear combinations of the_𝑏𝑠_-th 4-bit|
||slices of_𝐵_2_𝑘,𝑛_and_𝐵_2_𝑘_+1_,𝑛_.<br>(_𝐶𝐸,𝑆_)_𝑚,𝑘_⟨_𝑖_⟩<br>Selection signals derived from _𝐴𝑚,_2_𝑘_⟨_𝑖_⟩and<br>_𝐴𝑚,_2_𝑘_+1⟨_𝑖_⟩.<br>VPP_𝑛_<br>_𝑚,𝑘_⟨_𝑖_|_𝑏𝑠_⟩<br>Vector Partial Product at the _𝑏𝑠_-th 4-bit slice,<br>computed as a LUT-based linear combination of<br>_𝐵_2_𝑘,𝑛_and_𝐵_2_𝑘_+1_,𝑛_, weighted by the MHE results<br>of_𝐴𝑚,_2_𝑘_⟨_𝑖_⟩and_𝐴𝑚,_2_𝑘_+1⟨_𝑖_⟩.|
||PS_𝑚,𝑛_⟨_𝑏𝑠_⟩<br>Partial sum at the_𝑏𝑠_-th 4-bit slice of_𝐶𝑚,𝑛_.|
||† In Algorithm 1, _𝐵_∈Z_𝐾_×1 is a column vector and the bit-slice index_𝑏𝑠_is not<br>considered; thus, superscript_𝑛_and parameter⟨_𝑏𝑠_⟩are omitted in related notations.|



||**Algorithm 1:**Matrix-Vector Multiplication in MHE-TPE|
|---|---|
||**Input**<br>**:**Matrix_𝐴_∈Z_𝑀_×_𝐾_, Vector_𝐵_∈Z_𝐾_×1|
||**Output:**Vector_𝐶_∈Z_𝑀_×1 where_𝐶𝑀_×1 =_𝐴𝑀_×_𝐾_·_𝐵𝐾_×1|
||**Preprocessing Phase:**|
||**forall**_𝑘_∈{0_, . . . ,_⌈_𝐾_/2⌉−1}**_in parallel_ do**|
||LUT_𝑘_←ProcessPair(_𝐵_2_𝑘, 𝐵_2_𝑘_+1);|
||**Computation Phase:**<br>**for**_𝑖_=0**_to_** ⌈_𝐿𝐴_/2⌉−1**do**<br>**forall**_𝑚_∈{0_, . . . , 𝑀_−1}**_in parallel_ do**<br>**forall**_𝑘_∈{0_, . . . ,_⌈_𝐾_/2⌉−1}**_in parallel_ do**<br>(_𝐶𝐸,𝑆_)_𝑚,𝑘_⟨_𝑖_⟩←MHE(_𝐴𝑚,_2_𝑘_⟨_𝑖_⟩_,𝐴𝑚,_2_𝑘_+1⟨_𝑖_⟩);<br>VPP_𝑚,𝑘_⟨_𝑖_⟩←MHD �(_𝐶𝐸,𝑆_)_𝑚,𝑘_⟨_𝑖_⟩_,_LUT_𝑘_<br>�;<br>_𝐶𝑚_+=<br>��⌈_𝐾_/2⌉−1<br>_𝑘_=0<br>VPP_𝑚,𝑘_⟨_𝑖_⟩<br>�<br>≪2_𝑖_;|
|||



## **3.2 MHE-TPE Architecture** 

The fundamental restriction in conventional TPEs for mixed-precision computation stems from insufficient reduction dimensions within low-bit-width multipliers [8, 27, 50]. This physical decoupling of multiplication and reduction logic creates dual-dimensional mismatches: spatially in compressor tree compatibility with multiprecision weighted PPs, and temporally in accumulator bit-width adaptation to dynamic precision variations. 

For this reason, we propose a three-phase tensor computing paradigm that improves these issues through: (1) temporal multiplicand bit-slice encoding; (2) partial product generation; (3) unified partial product reduction. 

This three-stage paradigm eliminates conventional multiplication concepts by decomposing multiplicative operations into partial products, fusing bit-weight reduction within multiplication and vector inner product reduction across temporal or spatial dimensions. The unified reduction components enable consistent computation bit-width and enhanced hardware reusability for multi-precision operations. 

_3.2.1 Architecture Implementation._ The MHE-TPE microarchitecture shown in Fig. 7, for general matrix-vector multiplication (GEMV) features collaborative heterogeneous computing units for matrix _𝐴_ ∈ Z _[𝑀]_[×] _[𝐾]_ and vector _𝐵_ ∈ Z _[𝐾]_[×][1] (note that the dimensions of the 

5 

1629 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Q. Wu, J. Zhou, Z. Hu, Z. Zeng, H. Liang, J. Zhu, L. Tao, X. Zhang, Z. Cheng, L. Zhao, Y. Wei, X. Wang, and X. Jin 

**==> picture [500 x 298] intentionally omitted <==**

**----- Start of picture text -----**<br>
K<br>Scratchpad Memory (Matrix A)<br>MH Encoder MH Encoder MH Encoder M<br>MH Encoder MH Encoder MH Encoder<br>MH Encoder MH Encoder MH Encoder<br>TPE Tile<br>Fill VPP Pipeline Fill VPP Pipeline Compressor Tree 0 Fill VPP Pipeline Full Adder<br>MH Decoder MH Decoder MH Decoder<br>Compressor Tree 3 Full Adder<br>VPP<br>VPP  VPP<br>LUT Fill VPP Pipeline LUT Fill VPP Pipeline Compressor Tree 4 LUT Fill VPP Pipeline Full Adder<br>MH Decoder MH Decoder MH Decoder<br>K/2-<br>0 2<br>Compressor Tree 7 1 Full Adder<br>Fill VPP Pipeline Fill VPP Pipeline Compressor Tree M-3 Fill VPP Pipeline Full Adder<br>MH Decoder MH Decoder MH Decoder<br>Compressor Tree M-1 Full Adder<br>Fill VPP LUT Fill VPP LUT Fill VPP LUT<br>VPP Controller VPP Controller VPP Controller<br>K<br>Scratchpad Memory (Matrix B)<br>**----- End of picture text -----**<br>


**Figure 7: MHE-TPE microarchitecture for GEMV.** 

matrix here refer to the size of the sub-matrix that the hardware processes at a time). 

Four core modules constitute in this architecture: 

- (1) _𝐾_ /2 VPP controllers and VPP LUT (depth 8, bit-width _𝐿𝐵_ +2, DFF storage) 

- (2) _𝐾_ /2 MHD arrays, each comprising _𝑀_ /4 MHD units and sharing one VPP LUT, used for generating VPP to the input register array. 

- (3) _𝐾_ /2 MHE arrays, each comprising _𝑀_ /4 MHE units for selection signal generation to corresponding MHD. 

- (4) _𝑀_ compressor trees (each with _𝐾_ /2 input ports) perform reduction operations on the PPs stored in the input registers. 

_3.2.2 Computational Workflow (Algorithm 1 and Fig. 7)._ The process consists of two phases: preprocessing and computation. In the preprocessing phase, vector _𝐵_ is partitioned into _𝐾_ /2 consecutive 2-element subvectors _𝐵_ 2 _𝑘, 𝐵_ 2 _𝑘_ +1, loaded in parallel to VPP controllers via scratchpad memory, and use a full adder to generate: { _𝐵_ 2 _𝑘_ + _𝐵_ 2 _𝑘_ +1 _, 𝐵_ 2 _𝑘_ − _𝐵_ 2 _𝑘_ +1 _,_ 2 _𝐵_ 2 _𝑘_ + _𝐵_ 2 _𝑘_ +1 _, 𝐵_ 2 _𝑘_ − 2 _𝐵_ 2 _𝑘_ +1 _, 𝐵_ 2 _𝑘_ + 2 _𝐵_ 2 _𝑘_ +1 _,_ 2 _𝐵_ 2 _𝑘_ − _𝐵_ 2 _𝑘_ +1} in 6 clock cycles. Each controller precomputed results are stored in VPP-LUTs, establishing a TPU-like WS dataflow pattern. _**Notably, vector B is processed in its full precision, without involving bit-slice (bs) partitioning.**_ 

In the computation phase, matrix _𝐴_ is processed in 2-column groups through temporal iteration of 4-bit dual operands _𝐴𝑚,_ 2 _𝑘_ and _𝐴𝑚,_ 2 _𝑘_ +1 (2-bit each). MHE units encode inputs into selection signals 

driving MHD to generate _𝑀_ · _𝐾_ /2 VPPs. Subsequent _𝑀_ compressor trees (each with _𝐾_ /2 input ports) perform reduction operations. For matrix _𝐴_ with INT _𝐿𝐴_ precision, a single GEMV operation requires ⌈ _𝐿𝐴_ /2⌉ clock cycles. 

_3.2.3 Component Reusable Optimization._ As shown in Fig. 7, given the high degree of logical parallelism demonstrated by the MHE and MHD within the three-stage encoding, decoding, and reduction pipeline, which contributes to their lower logic latency, the multioperand summation logic in the compressor tree readily emerges as a timing bottleneck. Even with the incorporation of multi-stage pipelining in the reduction tree, it doesn’t remain easy to achieve logic latency comparable to that of the MHE and MHD. 

To mitigate this limitation, we implement a dual-clock domain design for the TPE pipeline. The fast clock domain (4× base frequency) drives the MHE and MHD units, while the slow clock domain (base frequency) drives the compressor trees. This dual-clock domain design enables hardware optimization through temporal multiplexing: employ one MHD operating at a fast clock through 4 cycles to serially fill the input registers of four compressor trees with VPP, and time-interleaved into 4 compressor trees, the input DFFs of the slow clock domain are fed through a single-bit valid cross-clock domain handshake and perform synchronization (yellow to red arrows in Fig.7). When necessary, backpressure is applied to the MHE pipeline. This reduces MHD/MHE units shared by one VPP LUT from _𝑀_ to _𝑀_ /4 while maintaining computational throughput. 

6 

1630 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

MHE-TPE: Multi-Operand High-Radix Encoder for Mixed-Precision Fixed-Point Tensor Processing Engines 

**==> picture [242 x 269] intentionally omitted <==**

**----- Start of picture text -----**<br>
MH Encoder Scratchpad Memory (Matrix A)<br>(CE,S)m,k<br>Tile 0 Tile 1 Tile 2 Tile 3<br>M × Local Reduce Module (upper)<br>Tile 4 Tile 5 Tile 6 Tile 7<br>M × Local Reduce Module (lower)<br>Tile NT-4 Tile NT-3 Tile NT-2 Tile NT-1<br>M × Local Reduce Module (lower)<br>Figure 8: MHE-TPE array, which comprises  𝑁𝑇 TPE Tiles.<br>LRM(Upper)  MHE-TPE Tile Output LRM(Lower)  MHE-TPE Tile Output<br>Matrix B INT 4 Path<br>Pipeline  Pipeline  Pipeline<br>Matrix B INT 8 Path Pipeline  Pipeline  Pipeline  buffer buffer buffer<br>Matrix B INT 16 PathMatrix B INT 32 Path buffer Mux buffer Mux buffer Mux Mux Mux Mux<br>>> 10bit Adder 10bit Adder<br>10bit Adder 10bit Adder<br>Matrix D x x x >> 11bit Adder >> Pipeline Buffer11bit Adder<br>Mux Mux<br>VPP Controller<br>Scratchpad Memory (Matrix C) Scratchpad Memory (Matrix B)<br>**----- End of picture text -----**<br>


**Figure 9: Two types of Local Reduce Module (LRM), when** _𝐾_ = 32 **.** 

## **3.3 Mixed-Precision MHE-TPE Array** 

_3.3.1 Architecture Implementation._ Fig. 8 extends the proposed mixed-precision TPE Tile (depicted as the gray region in Fig. 7) and incorporates spatiotemporal co-mapping to support arbitraryprecision matrix multiplication while ensuring balanced computational density. 

Given arbitrary-precision sub-matrices _𝐴_ ∈ Z _[𝑀]_[×] _[𝐾]_ and _𝐵_ ∈ Z _[𝐾]_[×] _[𝑁]_ where _𝑁_ = ⌈ _[𝑁𝑇] 𝐿𝐵_[×][4] ⌉, the control flow adheres to Algorithm 2. Each computational tile derived from the architecture in Fig. 7 by excluding MHE, VPP controller, and scratchpad memory, and the parameters of the TPE Tile are as follows: (1) _𝐾_ /2 VPP LUT with 8-depth and 6-bit width (4-bit _𝐿𝐵_ + 2-bit expansion). (2) _𝑀_ compressor trees with _𝐾_ /2 input ports, each port with 6-bit width, and output bit width is 6 + ⌈ _𝑙𝑜𝑔_ 2 ( _𝐾_ /2)⌉. 

From the perspective of the TPE array, the hardware components include: (1) Total _𝑁𝑇_ TPE Tiles; (2) _𝐾_ · _𝑀_ /8 MHE units generate selection signals ( _𝐶𝐸,𝑆_ ) computed from Matrix _𝐴_ , synchronized through systolic transmission between row-column tiles; (3) the Local Reduce Module (shown in Fig. 9) is specifically designed for inter-tile reduction operations across multiple computational tiles; (4) the VPP controller pre-computes the column vectors in Matrix _𝐵_ and sequentially fills VPP into the VPP LUT of each TPE Tile. 

_3.3.2 Spatiotemporal Mapping Methodology._ In preprocessing phase, arbitrary-precision Matrix _𝐵_ ∈ Z _[𝐾]_[×] _[𝑁]_ undergoes spatial mapping, 4-bit slices are stored in scratchpad memory with dynamic tile allocation, and continuous 4-bit slices are assigned to sequential TPE Tiles. As shown in Fig. 10, different precision cases correspond to 

|**Algorithm 2:**General Matrix Multiplication with Mixed-|**Algorithm 2:**General Matrix Multiplication with Mixed-|||
|---|---|---|---|
|Precision in Multi-Tile MHE-TPE||||
|**Input**<br>**:**Matrix_𝐴_∈Z_𝑀_×_𝐾_, Matrix_𝐵_∈Z_𝐾_×_𝑁_||||
|**Output:**Matrix_𝐶_∈Z_𝑀_×_𝑁_where_𝐶𝑀_×_𝑁_=_𝐴𝑀_×_𝐾_·_𝐵𝐾_×_𝑁_||||
|**Preprocessing Phase:**||||
|**forall**_𝑛_∈{0_, . . . , 𝑁_−1}**_in parallel_ do**||||
|**forall**_𝑘_∈{0_, . . . ,_⌈_𝐾_/2⌉−1}**_in parallel_ do**||||
|**forall**_𝑏𝑠_∈{0_, . . . ,_⌈_𝐿𝐵_/4⌉−1}**_in parallel_ **|**do**|||
|LUT_𝑛_<br>_𝑘_⟨_𝑏𝑠_⟩←||||
|ProcessPair(_𝐵_2_𝑘,𝑛_[4_𝑏𝑠_+3:4_𝑏𝑠_]_, 𝐵_2_𝑘_+1_,𝑛_[4_𝑏𝑠_+3:4_𝑏𝑠_]);||||
|||||
|||||
|**Computation Phase:**||||
|**for**_𝑖_=0**_to_** ⌈_𝐿𝐴_/2⌉−1**do**||||
|**forall**_𝑚_∈{0_, . . . , 𝑀_−1}**_in parallel_ do**||||
|**forall**_𝑘_∈{0_, . . . ,_⌈_𝐾_/2⌉−1}**_in parallel_ do**||||
|(_𝐶𝐸,𝑆_)_𝑚,𝑘_⟨_𝑖_⟩←MHE(_𝐴𝑚,_2_𝑘_⟨_𝑖_⟩_,𝐴𝑚,_2_𝑘_+1⟨_𝑖_⟩);||||
|||||
|**forall**_𝑛_∈{0_, . . . , 𝑁_−1}**_in parallel_ do**||||
|**forall**_𝑚_∈{0_, . . . , 𝑀_−1}**_in parallel_ do**||||
|**forall**_𝑏𝑠_∈{0_, . . . ,_⌈_𝐿𝐵_/4⌉−1}**_in parallel_ do**||||
|**forall**_𝑘_∈{0_, . . . ,_⌈_𝐾_/2⌉−1}**_in parallel_ do**||||
|VPP_𝑛_<br>_𝑚,𝑘_⟨_𝑖_|_𝑏𝑠_⟩←MHD<br>�<br>(_𝐶𝐸,𝑆_)_𝑚,𝑘_⟨_𝑖_⟩_,_|LUT_𝑛_<br>_𝑘_⟨_𝑏𝑠_⟩|�|;|
|PS_𝑚,𝑛_⟨_𝑏𝑠_⟩+=<br>��⌈_𝐾_/2⌉−1<br>_𝑘_=0<br>VPP_𝑛_<br>_𝑚,𝑘_⟨_𝑖_|_𝑏𝑠_⟩<br>�|≪2_𝑖_;|||
|||||
|||||
|||||
|_𝐶𝑚,𝑛_+= �⌈_𝐿𝐵_/4⌉−1<br>_𝑏𝑠_=0<br>�PS_𝑚,𝑛_⟨_𝑏𝑠_⟩≪4_𝑏𝑠_�;||||



different tile utilization strategies. This enables scalable data partitioning with fine-grained control over tile reuse and parallelism across bit-widths. 

For INT4, each tile directly maps one output column of _𝐵_ ∈ Z _[𝐾]_[×] _[𝑁𝑇]_ . For INT8, each column of _𝐵_ is distributed across 2 tiles to store the high and low 4-bit fields, and the effective mapping becomes _𝐵_ ∈ Z _[𝐾]_[×] _[𝑁𝑇]_[/][2] . For INT16 and INT32, each column of _𝐵_ spans 4 and 8 TPE Tiles, respectively, resulting in mappings of _𝐵_ ∈ Z _[𝐾]_[×] _[𝑁𝑇]_[/][4] and _𝐵_ ∈ Z _[𝐾]_[×] _[𝑁𝑇]_[/][8] . 

In computation phase, arbitrary-precision Matrix _𝐴_ ∈ Z _[𝑀]_[×] _[𝐾]_ employs temporal mapping: ❶ Outer-loop temporal iteration of dual operands _𝐴𝑚,_ 2 _𝑘_ ⟨ _𝑖_ ⟩ and _𝐴𝑚,_ 2 _𝑘_ +1⟨ _𝑖_ ⟩ in Algorithm 2. ❷ Systolic transmission of ( _𝐶𝐸,𝑆_ ) _𝑚,𝑘_ ⟨ _𝑖_ ⟩ selection signals requiring ⌈ _𝐿𝐴_ /2⌉ cycles for INT _𝐿𝐴_ precision (shown in Fig. 10). ❸ Partial sum PS _𝑚,𝑛_ ⟨ _𝑏𝑠_ ⟩+ = �� _𝑘_ ⌈ _𝐾_ =0/2⌉−1 VPP _𝑚,𝑘[𝑛]_[⟨] _[𝑖]_[|] _[𝑏𝑠]_[⟩] � ≪ 2 _𝑖_ and partial results are accumulation in each tile and corresponding LRM. 

The computational phase employs a hierarchical reduction mechanism: Within each MHE-TPE tile, a 4-bit sliced Matrix _𝐵_ executes GEMV with an arbitrary-precision Matrix _𝐴_ . Cross-tile reduction is achieved through the LRM, which performs bit-shifted accumulation of partial results across Matrix _𝐵_ slices. This distributed module establishes communication links between every 4 adjacent TPE Tiles, enabling coordinated reduction across 8 TPE Tiles for INT32 precision through LRM upper and LRM lower interconnections. 

In the light of the 4-bit positional difference between Matrix _𝐵_ slices mapped across tiles, the LRM utilizes pipeline buffers to align partial product weights during the outermost loop iteration ( _𝑖_ = 0 to ⌈ _𝐿𝐴_ /2⌉− 1). As illustrated by color-coded arrows in Fig. 9, the 

7 

1631 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Q. Wu, J. Zhou, Z. Hu, Z. Zeng, H. Liang, J. Zhu, L. Tao, X. Zhang, Z. Cheng, L. Zhao, Y. Wei, X. Wang, and X. Jin 

**==> picture [253 x 215] intentionally omitted <==**

**----- Start of picture text -----**<br>
MH Encoder Scratchpad Memory (Matrix A) NT<br>(CE,S)m,k<br>Tile 0 Tile 1 Tile 2 Tile 3<br>(CE,S)m,k M × Local Reduce Module (upper) INT4 Matrix BCase 1NT/2<br>Tile 4 Tile 5 Tile 6 Tile 7<br>(CE,S)m,k M × Local Reduce Module (Lower)<br>INT8 Matrix B<br>Case 2<br>Tile NT-8 Tile NT-7 Tile NT-6 Tile NT-5 NT/4<br>M × Local Reduce Module (upper)<br>(CE,S)m,k<br>Tile NT-4 Tile NT-3 Tile NT-2 Tile NT-1 INT16 Matrix B<br>Case 3<br>M × Local Reduce Module (Lower) NT/8<br>K K K<br>INT LA  M M M<br>Matrix A<br>INT32 Matrix B<br>Cycle 1 Cycle 2 Cycle LA/2 Case 4<br>Temporal mapping for multi-precision Matrix  𝐴<br>K<br>VPP Controller K<br>K<br>Scratchpad Memory (Matrix C) Scratchpad Memory (Matrix B)<br>Spatial mapping for multi-precision Matrix<br>𝐵<br>K<br>**----- End of picture text -----**<br>


**Figure 10: Temporal mapping for multi-precision Matrix** _𝐴_ **and spatial mapping for multi-precision Matrix** _𝐵_ **.** 

accumulation phase implements bit-shifting and summation according to _𝐴_ ’s bit-slice weights after MHE; ultimately, the operator can be implemented as Matrix _𝐶_ = _𝐴_ · _𝐵_ + _𝐷_ . 

This novel computational paradigm: bit-slice encoding → partial product generation → unified partial product reduction, achieves mixed-precision matrix multiplication through temporal mapping of multi-precision Matrix _𝐴_ and spatial tile-level mapping of multiprecision Matrix _𝐵_ . The architecture demonstrates superior energy efficiency in low-precision regimes by merging reduction dimensions to circumvent the intrinsic limitations of low-bit-precision multipliers. Crucially, precision scaling decouples from compressor tree parameters (determined solely by VPP LUT bit-width): matrix _𝐴_ precision requires only LRM accumulator register bit-width expansion, and Matrix _𝐵_ precision scales through spatial tile allocation in the MHE-TPE array. 

The minimal inter-tile communication (simple synchronization, independent computation) ensures excellent architectural scalability. This design eliminates the consumption of reconfigurable data flow with low-bit-width multiplier in traditional architecture while maintaining computational density across precision regimes. 

## **4 Experiments** 

This section evaluates the proposed architecture through componentlevel analysis, performance optimization, and scalability assessment. Section 4.1 outlines the experimental methodology, including technology parameters and metrics for area, power, and timing. Section 4.2 analyzes individual components (MHE, MHD, multi-fanout MHD-LUTs (VPP LUT and MHD), compressor trees) and a TPE Tile, focusing on area-power trade-offs under varying fanout (the number of MHDs driven by a single VPP LUT) and LUT counts. Section 4.3 assesses MHE-TPE array performance across configurations 

and matrix sizes, optimizing for area/power efficiency and computational density, and explores the scalability of optimized MHETPE designs. Section 4.4 evaluates robustness to temperature, voltage, and process variations, benchmarking throughput/efficiency at mixed-precision GEMM. In Sections 4.5 and 4.6, we discussed the factors affecting the utilization of the TPE array under different matrices as well as under practical DNN workloads. Section 4.7 compares the design’s computational density and power efficiency with other architectures. 

## **4.1 Experimental Setup** 

We implement our design in RTL and synthesize it using Synopsys Design Compiler [43] with UMC 22nm technology at an operating voltage ranging from 0.66 to 0.81V. Next, we use Synopsys VCS to generate FSDB waveforms based on the provided stimulus signals. These waveforms, along with the optimized netlist, corresponding process corners, and physical libraries, are input into Synopsys PrimeTime PX to evaluate hardware power consumption and timing performance. For placement and routing, as well as for generating GEF and GDS layout files, we utilize Synopsys IC Compiler. Finally, we conduct layout DRC and LVS checks using Synopsys IC Validator and Mentor Calibre to ensure design correctness and manufacturability. 

**Table 4: Component performance. MHE and MHD test on 4.0GHz, and others are tested on 1.0GHz.** 

|**Unit**|**Area(**_𝜇𝑚_2**)**|**Area(**_𝜇𝑚_2**)**|**Area(**_𝜇𝑚_2**)**|**Power**<br>**(mW)**|**Delay**<br>**(ns)**|
|---|---|---|---|---|---|
||**Logic**|**DFF**|**Total**|||
|**MHE**|13.91|26.46|40.37|0.11|0.15|
|**MHD**|31.06|17.64|48.70|0.09|0.17|
|**LRM(upp)**|190.02|232.26|422.28|0.30|0.60|
|**LRM(low)**|222.36|338.10|560.36|0.44|0.67|



## **4.2 MHE-TPE Performance Analysis** 

In the subsequent experiments, we first need to analyze the timing of individual MHE and MHD components to ensure that their latency remains below four times that of other components, thereby determining the operating clock frequencies for the compressor, controller, and LRM, and then we need to determine an optimal configuration of the TPE Tile parameter and the MHE-TPE array size, which serves as the tensor computation subarray unit within a tile. Multiple optimally configured MHE-TPEs are then integrated with peripheral circuits to form a complete TPE-Array for multiprecision tensor computations. 

The Table 4 analysis of single MHE, MHD, and LRM characteristics reveals critical trade-offs in area, power, and timing efficiency, especially in terms of timing, due to the high parallelism of MHE and MHD, which enables them to operate at high clock frequencies to generate PPs. Other components, such as the compressor tree, LRM, and VPP controller, incorporate half-adders or full-adders that rely on carry-chain propagation, which typically results in higher logic delay and operation at lower frequencies. 

The table 5 summarizes the area and power consumption characteristics under different configurations. For example, under 8 MHDs and 16 input port reduction tree configuration, enabling GEMV computations on matrices _𝐴_ ∈ Z[32][×][32] and vectors _𝐵_ ∈ Z[32][×][1] , and it needs 16 VPP LUT and 32 compressor trees. 

8 

1632 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

MHE-TPE: Multi-Operand High-Radix Encoder for Mixed-Precision Fixed-Point Tensor Processing Engines 

**Table 5: Area, power breakdown, computational density, and energy efficiency (for 2-bit** × **4-bit GEMV) of MHE-TPE with varying component sizes. The MHD operates under a 4.0 GHz, whereas 1.0 GHz timing specifications bind other components.** 

|**Array**|**Array**|**Matrix Size**|**Matrix Size**|**Area(**_𝜇𝑚_2**)**|**Area(**_𝜇𝑚_2**)**|**Area(**_𝜇𝑚_2**)**|**Power(mW)**|**Power(mW)**|**Power(mW)**|**TOPS**|**TOPS/mm**2|**TOPS/W**|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**MHD**|**Tree**|**M**|**K**|**MHD-LUT**|**Tree**|**Total**|**MHD-LUT**|**Tree**|**Total**||||
||8||16|1592|409|2001|1.60|0.63|2.23|0.128|63.98|57.53|
||16||32|3182|785|3967|3.19|1.21|4.40|0.256|64.53|58.18|
|1|24|4|48|4794|1322|6116|4.87|1.99|6.86|0.384|62.79|56.00|
||32||64|6384|1995|8378|6.48|2.50|8.98|0.512|61.11|57.02|
||40||80|8003|2733|10735|8.08|3.53|11.61|0.640|59.62|55.12|
||48||96|9588|2802|12390|9.77|4.18|13.95|0.768|61.98|55.08|
||8||16|2363|820|3183|2.83|1.26|4.09|0.256|80.43|62.59|
||16||32|4777|1575|6352|5.64|2.43|8.07|0.512|80.61|63.46|
|2|24|8|48|7159|2648|9807|8.55|4.00|12.55|0.768|78.31|61.20|
||32||64|9562|3816|13378|11.32|5.06|16.38|1.024|76.54|62.51|
||40||80|11990|5466|17456|14.41|7.05|21.46|1.28|73.33|59.64|
||48||96|14403|5613|20016|17.16|8.32|25.48|1.536|76.74|60.28|
||8||16|3983|1641|5623|5.38|2.52|7.90|0.512|91.05|64.84|
||16||32|7932|3151|11082|10.60|4.92|15.52|1.024|92.40|65.99|
|4|24|16|48|11969|5303|17272|16.04|8.04|24.08|1.536|88.93|63.79|
||32||64|16041|7683|23724|21.58|10.02|31.60|2.048|86.33|64.81|
||40||80|20032|10968|31000|26.94|14.07|41.01|2.56|82.58|62.42|
||48||96|24064|11210|35273|32.37|16.71|49.08|3.072|87.09|62.59|
||8||16|7120|3275|10394|10.30|5.03|15.33|1.024|98.51|66.79|
||16||**32**|**14460**|**6325**|**20785**|**20.79**|**9.79**|**30.58**|**2.048**|**98.53**|**66.97**|
|8|24|32|48|21747|10593|32340|31.34|16.04|47.38|3.072|94.99|64.84|
||32||64|29044|15417|44461|41.60|20.07|61.67|4.096|92.13|66.41|
||40||80|36307|21927|58234|51.93|28.18|80.11|5.12|87.92|63.91|
||48||96|43542|22427|65970|62.81|33.39|96.20|6.144|93.13|63.87|
||8||16|8777|4090|12867|12.85|6.29|19.14|1.28|99.48|66.89|
||16||**32**|**17726**|**7906**|**25632**|**25.80**|**12.18**|**37.98**|**2.56**|**99.87**|**67.41**|
|10|24|40|48|26608|13235|39842|38.74|20.01|58.75|3.84|96.38|65.36|
||32||64|35523|19161|54684|51.71|25.09|76.80|5.12|93.63|64.67|
||40||80|44207|27414|71621|64.66|35.10|99.76|6.4|89.36|64.15|
||48||96|53211|28029|81241|77.44|41.60|119.04|7.68|94.53|64.52|
||8||16|10377|4909|15285|15.31|7.57|22.88|1.536|100.49|67.13|
||16||**32**|**21043**|**9488**|**30531**|**31.17**|**14.53**|**45.70**|**3.072**|**100.62**|**67.22**|
|12|24|48|48|31323|15896|47219|46.18|23.90|70.08|4.608|97.59|65.75|
||32||64|41946|22910|64856|62.01|30.07|92.08|6.144|94.73|66.72|
||40||80|52273|32898|85171|77.13|42.08|119.21|7.68|90.17|64.42|
||48||96|62753|33620|96373|92.63|49.72|142.35|9.216|95.63|64.74|



Computational density improves significantly only when the MHD fanout exceeds 4. Low fanout results in MHD-LUTs dominating the total area, primarily due to DFFs occupying nearly 50% of the MHD-LUT area. Appropriately increase the number of MHD shared resources for each individual VPP LUT will enhances both MHD area efficiency and the compressor tree’s area proportion, improving overall efficiency. Computational density and energy efficiency first improve, then decline, and then improve as the compressor tree’s reduction dimension increases. For example, with an MHD fanout is 8, computational density rises, dips, and rises again with larger reduction dimensions. This trend stems from the tree’s pipeline stages: dimensions 8 ∼ 16 require 1 stage, 24 ∼ 32 need 2 stages, and 40 ∼ 48 demand 3 stages. Adding stages increases register area and power, initially reducing efficiency. Efficiency rebounds 

at 48 due to reduced proportion of register overhead (e.g., power rises 18.44 mW from 32→40, but only 16.09 mW from 40→48, with smaller tree contributions). Similarly, transitioning 16→24 adds stages, temporarily lowering efficiency. 

The dual-operand encoding and decoding of MHE and MHD allows a reduction dimension of _𝐾_ /2 to achieve equivalent performance to a _𝐾_ -dimensional matrix, effectively halving the tree’s area and eliminating extra register/power overhead. Through the experimental data in Table 5, we have identified three configurations exhibiting peak area, and energy efficiency: ❶ MHD fanout = 8, tree inport = 16, processing matrix dimension: _𝑀_ = 32 _, 𝐾_ = 32; ❷ MHD fanout = 10, tree inport = 16, processing matrix dimension: _𝑀_ = 40 _, 𝐾_ = 32; ❸ MHD fanout = 12, tree inport = 16, processing matrix dimension: _𝑀_ = 48 _, 𝐾_ = 32. 

9 

1633 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea Q. Wu, J. Zhou, Z. Hu, Z. Zeng, H. Liang, J. Zhu, L. Tao, X. Zhang, Z. Cheng, L. Zhao, Y. Wei, X. Wang, and X. Jin 

**Table 6: Area, power, area and energy efficiency (for 2-bit** × **4-bit GEMM) of MHE-TPE array with varying scales.** 

|**Table 6**|**Table 6**|**: Area, power, are**|**: Area, power, are**|**: Area, power, are**|**a an**|**d energy efciency (for 2-bit**×**4-**|**d energy efciency (for 2-bit**×**4-**|**d energy efciency (for 2-bit**×**4-**|**d energy efciency (for 2-bit**×**4-**|**bit GEMM) of MHE-TPE a**|**bit GEMM) of MHE-TPE a**|**bit GEMM) of MHE-TPE a**|**bit GEMM) of MHE-TPE a**|**rray wi**|**th varying s**|**cales.**|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**Array**||**Matrix Size**|||**NT**|**Area(**_𝜇𝑚_2**)**||||**Power(mW)**||||**TOPS**|**TOPS/mm**2|**TOPS/W**|
|**MHD**|**Tree**|**M**|**K**|**N**||**TPE**|**MHE**|**LRM**|**TOTAL**|**TPE**|**MHE**|**LRM**|**TOTAL**||||
|||||8/4/2/1|8|166272||31443|203816|245||24|285|16.38|80.39|57.49|
|||||16/8/4/2|16|332544||62887|402340|489||48|563|32.77|81.44|58.20|
|8|16|32|32|32/16/8/4|32|665088|5292|125774|799388|979|15|95|1119|65.54|81.98|58.57|
|||||48/24/12/6|48|997632||188661|1196437|1468||143|1676|98.30|82.16|58.65|
|||||**64/32/16/8**|**64**|**1330176**||**251548**|**1593485**|**1957**||**190**|**2227**|**131.07**|**82.25**|**58.86**|
|**Array**||**Matrix Size**|||**NT**|**Area(**_𝜇𝑚_2**)**||||**Power(mW)**||||**TOPS**|**TOPS/mm**2|**TOPS/W**|
|**MHD**|**Tree**|**M**|**K**|**N**||**TPE**|**MHE**|**LRM**|**TOTAL**|**TPE**|**MHE**|**LRM**|**TOTAL**||||
|||||8/4/2/1|8|205058||39305|251859|304||30|356|20.48|81.32|57.53|
|||||16/8/4/2|16|410115||78611|497100|608||59|701|40.96|82.40|58.43|
|10|16|40|32|32/16/8/4|32|820230|6617|157222|987583|1215|18|119|1392|81.92|82.95|58.85|
|||||**48/24/12/6**|**48**|**1230345**||**235833**|**1478066**|**1823**||**178**|**2082**|**122.88**|**83.14**|**59.02**|
|||||64/32/16/8|64|1640460||314444|1968549|2431||238|2778|163.84|83.23|58.98|
|**Array**||**Matrix Size**|||**NT**|**Area(**_𝜇𝑚_2**)**||||**Power(mW)**||||**TOPS**|**TOPS/mm**2|**TOPS/W**|
|**MHD**|**Tree**|**M**|**K**|**N**||**TPE**|**MHE**|**LRM**|**TOTAL**|**TPE**|**MHE**|**LRM**|**TOTAL**||||
|||||8/4/2/1|8|244244||47166|300398|366||36|429|24.58|81.81|57.29|
|||||16/8/4/2|16|488488||94333|592858|731||71|846|49.15|82.91|58.10|
|12|16|48|32|32/16/8/4|32|976977|7937|188666|1177780|1462|22|143|1682|98.30|83.47|58.44|
|||||**48/24/12/6**|**48**|**1465465**||**283000**|**1762702**|**2194**||**214**|**2511**|**147.46**|**83.65**|**58.72**|
|||||64/32/16/8|64|1953953||377333|2347623|2925||285|3358|196.61|83.75|58.55|



## **4.3 MHE-TPE Array Scalability Analysis** 

In Table 6, we evaluated the scalability of the MHE-TPE array illustrated in Fig. 8 using three optimal configurations to realize a complete MHE-TPE array multi-precision matrix multiplication module. The number of _𝑁𝑇_ primarily influences the dimension _𝑁_ of matrix _𝐵_ ∈ Z _[𝐾]_[×] _[𝑁]_ for four precision levels: INT4, INT8, INT16, and INT32. For example, when _𝑁𝑇_ = 64, the corresponding dimensions _𝑁_ of matrix _𝐵_ are 64, 32, 16, and 8 for INT4, INT8, INT16, and INT32, respectively. In terms of area efficiency, due to the shared nature of the MHE, specifically the broadcast pulsation of encoded values among MHE-TPE units, the area occupied by the MHE is independent of _𝑁𝑇_ . Consequently, as _𝑁𝑇_ increases, the area efficiency improves gradually. However, regarding energy efficiency, for configurations with MHD fanout of 10 and 12 (tree reduction dimension of 16), energy efficiency decreases when _𝑁𝑇_ exceeds 48. This reduction is primarily attributed to the insertion of additional invert buffers required to maintain timing constraints in high fanout conditions, introducing extra power overhead. 

Within the overall architecture, the encoded values output by the MHE are propagated systolically across TPE Tiles via FIFOs. Wherein every 8 TPE Tiles communicate through the LRM. Therefore, most TPE units operate relatively independently, ensuring effective scalability up to certain sizes. On the other hand, excessively large matrix dimensions in DNN systems can adversely impact operator utilization. Hence, in this study, the configuration with MHD fanout is 8, tree reduction dimension is 16, and _𝑁𝑇_ equal to 64, corresponding to matrix dimensions _𝑀_ = 32, _𝐾_ = 32, and _𝑁_ = 64/32/16/8, serves as the baseline for subsequent chip layout and mixed-precision matrix multiplication performance evaluation. 

## **4.4 Transistor Process Corners and Throughput** 

In this section, we evaluate the matrix multiplication performance, area efficiency, and energy efficiency of the MHE-TPE array macro under the configuration MHD = 8, Tree = 16, and _𝑁𝑇_ = 64. The tests cover a wide range of conditions, including multiple operating temperatures (−40[◦] C _,_ 0[◦] C _,_ 25[◦] C _,_ 85[◦] C), operating voltages (0.66 to 0.8 V), process corners (SSG, TT, FFG), and precision modes (4A4B, 8A4B, 8A8B, 16A16B, 16A4B, 16A8B, 32A8B, 32A16B, 32A32B). 

**Table 7: MHE-TPE array under different process corners.** 

|**Temp**|**V**|**Corner**|**Delay (ns)**|**Freq**|**Power(mW)**|
|---|---|---|---|---|---|
|-40◦C|0.72<br>0.81<br>0.66<br>0.77|SSG<br>SSG<br>FFG<br>FFG|0.29<br>0.20<br>0.27<br>0.15|3.2G/800M<br>4.0G/1.0G<br>3.6G/900M<br>4.0G/1.0G|1394<br>2231<br>1354<br>2029|
|0◦C|0.72<br>0.81<br>0.66<br>0.77|SSG<br>SSG<br>FFG<br>FFG|0.29<br>0.20<br>0.23<br>0.16|3.2G/800M<br>4.0G/1.0G<br>4.0G/1.0G<br>4.0G/1.0G|1411<br>2256<br>1487<br>2504|
|25◦C|0.70<br>0.80|TT<br>TT|0.23<br>0.17|3.8G/950M<br>4.0G/1.0G|1564<br>2227|
|85◦C|0.70<br>0.80|TT<br>TT|0.21<br>0.16|4.0G/1.0G<br>4.0G/1.0G|1745<br>2308|



As shown in Table 7, transistor mobility degradation at low temperatures impacts the maximum logic path delay. For example, under the SSG process at 0.72 V, the MHE and MHD operate at 3.2 GHz, while the tree and LRM components operate at 800 MHz. Under the FFG process at 0.66 V, the MHE and MHD run at 3.6 GHz, and the tree and LRM reach 900 MHz. At 0°C and 0.72 V, the 

10 

1634 

MHE-TPE: Multi-Operand High-Radix Encoder for Mixed-Precision Fixed-Point Tensor Processing Engines 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

**==> picture [506 x 103] intentionally omitted <==**

**Figure 11: Mixed-precision throughput and area efficiency under different temperatures and voltages of MHE-TPE array.** 

**==> picture [122 x 119] intentionally omitted <==**

**==> picture [122 x 119] intentionally omitted <==**

**==> picture [122 x 118] intentionally omitted <==**

**==> picture [122 x 118] intentionally omitted <==**

**==> picture [440 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Under 8A8B (b) Under 8A8B (c) Under M, N, K=4096 (d) Under M, N, K=4096<br>**----- End of picture text -----**<br>


**Figure 12: Factors affecting the utilization rate of MHE-TPE array: (a)** _𝑀_ **and** _𝐾_ **dimensions of the GEMM; (b)** _𝑀_ **dimension of the** _𝐴𝑀_ × _𝐾_ **and output buffer size; (c) computational precision of** _𝐴𝑀_ × _𝐾_ **and** _𝐵𝐾_ × _𝑁_ **; (d) precision of** _𝐴𝑀_ × _𝐾_ **and output buffer size.** 

system requires frequency scaling to maintain timing constraints. Under undervolting at room temperature (25°C, 0.7 V), the system operates at 3.8 GHz for MHE/MHD and 950 MHz for compressor tree and LRM. At high temperatures (85°C), electron mobility increases, which alleviates timing concerns but raises dynamic power consumption. Therefore, frequency scaling may be required under extremely low or high temperatures to sustain energy efficiency. 

As illustrated in Fig. 11, peak performance and computational density are achieved due to the MHE-TPE architecture’s fusion of reduction dimensions, ensuring high compute density within each TPE. The precision of Matrix _𝐴_ ’s scaling is mapped onto the temporal dimension of individual TPEs, while the Matrix _𝐵_ ’s precision scaling is distributed across the spatial dimensions of different TPE tiles. This enables near-proportional scaling of both performance per unit area and energy efficiency with respect to operand precision. 

Across varying temperatures and voltages, the 4A4B mode delivers approximately twice the compute density and energy efficiency of the 8A4B mode, 4 times that of 8A8B, and 16 times that of 16A16B. Similarly, the 16A4B mode offers approximately twice the compute density and energy efficiency of the 16A8B mode, 4 times that of 32A8B, 8 times that of 32A16B, and 16 times that of 32A32B, and for energy efficiency reaches its peak at low temperature and low voltage (0°C, 0.66 V), while area efficiency is maximized at room temperature under high voltage conditions (25°C, 0.8 V). 

## **4.5 Factors Affecting Compute Utilization** 

The MHE-TPE array employs a typical WS dataflow. Its key feature lies in writing sub-matrices of matrix _𝐵_ into VPP LUTs to enable 

data and computation reuse, thereby reducing operand bandwidth demands and improving computational energy efficiency. Specifically, increasing the reuse duration of sub-matrix _𝐵_ within the array effectively reduces computational resource waste caused by VPP LUT loading and pipeline idleness. Fig. 12 illustrates three key factors affecting array utilization: 

❶ **Matrix dimensions of computational workload (shown in Fig. 12(a))** . When the partial sum output buffer capacity is sufficient, the outer product computation enables complete reuse of matrix _𝐵_ sub-matrices through a single load. This requires traversing the tiled _𝑀_ -dimension of matrix _𝐴_ . Consequently, array utilization increases with the _𝑀_ -dimension. When _𝑀_ > 1920, utilization can exceed 90% of the theoretical compute efficiency. The reduction dimension _𝐾_ , as the outermost loop, does not affect array utilization. 

❷ **Output buffer size (shown in Fig. 12(b))** . Prolonged reuse of matrix _𝐵_ increases on-chip partial sum storage demands, requiring higher _𝑀_ -dimension tasks to be further partitioned to reduce storage pressure. However, array utilization saturates when the _𝑀_ - dimension reaches a threshold. Thus, a smaller buffer can achieve high utilization (see the yellow region in Fig. 12(b)). 

❸ **Matrix precision and output buffer size (shown in Fig. 12(c)(d))** . MHE-TPE array supports multiple precisions by traversing different bit weights along the temporal dimension of matrix _𝐴_ (referencing Fig. 10). This process shares the same VPP LUTs. Therefore, higher precision for matrix _𝐴_ extends the reuse time of matrix _𝐵_ , further enhancing array utilization. When matrix _𝐴_ ’s precision and output buffer size are both increased simultaneously beyond a specific threshold, array utilization can be stably maintained at a high level. 

11 

1635 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea Q. Wu, J. Zhou, Z. Hu, Z. Zeng, H. Liang, J. Zhu, L. Tao, X. Zhang, Z. Cheng, L. Zhao, Y. Wei, X. Wang, and X. Jin 

**==> picture [506 x 77] intentionally omitted <==**

**Figure 13: MHE-TPE array utilization under Llama3-8b workload (W4A8KV4) during different processing phase and token length. The bars represent the computational load proportions of operators within a single LlamaDecoderLayer, while the black line showcases the computational array utilization under this operator. The red dotted line represents the average utilization. The x-axis represents the** _𝑀_ **,** _𝐾_ **, and** _𝑁_ **dimensions of GEMM operations for each layer, respectively.** 

**==> picture [531 x 183] intentionally omitted <==**

**Figure 14: MHE-TPE array utilization under ResNet50. The bars represent the computational load proportions of operators within the total network, while the black line showcases the computational array utilization under this operator. The red dotted line represents the average utilization. The x-axis represents the** _𝑀_ **,** _𝐾_ **, and** _𝑁_ **dimensions of GEMM operations for each layer, respectively.** 

## **4.6 Compute Utilization under DNN Models** 

_4.6.1 Under Large Language Models Workload._ This study evaluates the utilization efficiency of the MHE-TPE within the LlamaDecoderLayer (comprising LlamaAttention and LlamaMLP modules) on the Llama3-8B model (weight quantization precision W4A8KV4) during both the prompt processing phase (batch size=1) and the token generation phase (batch size=32). 

**In prompt processing phase** : as depicted in Fig. 13(a), when the input sequence length increases from 128 to 1024, the computational load of the attention layer significantly rises from 2% to 12%. This phenomenon arises because the core matrix operation _𝑀_ -dimensions in the _𝑄𝐾𝑉_ projection layer (first layer) and the attention layer (second and third layers) are proportional to the sequence length. Conversely, in other layers (such as the postattention projection and internal FFN), the sequence length becomes the secondary dimension _𝑁_ due to transposition operations, exhibiting a weaker correlation with unit utilization. The computational load of these layers is primarily driven by their high _𝑀_ -dimensions, with the expanded dimension (up to 28,672) of the FFN’s up-projection layer being particularly prominent. This layer emerges as the highest computational operator within the LlamaDecoderLayer and results in peak MHE-TPE array utilization at this 

stage. Consequently, the increase in sequence length effectively enhances the utilization of the corresponding operators by elevating the _𝑀_ -dimension in the _𝑄𝐾𝑉_ projection layer and the attention layer, raising the array’s average utilization from 89.13% to 95.33%. 

**In token generation phase** : as shown in Fig. 13(b), the key difference between this phase and prompt processing lies in the constant sequence length of 1 for the attention layer’s matrix _𝑄_ . To optimize utilization during prompt processing, the attention layer employs the _𝑉[𝑇]_ ( _𝐾𝑄[𝑇]_ ) dataflow rearrangement technique ( _𝑉[𝑇]_ layout). Although the _𝐾𝑉_ cache continuously grows during token generation, the core dimension _𝑀_ of the _𝑉[𝑇]_ remains unchanged, resulting in persistently low utilization during the second stage of attention computation. Given that the computational proportion of the attention layer increases with context expansion, the overall array average utilization exhibits a declining trend. 

_4.6.2 Under CNN Workload._ Fig. 14 displays the variation in MHETPE array GEMM operator utilization for the ResNet50 (input size: 3×224×224) under different batch sizes. When the batch size is 1, the network’s first 20 layers maintain a high average utilization (>80%). The core mechanism lies in the fact that after the convolution layer undergoes img2col transformation, the _𝑀_ -dimension of the executed core GEMM operator corresponds to the spatial resolution (height × width) of the activation map. In the shallow 

12 

1636 

MHE-TPE: Multi-Operand High-Radix Encoder for Mixed-Precision Fixed-Point Tensor Processing Engines 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

**Table 8: Comparison with other architectures.** 

|**Architecture**|**Architecture**|**MHE-TPE Array§**|**Systolic Array** (TPU-like)|**LUT-TC[32]**|**LUTein[16]**|**UNPU[22]**|**Sam. NPU[36]**|**RaPiD[47]**|
|---|---|---|---|---|---|---|---|---|
|**Technology(nm)**||UMC 22|UMC 22|TSMC 28|Samsung28|65|4|7|
|**Supply Voltage (V)**||0.66∼0.81|0.66∼0.81|0.9|/|0.63∼1.1|0.55∼1.0|0.55∼0.75|
|**Frequency (MHz)**||800∼1000|500∼1000|1000|250|200|332∼1196|1000∼1600|
|**TPE Area(mm**2**) †**||1.59(pre-layout)|/|0.187|0.2|3.55|1.88|1.98|
|**Support**<br>**Precision**||A:INT2∼32;<br>B:INT4,8,16,32|dedicated precision|A:FP/INT8, FP/INT16;<br>B: INT1∼INT4|Sparse INT4,<br>7,10,13|INT1∼INT16|INT4,8,16<br>/FP16|FP8,16,32,<br>INT 2,4|
|**TOPS/W †**|4A<br>8A<br>16A<br>32A<br>|**4B:**39.88<br>**4B:**19.94;**8B:**9.97<br>**4B:**9.97;**8B:**4.98;**16B:**2.49<br>**4B:**4.98;**8B:**2.49;**16B:**1.24;**32B:**0.62|**4B:**13.46<br>**4B:**11.64;**8B:**6.73<br>**4B:**10.76;**8B:**5.82;**16B:**3.44<br>**4B:**5.95;**8B:**3.52;**16B:**1.82;**32B:**0.94|/<br>**4B:**15.95;**8B:**7.97<br>**4B:**8.41<br>/|/<br>**7B:**1.99<br>/<br>/|**4B:**6.97<br>**8B:**3.49<br>**16B:**0.1<br>/|**4B:**23.18<br>**8B:**11.59<br>/<br>/|**4B:**40.36<br>/<br>/<br>/|
|**TOPS/mm**2 **†**|4A<br>8A<br>16A<br>32A<br>|**4B:**39.13<br>**4B:**19.56;**8B:**9.78<br>**4B:**9.78;**8B:**4.89;**16B:**2.44<br>**4B:**4.89;**8B:**2.44;**16B:**1.22;**32B:**0.61|**4B:**18.76<br>**4B:**12.67;**8B:**9.32<br>**4B:**8.49;**8B:**5.66;**16B:**3.17<br>**4B:**4.57;**8B:**3.01;**16B:**1.83;**32B:**0.97|/<br>**4B:**30.98;**8B:**15.48<br>**4B:**15.49<br>/|/<br>**7B:**4.535<br>/<br>/|**4B:**0.4<br>**8B:**0.2<br>**16B:**0.1<br>/|**4B:**20.94<br>**8B:**10.47<br>/<br>/|**4B:**52.97<br>/<br>/<br>/|



† Fixed-point tensor processing engines from chip area and power breakdown. § Power was listed in transistor tt 0.7V process corner at 25 degrees Celsius. 

layers of the network, this dimension is typically large, ensuring the efficient utilization of computational units. However, as the network progressively downsamples layer by layer, the activation map size significantly shrinks, causing the _𝑀_ -dimension to decrease accordingly. Simultaneously, the number of output channels gradually increases (corresponding to the _𝑁_ -dimension of GEMM). These inverse variations in dimensions result in a significant decline in utilization dominated by the _𝑀_ -dimension within the deeper layers, ultimately yielding an average utilization of only 60.88%. A viable optimization strategy is to employ a transposed dataflow layout for inference in the deeper layers of the network. This approach transforms the number of output channels (originally the _𝑁_ -dimension) into the _𝑀_ -dimension during matrix operations, making it the core computational dimension. The reduction in spatial dimension (originally the _𝑀_ -dimension) is instead converted into the secondary dimension ( _𝑁_ -dimension). This dataflow transformation aims to enhance resource utilization under single-batch input scenarios. Empirical results show that when the batch size increases to 32, the full utilization of parallelism among samples within a batch can amplify the average utilization to over 96%. 

**==> picture [242 x 92] intentionally omitted <==**

**----- Start of picture text -----**<br>
3.18 mm<br>TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile<br>TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile<br>TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile<br>TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile<br>TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile<br>TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile<br>TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile<br>TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile TPE Tile<br>LRM LRM LRM LRM LRM LRM LRM LRM<br>1.10 mm<br>**----- End of picture text -----**<br>


**Figure 15: MHE-TPE array macro layout.** 

## **4.7 Comparison with Other Architectures** 

To enable a fair comparison, we adopt the energy and performance results measured under standard conditions (25°C, 0.7 V) as the baseline for evaluating our design against prior work. Table 8 summarizes the comparison across multiple tensor computation architectures in both multi-precision and single-precision settings. Comparative analysis of dedicated precision systolic arrays versus the MHE-TPE scheme reveals that, regarding area efficiency, 

consistent with Table 1 conclusions, the dedicated INT4 array demonstrates merely twice the area efficiency of the INT8 array. Under low-precision hybrid computing modes, the MHE-TPE solution outperforms conventional approaches in both area and energy efficiency. Particularly when matrix _𝐵_ precision remains fixed at 4 bits while only increasing matrix _𝐴_ precision, MHE-TPE maintains area efficiency comparable to dedicated architectures. However, as the matrix _𝐵_ bit-width increases (in configurations like 16A16B, 32A8B, 32A16B, and 32A32B), dedicated architectures exhibit significant advantages in both area efficiency and energy efficiency. This phenomenon originates from the universal design of the MHE-TPE scheme: to support 4-bit weight slicing (matrix _𝐵_ bit-slices), its VPP LUT bit-width is constrained within 6 bits. 

The LUT-TensorCore [32] employs a lookup-table-based approach, storing the summation results of 4 activation groups, which are then multiplied with 1-bit weights. To build efficient lookup tables, LUT-TensorCore applies post-training quantization (PTQ) to large language models (LLMs) to eliminate zero bits in weights that would otherwise degrade the multiplication result. However, this method has two major limitations. First, it requires software-level coordination, making it unsuitable for general-purpose matrix computations. Second, it lacks flexibility in supporting a wide range of precision configurations. Its heavy reliance on low-bit weight precision means that in high-bit-width scenarios, the number of partial products from 1-bit multiplications increases significantly, degrading both energy and area efficiency. 

LUTein [16] utilizes a single-operand lookup table based on 4- bit MBE. The core idea stems from the observation that in DNN workloads, INT8-quantized weights and activations exhibit a high proportion of computation concentrated in 4-bit bit-slices. As such, LUTein performs matrix multiplication using zero-skipping within bit slices, allowing INT4 computations to approximate the area density and energy efficiency of INT8 computations in most cases. However, LUTein depends heavily on the statistical distribution of the input data, limiting its generality. Under dense INT8 multiplication, this can lead to imbalanced compute density across precisions, resulting in a drop in both area and energy efficiency. 

UNPU [22] follows a concept similar to LUT-TensorCore. It performs weight-dependent 1-bit serial multiplication using a linear combination of three activation values as the lookup index, followed 

13 

1637 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Q. Wu, J. Zhou, Z. Hu, Z. Zeng, H. Liang, J. Zhu, L. Tao, X. Zhang, Z. Cheng, L. Zhao, Y. Wei, X. Wang, and X. Jin 

by partial product accumulation. The number of partial accumulations is constrained by the bit width of the weights. UNPU uses 8 lookup tables for the possible multiplication-accumulation outputs. In contrast, the dual-operand encoding in our design also employs 8 lookup tables, but each table effectively computes the multiplication and accumulation of two operands. This results in more than double the throughput under INT8 precision, using the same LUT storage depth. 

Samsung’s NPU [36] design for mobile SoCs integrates multiple multiplier-width configurations within its fixed-point tensor processing PE, as shown in Fig. 3(b). The PE includes one INT8 unit, two INT8×INT4 units, and one INT4 unit. When executing INT4 tensor computation, all 4 multipliers are activated. When executing INT8 computation, one INT8 and two INT8×INT4 units are activated alongside the reduction tree. While this improves INT8 computation efficiency under multi-precision, it leads to severe hardware underutilization in INT4 computations. For example, executing INT8 multiplication wastes one INT4 multiplier, while INT4 computation leaves three INT8 multipliers idle. 

IBM’s RaPiD NPU [47] adopts a dedicated hardware parallelism strategy, implementing eight INT4 and sixteen INT2 multipliers in its integer pipeline engine to match operand bit-widths. This design trades area for energy efficiency at fixed precision levels, which makes it difficult to scale to higher or mixed-precision computation. 

The OPT4E[52] focuses on INT8 computation with specialized bit-sparse encoding techniques. Compared to these architectures, our multi-precision general-purpose compute design achieves comparable area and energy efficiency to dedicated INT8 MAC-based and sparse bit-encoded architectures. Therefore, our approach demonstrates robust precision scalability while maintaining balanced compute density across various precision levels. 

Bitfusion[40] employs a fundamental design based on 1-bit multiplication, constructing high-precision multiplication by configuring 2-bit multiplication units supplemented with shifters. For vector reduction operations, the computational results from multiple calculation modules require shifting and accumulation. However, this approach exhibits two significant limitations: First, it does not encode the multiplicand (e.g., via Booth encoding), resulting in an excessive number of PPs terms (requiring n shifting and accumulation operations for n-bit multiplication). In contrast, MBE can reduce PPs terms to ⌈ _𝑛_ /2⌉; in our optimized solution for GEMM, this can be further reduced to ⌈ _𝑛_ × _𝐾_ /4⌉. Consequently, for high-precision, large-bit-width multiplication operations, Bitfusion demonstrates relatively low resource efficiency in calculation module consumption due to the generation of excessive PPs. Second, its reduction logic lacks efficiency. Since this scheme accumulates the shifted outputs of calculation modules, the adder must reserve the maximum bit width required to accommodate the largest shift amount. Conversely, the MHE-TPE adopts a same-bit-weight vector reduction strategy (i.e., parallel reading of vector elements sharing the same bit-weight in matrix _𝐴_ ), thereby reducing resource overhead through the reuse of low-bit-width compressor trees. 

VecPAC[44] enhances flexibility via a coarse-grained reconfigurable array (CGRA) architecture, but this comes at the cost of additional area overhead from physical interconnects and routing selectors. Furthermore, its highly modularized isolation design constrains the potential for fusion optimization between units (e.g., 

multi-number addition could be fused into half-adders followed by full-adders rather than using full-adder trees). 

## **5 Conclusion** 

This study begins by analyzing the redundancy in the spatial and temporal reduction dimensions within typical spatial accelerators [4, 7, 10, 12, 14, 16–20, 24–26, 29, 33, 34, 37, 39, 41, 49, 54]. To optimize the resolution of this issue, we propose a high-radix dualoperand encoder to halve the number of PPs in vector inner-product reductions, thereby reducing the area and energy overheads of the accumulation trees. Furthermore, we investigate the issue of imbalanced compute density under mixed-precision configurations in current tensor computation engines. To address this, we introduce a novel computational paradigm based on bit-slice encoding, partial product generation, and accumulation. Our design maps multi-precision computation by projecting the precision of matrix _𝐴_ onto the temporal domain and that of matrix _𝐵_ onto the spatial domain. This enables efficient mixed-precision matrix multiplication and alleviates compute density imbalance in low-precision operations. 

## **References** 

- [1] 2017. Nvidia tesla v100 gpu architecture white paper. https://images.nvidia.com/ content/volta-architecture/pdf/volta-architecture-whitepaper.pdf. 

- [2] 2020. Nvidia A100 gpu architecture white paper. https://images.nvidia.com/aemdam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf. 

- [3] Syed Asad Alam, Andrew Anderson, Barbara Barabasz, and David Gregg. 2022. Winograd convolution for deep neural networks: Efficient point selection. _ACM Transactions on Embedded Computing Systems_ 21, 6 (2022), 1–28. 

- [4] J. Albericio, P. Judd, A. Delmás, S. Sharify, and A. Moshovos. 2016. Bit-pragmatic Deep Neural Network Computing. arXiv:1610.06920 [cs.LG] https://arxiv.org/ abs/1610.06920 

- [5] Orest J Bedrij. 1962. Carry-select adder. _IRE Transactions on Electronic Computers_ 3 (1962), 340–346. 

- [6] Yaniv Blumenfeld, Itay Hubara, and Daniel Soudry. 2024. Towards Cheaper Inference in Deep Networks with Lower Bit-Width Accumulators. _arXiv preprint arXiv:2401.14110_ (2024). 

- [7] Stephen Cass. 2019. Taking AI to the edge: Google’s TPU now comes in a maker-friendly package. _IEEE Spectrum_ 56, 5 (2019), 16–17. 

- [8] Yi Chen, Yongwei Zhao, Yifan Hao, Yuanbo Wen, Yuntao Dai, Xiaqing Li, Yang Liu, Rui Zhang, Mo Zou, Xinkai Song, et al. 2024. Cambricon-C: Efficient 4-Bit Matrix Unit via Primitivization. In _2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . IEEE, 538–550. 

- [9] Fu-Chiung Cheng, Stephen H Unger, and Michael Theobald. 2000. Self-timed carry-lookahead adders. _IEEE Trans. Comput._ 49, 7 (2000), 659–672. 

- [10] Alberto Delmas Lascorz, Patrick Judd, Dylan Malone Stuart, Zissis Poulos, Mostafa Mahmoud, Sayeh Sharify, Milos Nikolic, Kevin Siu, and Andreas Moshovos. 2019. Bit-tactical: A software/hardware approach to exploiting value and bit sparsity in neural networks. In _Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems_ . 749–763. 

- [11] Aamir A Farooqui and Vojin G Oklobdzija. 1998. General data-path organization of a MAC unit for VLSI implementation of DSP processors. In _1998 IEEE International Symposium on Circuits and Systems (ISCAS)_ , Vol. 2. IEEE, 260–263. 

- [12] Axel Feldmann and Daniel Sanchez. 2023. Spatula: A hardware accelerator for sparse matrix factorization. In _Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture_ . 91–104. 

- [13] Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W Mahoney, and Kurt Keutzer. 2022. A survey of quantization methods for efficient neural network inference. In _Low-power computer vision_ . Chapman and Hall/CRC, 291–326. 

- [14] Christopher Grimm, Jinseok Lee, and Naveen Verma. 2024. Training Neural Networks With In-Memory-Computing Hardware and Multi-Level Radix-4 Inputs. _IEEE Transactions on Circuits and Systems I: Regular Papers_ (2024). 

- [15] Oscar Gustafsson, Andrew G Dempster, and Lars Wanhammar. 2004. Multiplier blocks using carry-save adders. In _2004 IEEE International Symposium on Circuits and Systems (IEEE Cat. No. 04CH37512)_ , Vol. 2. IEEE, II–473. 

- [16] Dongseok Im and Hoi-Jun Yoo. 2024. Lutein: Dense-sparse bit-slice architecture with radix-4 lut-based slice-tensor processing units. In _2024 IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . IEEE, 747–759. 

14 

1638 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

MHE-TPE: Multi-Operand High-Radix Encoder for Mixed-Precision Fixed-Point Tensor Processing Engines 

- [17] Zhe Jia, Blake Tillman, Marco Maggioni, and Daniele Paolo Scarpazza. 2019. Dissecting the graphcore ipu architecture via microbenchmarking. _arXiv preprint arXiv:1912.03413_ (2019). 

- [18] Norm Jouppi, George Kurian, Sheng Li, Peter Ma, Rahul Nagarajan, Lifeng Nai, Nishant Patil, Suvinay Subramanian, Andy Swing, Brian Towles, et al. 2023. Tpu v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings. In _Proceedings of the 50th Annual International Symposium on Computer Architecture_ . 1–14. 

- [19] Norman P Jouppi, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Boden, Al Borchers, et al. 2017. In-datacenter performance analysis of a tensor processing unit. In _Proceedings of the 44th annual international symposium on computer architecture_ . 1–12. 

- [20] Patrick Judd, Jorge Albericio, Tayler Hetherington, Tor M Aamodt, and Andreas Moshovos. 2016. Stripes: Bit-serial deep neural network computing. In _2016 49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . IEEE, 1–12. 

- [21] Shiann-Rong Kuang, Jiun-Ping Wang, and Cang-Yuan Guo. 2009. Modified booth multipliers with a regular partial product array. _IEEE Transactions on Circuits and Systems II: Express Briefs_ 56, 5 (2009), 404–408. 

- [22] Jinmook Lee, Changhyeon Kim, Sanghoon Kang, Dongjoo Shin, Sangyeob Kim, and Hoi-Jun Yoo. 2018. UNPU: An energy-efficient deep neural network accelerator with fully variable weight bit precision. _IEEE Journal of Solid-State Circuits_ 54, 1 (2018), 173–185. 

- [23] Sae Kyu Lee, Ankur Agrawal, Joel Silberman, Matthew Ziegler, Mingu Kang, Swagath Venkataramani, Nianzheng Cao, Bruce Fleischer, Michael Guillorn, Matthew Cohen, et al. 2021. A 7-nm four-core mixed-precision AI chip with 26.2TFLOPS hybrid-FP8 training, 104.9-TOPS INT4 inference, and workload-aware throttling. _IEEE Journal of Solid-State Circuits_ 57, 1 (2021), 182–197. 

- [24] Gang Li, Weixiang Xu, Zhuoran Song, Naifeng Jing, Jian Cheng, and Xiaoyao Liang. 2022. Ristretto: An atomized processing architecture for sparsitycondensed stream flow in CNN. In _2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . IEEE, 1434–1450. 

- [25] Jiansong Li and Zihan Jiang. 2020. Performance analysis of cambricon mlu100. In _Benchmarking, Measuring, and Optimizing: Second BenchCouncil International Symposium, Bench 2019, Denver, CO, USA, November 14–16, 2019, Revised Selected Papers 2_ . Springer, 57–66. 

- [26] Heng Liao, Jiajin Tu, Jing Xia, Hu Liu, Xiping Zhou, Honghui Yuan, and Yuxing Hu. 2021. Ascend: a scalable and unified architecture for ubiquitous deep neural network computing: Industry track paper. In _2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . IEEE, 789–801. 

- [27] Fangxin Liu, Ning Yang, Haomin Li, Zongwu Wang, Zhuoran Song, Songwen Pei, and Li Jiang. 2024. Spark: Scalable and precision-aware acceleration of neural networks via efficient encoding. In _2024 IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . IEEE, 1029–1042. 

- [28] Yun-Chen Lo and Ren-Shuo Liu. 2023. Bucket getter: A bucket-based processing engine for low-bit block floating point (bfp) dnns. In _Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture_ . 1002–1015. 

- [29] Hang Lu, Liang Chang, Chenglong Li, Zixuan Zhu, Shengjian Lu, Yanhuan Liu, and Mingzhe Zhang. 2021. Distilling bit-level sparsity parallelism for general purpose deep learning acceleration. In _MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture_ . 963–976. 

- [30] Wenyan Lu, Guihai Yan, Jiajun Li, Shijun Gong, Yinhe Han, and Xiaowei Li. 2017. Flexflow: A flexible dataflow accelerator architecture for convolutional neural networks. In _2017 IEEE International Symposium on High Performance Computer Architecture (HPCA)_ . IEEE, 553–564. 

- [31] Arnab Neelim Mazumder, Jian Meng, Hasib-Al Rashid, Utteja Kallakuri, Xin Zhang, Jae-Sun Seo, and Tinoosh Mohsenin. 2021. A survey on the optimization of neural network accelerators for micro-ai on-device inference. _IEEE Journal on Emerging and Selected Topics in Circuits and Systems_ 11, 4 (2021), 532–547. 

- [32] Zhiwen Mo, Lei Wang, Jianyu Wei, Zhichen Zeng, Shijie Cao, Lingxiao Ma, Naifeng Jing, Ting Cao, Jilong Xue, Fan Yang, et al. 2024. Lut tensor core: Lookup table enables efficient low-bit llm inference acceleration. _arXiv preprint arXiv:2408.06003_ (2024). 

- [33] Thomas Norrie, Nishant Patil, Doe Hyun Yoon, George Kurian, Sheng Li, James Laudon, Cliff Young, Norman Jouppi, and David Patterson. 2021. The design process for Google’s training chips: TPUv2 and TPUv3. _IEEE Micro_ 41, 2 (2021), 56–63. 

- [34] Yunjie Pan, Jiecao Yu, Andrew Lukefahr, Reetuparna Das, and Scott Mahlke. 2023. BitSET: Bit-serial early termination for computation reduction in convolutional neural networks. _ACM Transactions on Embedded Computing Systems_ 22, 5s (2023), 1–24. 

_Solid-State Circuits_ 58, 1 (2022), 189–202. 

   - [37] Raghu Prabhakar, Sumti Jairath, and Jinuk Luke Shin. 2022. Sambanova sn10 RDU: A 7nm dataflow architecture to accelerate software 2.0. In _2022 IEEE International Solid-State Circuits Conference (ISSCC)_ , Vol. 65. IEEE, 350–352. 

   - [38] Mark R Santoro and Mark A Horowitz. 1989. SPIM: a pipelined 64* 64-bit iterative multiplier. _IEEE journal of solid-state circuits_ 24, 2 (1989), 487–493. 

   - [39] Sayeh Sharify, Alberto Delmas Lascorz, Mostafa Mahmoud, Milos Nikolic, Kevin Siu, Dylan Malone Stuart, Zissis Poulos, and Andreas Moshovos. 2019. Laconic deep learning inference acceleration. In _Proceedings of the 46th International Symposium on Computer Architecture_ . 304–317. 

   - [40] Hardik Sharma, Jongse Park, Naveen Suda, Liangzhen Lai, Benson Chau, Joon Kyung Kim, Vikas Chandra, and Hadi Esmaeilzadeh. 2018. Bit fusion: bit-level dynamically composable architecture for accelerating deep neural networks. In _Proceedings of the 45th Annual International Symposium on Computer Architecture_ (Los Angeles, California) _(ISCA ’18)_ . IEEE Press, 764–775. https://doi.org/10.1109/ISCA.2018.00069 

   - [41] Man Shi, Vikram Jain, Antony Joseph, Maurice Meijer, and Marian Verhelst. 2024. BitWave: Exploiting Column-Based Bit-Level Sparsity for Deep Learning Acceleration. In _2024 IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . IEEE, 732–746. 

   - [42] Jujavarapu Sravana, S. K. Hima Bindhu, K. Sharvani, P. Sai Preethi, Saptarshi Sanyal, Vallabhuni Vijay Vallabhuni Vijay, and Rajeev Ratna Vallabhuni. 2022. Implementation of Spurious Power Suppression based Radix-4 Booth Multiplier using Parallel Prefix Adders. In _2021 4th International Conference on Recent Trends in Computer Science and Technology (ICRTCST)_ . 428–433. 

   - [43] Synopsys Inc. 2022. _Design Compiler User Guide_ . 

   - [44] Cheng Tan, Deepak Patil, Antonino Tumeo, Gabriel Weisz, Steve Reinhardt, and Jeff Zhang. 2023. Vecpac: A vectorizable and precision-aware cgra. In _2023 IEEE/ACM International Conference on Computer Aided Design (ICCAD)_ . IEEE, 1–9. 

   - [45] Fengbin Tu, Shouyi Yin, Peng Ouyang, Shibin Tang, Leibo Liu, and Shaojun Wei. 2017. Deep convolutional neural network architecture with reconfigurable computation patterns. _IEEE Transactions on Very Large Scale Integration (VLSI) Systems_ 25, 8 (2017), 2220–2233. 

   - [46] Sreehari Veeramachaneni, Kirthi M Krishna, Lingamneni Avinash, Sreekanth Reddy Puppala, and MB Srinivas. 2007. Novel architectures for high-speed and low-power 3-2, 4-2 and 5-2 compressors. In _20th International Conference on VLSI Design held jointly with 6th International Conference on Embedded Systems (VLSID’07)_ . IEEE, 324–329. 

   - [47] Swagath Venkataramani, Vijayalakshmi Srinivasan, Wei Wang, Sanchari Sen, Jintao Zhang, Ankur Agrawal, Monodeep Kar, Shubham Jain, Alberto Mannari, Hoang Tran, et al. 2021. RaPiD: AI accelerator for ultra-low precision training and inference. In _2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)_ . IEEE, 153–166. 

   - [48] Christopher S Wallace. 1964. A suggestion for a fast multiplier. _IEEE Transactions on electronic Computers_ 1 (1964), 14–17. 

   - [49] Gang Wang, Siqi Cai, Wenjie Li, Dongxu Lyu, and Guanghui He. 2024. BSViT: A Bit-Serial Vision Transformer Accelerator Exploiting Dynamic Patch and Weight Bit-Group Quantization. _IEEE Transactions on Circuits and Systems I: Regular Papers_ (2024). 

   - [50] Junbin Wang, Shaoxia Fang, Xi Wang, Jiangsha Ma, Taobo Wang, and Yi Shan. 2021. High-performance mixed-low-precision cnn inference accelerator on fpga. _IEEE Micro_ 41, 4 (2021), 31–38. 

   - [51] Qizhe Wu, Yuchen Gui, Zhichen Zeng, Xiaotian Wang, Huawen Liang, and Xi Jin. 2024. EN-T: Optimizing Tensor Computing Engines Performance via EncoderBased Methodology. In _2024 IEEE 42nd International Conference on Computer Design (ICCD)_ . IEEE, 608–615. 

   - [52] Qizhe Wu, Huawen Liang, Yuchen Gui, Zhichen Zeng, Zerong He, Linfeng Tao, Xiaotian Wang, Letian Zhao, Zhaoxi Zeng, Wei Yuan, Wei Wu, and Xi Jin. 2025. Exploring the Performance Improvement of Tensor Processing Engines through Transformation in the Bit-weight Dimension of MACs. In _2025 IEEE International Symposium on High Performance Computer Architecture (HPCA)_ . 685–700. 

   - [53] Rui Xu, Sheng Ma, Yang Guo, and Dongsheng Li. 2023. A survey of design and optimization for systolic array-based dnn accelerators. _Comput. Surveys_ 56, 1 (2023), 1–37. 

   - [54] Jianxun Yang, Zhao Zhang, Zhuangzhi Liu, Jing Zhou, Leibo Liu, Shaojun Wei, and Shouyi Yin. 2021. Fusekna: Fused kernel convolution based accelerator for deep neural networks. In _2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . IEEE, 894–907. 

- [35] Gunho Park, Jaeha Kung, and Youngjoo Lee. 2023. Simplified Compressor and Encoder Designs for Low-Cost Approximate Radix-4 Booth Multiplier. _IEEE Transactions on Circuits and Systems II: Express Briefs_ 70, 3 (2023), 1154–1158. 

- [36] Jun-Seok Park, Changsoo Park, Suknam Kwon, Taeho Jeon, Yesung Kang, Heonsoo Lee, Dongwoo Lee, James Kim, Hyeong-Seok Kim, YoungJong Lee, et al. 2022. A multi-mode 8k-MAC HW-utilization-aware neural processing unit with a unified multi-precision datapath in 4-nm flagship mobile SoC. _IEEE Journal of_ 

15 

1639 

