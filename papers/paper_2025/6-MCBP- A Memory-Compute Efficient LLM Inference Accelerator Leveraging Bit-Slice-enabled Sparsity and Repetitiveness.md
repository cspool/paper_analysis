## **MCBP: A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness** 

Huizheng Wang 

Zichuan Wang School of Integrated Circuits Tsinghua University Beijing, China wang.zichuan@foxmail.com 

## Zhiheng Yue 

School of Integrated Circuits Tsinghua University Beijing, China wanghz22@mails.tsinghua.edu.cn 

School of Integrated Circuits Tsinghua University Beijing, China yuezh20@mails.tsinghua.edu.cn 

Yousheng Long School of Integrated Circuits Tsinghua University Beijing, China longys21@mails.tsinghua.edu.cn 

## Taiquan Wei 

Jianxun Yang School of Integrated Circuits Tsinghua University Beijing, China jianxunyang@hotmail.com 

School of Integrated Circuits Tsinghua University Beijing, China weitq20@mails.tsinghua.edu.cn 

## Shaojun Wei 

## Chao Li 

## Yang Wang 

Department of Computer Science and School of Integrated Circuits Engineering Tsinghua University Shanghai Jiao Tong University Beijing, China Shanghai, China wsj@tsinghua.edu.cn lichao@cs.sjtu.edu.cn 

School of Integrated Circuits Tsinghua University Beijing, China wangyang_imec@mail.tsinghua.edu.cn 

Shouyi Yin 

## Yang Hu[∗] 

School of Integrated Circuits Tsinghua University Beijing, China Shanghai Artificial Intelligence Laboratory Shanghai, China yinsy@tsinghua.edu.cn 

School of Integrated Circuits Tsinghua University Beijing, China hu_yang@tsinghua.edu.cn 

## **Abstract** 

coding (BSTC), which reduces weight access via exploiting significant sparsity in high-order bit-slice weight; 3) Bit-grained progressive prediction (BGPP), which reduces KV cache access by leveraging early-termination-based bit-grained prediction. These techniques, supported by custom accelerator designs, effectively alleviate the burden in GEMM, weight access, and KV cache access. Extensive experiments on 26 benchmarks show that MCBP achieves 9 _._ 43× speed up and 31 _._ 1× higher energy efficiency than Nvidia A100 GPU. Compared to SOTA Transformer accelerators, MCBP achieves 35×, 5 _._ 2× and 3 _._ 2× energy saving than Spatten, FACT and SOFA, respectively. 

Large language models (LLMs) face significant inference latency due to inefficiencies in GEMM operations, weight access, and KV cache access, especially in real-time scenarios. This highlights the need for a versatile compute-memory efficient accelerator. Unfortunately, existing Transformer accelerators struggle to address both aspects simultaneously, as they focus on value-level processing, missing fine-grained opportunities to optimize computation and memory collaboratively. This paper introduces MCBP, a bitgrained compute-memory efficient algorithm-hardware co-design that leverages bit-slice (BS) enabled repetitiveness and sparsity to accelerate LLM inference. MCBP features three key innovations: 1) BS-repetitiveness-enabled computation reduction (BRCR), which eliminates redundant GEMM computations via leveraging redundancy hidden among BS vectors; 2) BS-sparsity-enabled two-state 

## **Keywords** 

Transformer accelerator, Bit-serial, Repetition, Sparsity, Latency 

## **ACM Reference Format:** 

∗Corresponding author 

Huizheng Wang, Zichuan Wang, Zhiheng Yue, Yousheng Long, Taiquan Wei, Jianxun Yang, Yang Wang, Chao Li, Shaojun Wei, Yang Hu, and Shouyi Yin. 2025. MCBP: A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness. In _58th IEEE/ACM International Symposium on Microarchitecture (MICRO ’25), October 18–22, 2025, Seoul, Republic of Korea._ ACM, New York, NY, USA, 17 pages. https: //doi.org/10.1145/3725843.3756037 

This work is licensed under a Creative Commons Attribution 4.0 International License. _MICRO ’25, Seoul, Republic of Korea_ 

© 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1573-0/25/10 https://doi.org/10.1145/3725843.3756037 

1592 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Wang et al. 

**==> picture [242 x 72] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) End-to-end latency breakdown for Llama7B (b) Summary of SOTA Transformer accelerators<br>GEMM Weight load KV load Others GEMM Weight load KV load<br>100<br>8060 SpattenSanger [[1]][[2]] Head pruning Val top-kVal top-k #<br>FACT [[3]] Mixed P Val top-k<br>40<br>20 SOFA MCBP [[4]] Bit S+R * Bit-slice level Val top-kBit top-k<br>0 1k 2k 4k 8k 16k 32k 64k 128k * Bit Sparsity + Repetition # Value-level top-k prediction<br>Prompt length [1] MICRO 22 [2] HPCA 22 [3] ISCA 23 [4] MICRO 24<br>Norm latency (%)<br>**----- End of picture text -----**<br>


**Figure 1: (a) Key bottleneck breakdown of end-to-end latency for Llama7B (Batch=4) on NVIDIA A100 GPU with TensorRTLLM. (b) Summary of current Transformer accelerators.** 

## **1 Introduction** 

Large language models (LLMs) are transforming AI with impressive capabilities across tasks, like code generation [10, 65, 76], chatbots [12, 87]. As many LLM-based services rely on real-time interactions [40, 51, 70], inference latency has emerged as a critical performance metric. 

LLM inference [84] comprises two stages: prefill and decoding. In the prefill stage, the model processes all input tokens (i.e., prompt) in parallel to generate the first token, while storing intermediate Key/Value (KV) tensors, known as the _KV Cache_ . In the subsequent decoding stage, the model generates tokens autoregressively, each iteration requiring access to the full model weights and KV cache. 

However, the differing processing characteristics of the prefill and decoding stages, each presenting distinct resource intensities, make end-to-end inference optimization more challenging. Fig. 1 (a) shows an end-to-end latency breakdown for LLaMA-7B under varying prompt lengths, including both the prefill and decoding stages. In this setting, the decoding is fixed at 16 tokens. The major latency contributors are categorized into GEMM computation, weight loading, KV cache loading, and others. The results indicate that all three factors significantly impact end-to-end latency across different prompt conditions. For short prompts (e.g., 1k tokens), weight loading during the decoding stage dominates, accounting for 52 _._ 4% of the latency. As prompt length increases, GEMM computation in prefill stage and KV cache loading during decoding emerge as the primary bottlenecks. **These trends highlight the need for joint optimization of GEMM computation, weight access, and KV cache access to enhance end-to-end inference efficiency.** 

As summarized in Fig.1 (b), though numerous Transformer accelerators have been proposed [25, 26, 59, 72, 74, 75, 92, 94, 104, 113], most of them focus on leveraging token sparsity to mitigate the quadratic complexity of attention, which becomes a bottleneck for long inputs. Although this also partially reduces the KV load, their value-level top- _𝑘_ prediction involves redundant computation and memory overhead, leading to inefficiency. In addition, while FACT [72] and Spatten [94] can partially mitigate the GEMM and weight load bottlenecks via mixed-precision computation and head pruning, respectively, they lack a holistic optimization strategy that addresses all performance bottlenecks. **These limitations motivate a specialized LLM accelerator capable of jointly optimizing GEMM computation, weight access, and KV cache access.** 

We conduct an in-depth analysis of the root causes behind computation and memory inefficiencies in LLM inference, identifying that these challenges can be effectively addressed via a co-designed bit-level data storage and computation scheme. As illustrated in Fig. 2 (c), this work takes the first step toward addressing all major LLM bottlenecks through a unified bit-level optimization strategy. 

As depicted in Fig. 2 (a), we introduce the "grouped bit-slice (BS)" effect, wherein redundancy among BS vectors can be maximally exploited to reduce computation complexity while minimizing extra overhead. Throughout this paper, for an INT-quantized _𝑘_ -bit vector, it can be decomposed into _𝑘_ individual 1-bit vectors, and referred to as bit-slice vectors for clarity. Assuming that two bit slices of weight vectors are multiplied with a set of INT8 vectors X, resulting in outputs _𝑌_ 0 and _𝑌_ 1. As exemplified in Fig. 2 (a), computing _𝑌_ 0 and _𝑌_ 1with a naïve BS-vector-isolation strategy requires 4 additions, whereas the group BS approach needs at most 2 ADDs, by leveraging the intrinsic repetitiveness among BS vectors. 

However, it is non-trivial to harness the newly identified redundancy and sparsity at bit level. The optimal granularity for bit-level processing and data compression must be carefully determined to avoid diminishing returns from control overhead due to overly fine-grained processing. Specifically, we elaborate on three key opportunities and challenges that arise from adopting bit-level processing and data compression: 

a) **Unexploited redundancy among BS vectors** . Existing designs lack an effective method to exploit redundancy across BS vectors without incurring significant bit-level control overhead. While some prior works [31, 79, 93, 95] have explored leveraging redundancy across convolution channels in CNNs to accelerate computation, such techniques are not applicable to LLMs. This is due to: (1) the relatively small number of channels in CNNs, which eliminates the need for fine-grained granularity control; and (2) the small convolution kernel sizes, which incur small matching overhead for repetitive items. By contrast, the huge matrix sizes in LLMs make it challenging to efficiently identify redundancy across BS vectors in hardware, and highlight the need for a carefully selected grouping granularity to balance efficiency and overhead. 

b) **Untapped sparsity resided in BS weight matrix** , due to the mismatch between value-level compression and inherent BS level sparsity. This limitation stems from the conventional valuecentric memory storage paradigm, which inherently favors valuelevel compression techniques [28, 29, 110]. While such methods are straightforward and widely adopted, they hinder the full exploitation of the fine-grained sparsity present at the bit-slice level. This underscores the need for a bit-dimensional compression strategy, with consistent data organization across the memory hierarchy. 

c) **Inefficient Top-k prediction mechanism** , due to redundant KV cache access. The widely used top- _𝑘_ mechanism in LLMs alleviates attention complexity by speculating attention sparsity and avoiding trivial token computation [72, 92, 94, 104]. However, current value-based top- _𝑘_ prediction is inefficient, leading to redundant IO traffic, which in turn makes the prediction itself become a bottleneck in latency. A finer-grained top- _𝑘_ mechanism is needed to reduce I/O overhead while maintaining sparsity effectiveness. 

To this end, we propose an algorithm-hardware co-design for LLM inference optimization, named MCBP. It features three key designs that correlate to three challenges, as shown in Fig. 2 (b). 

1593 

MCBP: A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness 

**==> picture [242 x 125] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Illustration for bit-slice (BS) sub-matrix and vector  (c)  Comparison with traditional<br>A  k -bit weight matrix redundancExploitable y overheadControl Current LLM acceleratorsLLM accelerators<br>Bit-Slice vector Low High<br>Weight  Value-level<br>LSB Bit-Slice vectorGrouped High Low GEMM load independent optimization<br>YY01==Y0  BS vector-isolation strategy 00=X111+X004, 10Y1MSB=X11 1xx+XXX300+XXX114 XX(4 ADDS)22 XX33 XX44 TT (RI=XRe Grouped BS vectors 00 petitive item111+X00 4 01 11 ) x X0YY(2 ADDSX01=RI=RI+X1 X2 X33) X4 T KV cache load MCBP (bit-level redundancy)Missed Opportunity<br>(b) Illustration for MCBP optimizations in LLM inference Weight  Bit-level<br>GEMM Weight Load KV Cache Load GEMM load comprehensive<br>optimization<br>Prefill stage Decoding stage<br>Bit-level Sparsity<br>KV cache  Bit-level Prediction<br>BRCR (Sec 3.1) BSTC (Sec 3.2) BGPP (Sec 3.3) load Bit-level Repetitiveness<br>**----- End of picture text -----**<br>


**Figure 2: Comparison between MCBP and existing works.** 

1) We propose a BS-repetitiveness-enabled computation reduction (BRCR) strategy for accelerating GEMM. It identifies and reuses repetitive computations between multiple weight-BS vectors by grouping them at an appropriate granularity. This eliminates repetitive operations among grouped vectors while amortizing bit-level control overhead across them. 

2) We propose a BS-sparsity-enabled two-state coding (BSTC) for weight de/compression. It strategically employs bit-slice independent encoding to exploit the significant sparsity commonly hidden in high-order bit slices. Meanwhile, we perform a joint exploration of BSTC granularity and BRCR granularity, identifying the optimal granularity configuration for seamless weight decompression and computation that maximizes overall system benefits. 

3) We design a bit-grained progressive prediction (BGPP) mechanism, to reduce unnecessary KV cache traffic during the attention sparsity prediction stage. BGPP employs a progressive bit-level filter to incrementally eliminate trivial Keys in each prediction round, enabling early termination to avoid redundant computation and memory access associated with them. 

To support the above optimization mechanisms effectively, we design a dedicated accelerator named MCBP: 1) For BRCR, it employs a Content Addressable Memory (CAM) to accelerate the identification for repetitive computations, thus significantly reducing merging latency for repetitive operations. 2) For BSTC, lightweight and customized encoders/decoders are designed to enhance data compression and decompression efficiency. Additionally, it reformulates the data layout in memory to facilitate seamless bit-prioritized computation, thereby reducing data reorder overhead. 3) For BGPP, a bit-grained adaptive threshold-aware clock-gated prediction module is designed to achieve low-power attention sparsity prediction. The MCBP accelerator achieves an average energy efficiency of 22740 GOPS/W, which is 31 _._ 1×, 35×, 5 _._ 2× and 3 _._ 2× higher than A100 GPU, SOTA accelerator Spatten, FACT and SOFA, respectively. 

## **2 Background and Motivation** 

## **2.1 Large Language Models (LLMs)** 

LLMs [4, 9, 85] are based on Transformer architectures [90]. Initially, a length- _𝑆_ sequence is projected into three spaces, termed Query (Q), Key (K) and Value (V), respectively. Next, Q and K are multiplied to generate an attention matrix A with R _[𝑆]_[×] _[𝑆]_ , which represents the correlation of each token pair. The attention matrix is then passed 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

**==> picture [242 x 95] intentionally omitted <==**

**----- Start of picture text -----**<br>
❶ Pre-compute stage [0,3] ❸ Formal compute stage<br>(via low bit-width)<br>Only load and compute<br>Q (4bit) Esti. Atten S H Vital KVs<br>x = Output<br>K (4bit) S V H<br>sort  Top-2 indices H S P<br>[0,3] S K [T] S = QK [T]<br>(Pick top- k  ids) H Q<br>❷ Top- k  sort stage Simultaneously optimize for QK [T] , [ softmax, and P*V]<br>MatMul<br>MatMul Softmax<br>**----- End of picture text -----**<br>


**Figure 3: Top-** _𝑘_ **sparsity prediction for attention acceleration.** 

through a softmax operation and multiplied with V activation, resulting in a matrix O ∈ R _[𝑆]_[×] _[𝐻]_ , where _𝐻_ denotes hidden dimension. Finally, a feed-forward network generates the output results. 

**LLM Integer Quantization** . Quantization reduces LLMs’ compute and memory costs. Early work like Q8-BERT [109] achieves INT8 weight quantization with minimal accuracy loss. In 2022, LLM.int8 [17] scaled INT8 quantization to 175B models with few INT16 outliers. SmoothQuant [101] later enabled lossless 8-bit weight and activation quantization for LLMs with up to 530B. In 2024, Atom [112] implements 8-bit KV cache quantization. Quantization has become a prominent trend for deploying LLMs, supported by frameworks like TensorRT [66]. Thus, optimizing compute and memory access for integer-quantized LLMs is an increasingly critical topic. 

## **2.2 Attention Sparsity and Top-** _𝑘_ **Prediction** 

The standard attention mechanism in LLMs captures global context correlation via dense attention matrices. However, weak correlations between tokens produce many small attention scores, which are further suppressed by the softmax operation, further pushing them toward zero. This creates opportunities for _attention sparsity_ . 

To exploit the _attention sparsity_ for computation acceleration, the top- _𝑘_ prediction mechanism has been proposed [25]. Fig. 3 illustrates its workflow via a 1 × _𝑆_ attention example. Typically, it consists of three stages. Firstly, a _Pre-compute stage_ estimates the attention matrix with a low-overhead paradigm (e.g., 4 bit MSB). The _Top-k sort stage_ then selects the indices of the top- _𝑘_ highestscoring Keys for each Query. For example, in the estimated attention in Fig. 3, Keys [0, 3] are identified as top-2 candidates for the current Query. Finally, the indices [0, 3] are transferred to _Formal compute stage_ , which performs full-precision QK _[𝑇]_ (8bit), softmax (FP16) and PV (8bit), using only these selected Keys and Values (i.e., [0, 3]). The top- _𝑘_ mechanism has been widely adopted in recent accelerators [72, 92, 94, 104] to improve attention efficiency. 

## **2.3 Opportunity and Observation at Bit-level** 

Fig. 4 (a) shows that value-level representation obscures bit-level optimization opportunities. In the 2-bit value matrix, only six elements are zero, and no column vectors are repeated (Repeated column vectors can be used to accelerate GEMM). This is due to bit concatenation, where a _𝑘_ -bit zero requires all _𝑘_ bits to be zero simultaneously. In contrast, decomposing the matrix into two 1-bit slices (MSB and LSB) reveals enhanced sparsity and redundancy. The MSB slice exhibits 14 zeros, yielding a 70% sparsity rate (14/20). 

1594 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Wang et al. 

**==> picture [239 x 192] intentionally omitted <==**

**----- Start of picture text -----**<br>
A 2-bit value-level  High-order (MSB) bit   Low-order (LSB) bit<br>weight matrix slice matrix slice matrix<br>00 11 00 00 11 x0 0 1 0 0 1 x0 0 1 0 0 1 x0<br>000011 010110 010011 000111 101110 x xxxx1234 = 2 [1 ] x  000 010 010 000 101 x xxxx1234 + 2 [0 ] x  011 110 011 111 110 x xxxx1234<br>0  repetitive column vectors   6  zeros values 14  zeros values 4  repetitive column vectors<br>(a) Value decomposition Bit-slice Sparsity Bit-slice Repetitiveness<br>LSB matrix 9 ADDs<br>YYYYY01234 = 0011 1110 0011 0111 1110 x xxxxx01234 Y0 = 0 + x1 + 0 + 0 + x4Y1 = 0 + x1 + 0 + x3 + x4Y3 = x0 + 0 + x2 + x3 + 0Y2 = x0 + x1 + x2 + x3 + x4 Y0 & Y1 & Y2 Vector Inner p Y2 & Y3 roducts Repetitive additions x0+x2+x3x1+x4<br>(b) Separate computation among BS vectors  30% more operations<br>LSB matrix (W) X Enumeration  Index matrix  X E X`<br>matrix (E) (I)<br>x = x x = x<br>2 ADDs 4 ADDs<br>(c) Principle for transferring repetitive vectors into computation reduction<br>**----- End of picture text -----**<br>


**Figure 4: Bit-level sparsity and repetition opportunities.** 

This aligns with the near-Gaussian distribution of weights [35, 52], where higher-order bits tend to be zero. Additionally, the 1st and 2nd columns in the LSB slice are identical to the 3rd and 5th, respectively, indicating increased repetition after bit-level decomposition. Notably, a 2-bit integer GEMV is functionally equivalent to a shiftand-accumulate operation over the two bit-slice matrices, where the MSB slice is weighted by 2[1] and the LSB by 2[0] . This demonstrates that bit-level decomposition preserves full compute equivalence while exposing fine-grained sparsity and redundancy. We refer to the above two opportunities as **BS sparsity** and **BS repetitiveness** . 

However, directly computing with BS sparsity or repetitive vectors in a naive manner is still inefficient. With the LSB slice matrix in Fig. 4 (a) as an example, Fig. 4 (b) illustrates computing each BS vector independently. This results in redundant operations. Specifically, _𝑥_ 1 + _𝑥_ 4 is calculated three times across _𝑌_ 0, _𝑌_ 1 and _𝑌_ 2, while _𝑥_ 0 + _𝑥_ 2 + _𝑥_ 3 is recalculated twice, leading to a 30% more operations. This inefficiency arises from failing to exploit redundancy across BS vectors. Naturally, this raises a key question: how can we harness such inherent repetitiveness to reduce overall computation? 

**Opportunity** : Fig. 4 (c) illustrates an effective computation reduction strategy. First, it transforms the LSB matrix (denoted as W) involving repetitive column vectors into an _enumeration matrix_ (E) and an _index matrix_ (I). Specifically, E stores unique column vectors from W and I records the mapping between each column in W and its corresponding vector in E. For example, the 3rd column of W matches the 1st column of E, so the value of I (1,3) is 1. This transformation rewrites W × X as E × I × X. The intermediate result X[′] = I × X requires 2 additions, and E × X[′] requires 4 additions, yielding a 30% reduction compared to the naive computation that demands 9 additions. We refer to this approach as a redundancy elimination strategy based on BS vector grouping. By grouping multiple BS vectors as a group matrix, the strategy identifies and eliminates redundant computations among them. Its effectiveness depends on the **repeated column vectors** within the group matrix: more repetition leads to lower computation. 

**Challenges** : Despite its promise, designing an efficient bit-level accelerator for LLM inference remains a challenging task. 

**(Challenge 1)** Directly grouping a large number of BS vectors results in low repetition rates among column vectors. 

As depicted in Fig. 5 (a), the number of repetitive column vectors in an 8 × 8 BS matrix is significantly smaller than the number of repetitive column vectors after decomposing it into two 4 × 8 submatrices, where we denote 4 as the group size _𝑚_ . This follows from the pigeonhole principle [1]: When the number of holes is less than the number of pigeons, at least one hole will contain more than one pigeon. As _𝑚_ decreases, the number of available "holes" (i.e., at most 2 _[𝑚]_ ) is reduced, thus the probability of repetitive column vectors increases. Our analysis across five LLMs in Fig. 5(b) reveals that, compared to the vanilla full-size merge, the group-wise merge achieves, on average, a 5 _._ 1× reduction in computation. 

**Key idea.** We decompose the large weight matrix ∈ R _[𝐻]_[×] _[𝐻]_ in LLMs into several smaller group matrices of size R _[𝑚]_[×] _[𝐻]_ , where _𝐻_ is hidden dimension. To support this, we design a CAM-based match unit, which significantly reduces the latency associated with the search process for repetitive column vectors along the H dimension. 

**(Challenge 2)** Value-level compression is incompatible with bit-level computation paradigms and obscures BS sparsity. 

As shown in Fig. 5 (c), this inefficiency stems from two primary factors: (1) Value-level compression achieves only a 30% sparsity rate (SR), which is 2 _._ 5× lower than bit-level sparsity; (2) Valuelevel formats require bit reordering for bit-level PEs, which incurs additional on-chip overhead. Fig. 5 (d) further show that, across five LLMs, bit sparsity is on average 10 _._ 1× higher than value sparsity. 

**Illustration for the bit reorder** : Traditional memory layouts store multi-bit activations contiguously across bits. As a result, when computing the MSB slice, a large number of LSBs are also fetched unnecessarily. To enable bit-slice-based processing, a bitreordering step is required to extract and reorganize the relevant MSB data into a contiguous MSB slice for input to the processing elements (PEs). We refer to this operation as bit-reorder. 

**Illustration for the bit sparsity** : For a _𝑘_ -bit matrix, we first compute the bit sparsity of the bit-slice matrices for each bit position. The overall bit sparsity of the 8-bit matrix is then the average bit sparsity across all bit-slice matrices for each bit position. 

**Key idea.** We propose an effective two-state coding scheme operating along the bit-slice dimension, naturally aligned with bitlevel computation and eliminating the overhead of data reordering. The coding and computation are associated designed and operates at the same group granularity to ensure global maximum benefits. Lightweight en/decoders are designed to enable greater parallelism and low-power data coding within the same area budget. 

**(Challenge 3)** The current Top- _𝑘_ prediction is coarse-grained and involves redundant computation and memory access. 

As depicted in Fig.5 (f), although the top- _𝑘_ prediction successfully reduces the overall attention latency by 45%, the bottleneck shifts to the prediction process itself. Therefore, it is imperative to 

1595 

MCBP: A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

**==> picture [507 x 142] intentionally omitted <==**

**----- Start of picture text -----**<br>
Optimization for computation (Sec 3.1 BRCR)  Optimization for weight access (Sec 3.2 BSTC) Optimization for KV cache access (Sec 3.3 BGPP)<br>❶ Vanilla scheme: merge  ❷ Ours: Partition vectors then  ❶ Traditional value-level  ❷ Ours: bit-dimension structured  ❶ Current value-level  ❷ Ours: Bit-gained progressive<br>full-size repetitive vectors merge repetitive ones by group compression two-state coding top-k prediction prediction<br>8 01000100 00010001 01001000 00100100 8 00010000 00100100 00101000 10000001 01000100 00010001 01001000 01000010 8 00010000 01000010 00101000 00011000 ☹ 11 Low Sparsity Ratio (30%) 111101 reorder 1110 Bit  10101110 0000001111 00000111 0000Bit PE0Bit PE10011 Bit-slicingreorderw/o Bit  ☺ High Sparsity Ratio (75%) 110111 1010 Bit PE0Bit PE10000 0000 0000 Generated Results q 0 4bit parallel multiplier 00 001 111 110 01 Load from     DRAM 1 k 000 ☹☹ 11  Computation  Load of all-bit Kfor all bits  00CMP11 Bit-serial multiplier q termination,  0 ① ( ≤ 0   Early  0  31)  100 01 1 101 11<< k 1 000 ③② 11  Eliminate  Eliminate  0 Not loadcompload 01CMP1<br>① Low repetition opportunity  ① High repetition opportunity  01 0 Bit PE2 0 Bit PE2 The  k -th value: 75  The  k -th value: 75<br>② Huge search complexity  (a) ② Low search complexity 10 1 Bit PE3 (c) 1 Bit PE3 (e)<br>86 Vanilla full-size bit-level merge(b) Comparison of comp reduction effectGroup-wise bit-level merge15.85 (d)1510 Value sparsity vs Bit sparsity Value Sparsity Bit-Sparsity(mean for all BS matrix)10.08 (f)1.51 Latency breakdown for llama7B attentionFormal ComputingPrediction 4 Vanilla Top-k(g) Memory access reductionOurs Theoretically optimal2.9<br>4 45% 56% 2<br>2 5.1x 5 10.1x 0.5<br>0 OPT1B3 Blooom1B7 Qwen7B Llama7B Llama13B Mean 0 OPT1B3 Bloom1B7 Qwen7B Llama7B Llama13B Mean 0 Dense Attention Top-k attentionTop-k a tention 0Llama7Bcola Llama7Bdolly Llama13Bdolly Mean<br>Group size<br>49.6%<br>KV cache access<br>)(4 m<br>Comp Reduction<br>Sparsity Ratio Norm latency<br>**----- End of picture text -----**<br>


**Figure 5: Challenges and our strategies for applying bit-level computing to computation-memory-efficient LLM inference.** 

**==> picture [239 x 72] intentionally omitted <==**

**----- Start of picture text -----**<br>
Pre-deployment Preparation  User Inference ( Online )<br>( Offline ) Minimized Mem. access<br>Models:  Bloom,  LLM Model Configuration BL K cache BL Sparsity Pred.<br>Llama7B... VL query BGPP (3.3)<br>BL encoding: BSTC (3.2) VL results Vital Key Indices<br>Sparse  BL GEMM Acceler.<br>Bloom ... models:  Memory BL weightVL input BRCR (3.1)<br>(BL: bit-level;  VL: value-level) Minimized Mem. access Reduced Comps<br>weights<br>HBM<br>Pre-compressed<br>**----- End of picture text -----**<br>


**Figure 6: The preparation and execution flow of MCBP.** 

further optimize the top- _𝑘_ prediction process. Fig.5 (e) illustrates inefficiencies in value-based top- _𝑘_ prediction using an example where the threshold is 75. To identify whether the current Key (0101) belongs to the top- _𝑘_ set, the value-based approach loads the 4bit K entry from HBM and then executes computation for 8bit results. However, we observe that the top 2 bits alone are sufficient to determine that the final result (≤31) will fall below the threshold, making the remaining 2-bit computation and memory access unnecessary. 

**Key idea** . We propose a bit-grained progressive prediction with early termination. Attention scores are estimated bit-wise from MSB to LSB. This allows computation and KV cache access to be terminated early once the partial result exceeds the feasible top- _𝑘_ range. As shown in Fig. 5 (g), this reduces KV cache accesses by up to 50% across three scenarios, compared to value-level prediction. 

**Review of Transformer accelerators** . Unfortunately, current Transformer accelerators still struggle with computation and memory access issues, due to their inability to exploit bit-grained opportunities for coordinated optimization. Table 1 summarizes their features. The majority of existing works [25, 26, 59, 75, 104] focus on accelerating attention whose quadratic complexity dominates earlier encoder-based models, like BERT. However, their strategies are less effective for decoder-only LLMs during the autoregressive decoding stage, where performance is severely constrained by memory access. While Energon [113] and SpAtten [94] realize challenges with memory access, their coarse-grained pruning fails to handle fine-grained bit-level optimizations for both linear weights and KV cache. SOFA [92] exhibits compute-memory co-optimization, but is restricted to attention. FACT [72] targets whole model computation reduction but lacks support for fine-grained KV cache and weight loading optimizations. These limitations motivate us to design an 

**Table 1: Summary for SOTA Transformer Accelerators.** 

|**Accelerato**|**r**<br>**GEMM**<br>QKV&FFN<br>Atten.|**Memory access**|**P**&**D**<br>**stage**|**Optimiz.**<br>**Level**|
|---|---|---|---|---|
|||Weight<br>KV Cache|||
|A3 [25]<br>**ELSA**[26]|×<br>✓<br>×<br>✓|×<br>×<br>×<br>×|P only<br>P only|Value<br>Value|
|<br>**Sanger[59]**|×<br>✓|×<br>×|<br>P only|Value|
|**DOTA**[75]|×<br>✓|×<br>×|P only|Value|
|**DTATrans[**|**104]**<br>×<br>✓|×<br>×|P only|Value|
|**Energon**[1|13]<br>×<br>✓|×<br>Low|P only|Value|
|**SpAtten[94**|**]**<br>✓<br>✓|×<br>Low|P&D|Value|
|**SOFA**[92]|×<br>✓|✓<br>×|P only|Value|
|**FACT[72]**|✓<br>✓<br><br>|Low<br>×<br><br>|P only|Value|
|**MCBP**|�<br>�|�<br>�|**P**&**D**|**Bit**|



efficient LLM inference accelerator that jointly optimizes GEMM computation, weight access, and KV cache access across both prefill and decoding stages. 

## **3 Algorithm Optimizations of MCBP** 

Based on the three challenges, we propose three corresponding optimization strategies: BRCR, BSTC, and BGPP. Fig. 6 depicts the overall execution flow of MCBP. Model weights are offlinecompressed into a bit-level (BL) sparsity format (BSTC, §3.2). During inference, the BL-compressed weights are loaded and decompressed, then sent for GEMM acceleration (BRCR, §3.1), the BL KV cache are on-demand fetched to predict attention sparsity (BGPP, §3.3). 

## **3.1 BS-Repetitiveness-enabled Computation Reduction for GEMM (BRCR)** 

As depicted in Fig.7 (a), the core idea of BRCR is first to decompose an _𝑘_ -bit weight matrix into _𝑘_ bit-slice (BS) matrices. Then, for each BS matrix, it extracts _𝑚_ rows of these matrices and merges them as a _Group matrix_ . Thus, it will process _𝑚_ rows each time, instead of all rows. For clarity, we use GEMV to illustrate the acceleration mechanism, which is also effective in GEMM scenarios. Overall, two key steps are required to achieve computation acceleration. 

**1) Merging repetitive operations** . As depicted by Fig. 7 (b), this step first ① identifies repeated entries (i.e., column vectors) in the _Group matrix_ G, then ② merge their corresponding activations 

1596 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Wang et al. 

**==> picture [506 x 284] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Initial: Group and Split weights along bit-dimension (b) G X Step 1: Merging repetitive operations Step 2: Computation reconstruction (c)<br>① Identify  000 x0 Complexity: H*(1-bs) additions  Complexity: m x 2 [m-1 ] value-level additions<br>H 0000001 000000000000 Hidden dimension 1011101 011011011011 0010100 100100100100 0010100 010010010010 0010000 010010010010 0100011 011011011011 0001000 100100100100 1010110 ( 011011011011 H ..................... ) m 000 Group matrix (G) 011 In bit format 100 010 H 010 011 100 011 Activation (X) ......... value format × xxxxxxxx01234567 H repetitive entries H 01110000011100011101111001... 1010 xxxxxxxxxx...10123456789 ②  Add xactivationsG i Merge = j  (0-7) i  to z j , if  x1 Merged activation vector   + xxx32xx + xxxx+x10110895 + x64 (MAV) 7 = zzzzzzzz01234567 2 [m] m Outputs yyy012 For one row, a half of data is = Enumeration matrix 0 0 0 0 0 0 0  0, without add  2 1 [m ]  0  1 1 (=8 if  1  0 0  0  1 1 1 11 m  0  Z =3) 1 1i 1 [×] MAV zzzzzzzz01234567<br>Weight matrix Group matrix Activation Z<br>Figure 7: Bit-slice-repetitiveness-enabled computation for GEMM (BRCR).<br>Group matrix in sign-magnitude (SM) Format Save original 8 [th] ,2 [nd] ,1 [st ] BS matrix  by BRCR is  𝑘 ( 𝐻 ×(1−1−− 𝑏𝑠 [[˜]] )+ 𝑚 ×22 [[𝑚]] [[−][1]][[1]] ). By contrast, existing sparsity-. By contrast, existing sparsity-<br>Bit width without compression<br>-5-324 1010 00010000 000101000000 0100011010010000 0110001000000000 1010111001100110 0000111010101001 0000111000101010 000010000000 00100000 1101 m 00000 00000 Indicators 00000 10001 + 00000 10001 00000 0 Compressed BS  0 0 matrix 10001 0 10001 0 𝑏𝑠 vs))∼×0.07,∼×0.07,×0.07,0.07, 𝐻 aware bit-serial computing (BSC) [reduction compared to value sparsity and naive BSC.˜reduction compared to value sparsity and naive BSC.˜ ))∼×0.07,∼×0.07,×0.07,0.07,  𝑚 additions.××  𝑚 𝑘 =4), BRCR achieves up to 12×  𝑣𝑠 additions. For typical LLM models (HAnd∼4k,×consumes4k,×consumes the value-based2, 15sparsity] consumes2, 15sparsity] consumes, 15sparsity] consumes 15sparsity] consumessparsity] consumes] consumesscheme . 1× and 3scheme× and 3scheme and 3scheme . 8×( computation( computation computation  𝑘 ×( computation( computation computation 𝐻 ∼4k,×consumes4k,×consumes 𝑚𝑏𝑠𝑏𝑠 [[˜]] × (∼0.70,1 −∼0.70,1 −0.70,1 −<br>8 [th] ... 2 [nd] 1 [st] 7 [th ] BS matrix  Off-line pre-<br>(sign bit)  (High SR) compression Verify the existence for redundancy based on pigeonhole<br>(a) Illustration for proposed bit-sparsity two-state coding (BSTC)  principle . Any  𝑚 -row binary matrix can have at most 2 [[𝑚]] types<br>(b) CR with group size (m) & sparsity ratio (SR) 54 SR=0.95SR=0.75 SR=0.65SR=0.9 SR=0.85 120100 (c) SR in Llama7B and Qwen7B with SM formatLlama7B Qwen7B CoTwo-state ding gain>1 of column vectors. Since LLMs (e.g., Bloom-7B, GPT-3) have hid-den dimensionsden dimensions  𝐻 (4k–12k) far exceeding 2 [[𝑚]] , there are abundant<br>32 806040 opportunities for redundancy in LLMs.<br>1 20 Key Insights: There is a key sweetspot of  𝑚 that achieves<br>0 0<br>1 2 3 4 5 6 7 8 9 10 8 [th] 7 [th] 6 [th] 5 [th] 4 [th] 3 [rd] 2 [nd] 1 [st] 8 [th] 7 [th] 6 [th] 5 [th] 4 [th] 3 [rd] 2 [nd] 1 [st] the maximum computation reduction while minimizing re-<br>Group size (m) Bit slice (position) Bit slice (position)<br>Group   (m) size<br>Group size m Dedicated  encoder<br>SR(%)<br>Compress. ratio (CR)<br>... ...<br>**----- End of picture text -----**<br>


by BRCR is _𝑘_ ( _𝐻_ ×(1−1−− _𝑏𝑠_[[˜]] )+ _𝑚_ ×22 _[[𝑚]]_[[−][1]][[1]] ). By contrast, existing sparsity-. By contrast, existing sparsity- _𝑏𝑠_ ˜ And the value-based2, 15sparsity] consumes2, 15sparsity] consumes, 15sparsity] consumes 15sparsity] consumessparsity] consumes] consumesscheme _𝑘 𝐻_ ×consumesconsumes1 − − _𝐻 𝑚_ additions.×× _𝑘_ × _𝑣𝑠_ additions. For typical LLM models (HAnd∼4k,×consumes4k,×consumes _𝑚𝑏𝑠𝑏𝑠_[[˜]] × (∼0.70,1 −∼0.70,1 −0.70,1 − vs))∼×0.07,∼×0.07,×0.07,0.07, _𝑚 𝑘_ =4), BRCR achieves up to 12× _._ 1× and 3scheme× and 3scheme and 3scheme _._ 8×( computation( computation computation aware bit-serial computing (BSC) [reduction compared to value sparsity and naive BSC.˜reduction compared to value sparsity and naive BSC.˜ 

**Verify the existence for redundancy based on pigeonhole principle** . Any _𝑚_ -row binary matrix can have at most 2 _[[𝑚]]_ types of column vectors. Since LLMs (e.g., Bloom-7B, GPT-3) have hid-den dimensionsden dimensions _𝐻_ (4k–12k) far exceeding 2 _[[𝑚]]_ , there are abundant opportunities for redundancy in LLMs. 

**Key Insights: There is a key sweetspot of** _𝑚_ **that achieves the maximum computation reduction while minimizing reconstruction overhead.** For a GEMV with a _𝑘_ -bit weight matrix ∈ R _[𝐻]_[×] _[𝐻]_ , the total operations of BRCR are _𝑘𝐻_[2] / _𝑚_ × (1 − _𝑏𝑠_[˜] ) + _𝑘𝐻_ 2 _[𝑚]_[−][1] . The group size _𝑚_ introduces an interesting trade-off. If _𝑚_ is too small, it fails to exploit sufficient redundancy between the bit-slice vectors. Conversely, if _𝑚_ is too large, the exponentially increasing reconstruction cost (i.e., 2 _[𝑚]_[−][1] ) offsets the benefits of redundancy removal. The DSE for optimal _𝑚_ is provided in §5.2. 

**Figure 8: BS-sparsity-enabled two-state coding (BSTC).** 

into a _merged activation vector_ (MAV), denoted as Z. This is implemented by accumulating each activation into the partial sum of the corresponding entry in Z, based on the value of each column in G (We denote as Grouped index). For example, the 3rd and 4th columns of the group matrix are both 010 (i.e. _𝐺_ 3= _𝐺_ 4=2), so their corresponding activations, _𝑥_ 3 and _𝑥_ 4 are added to the entry ( _𝑧_ 2) of the Z. Notably, for a bit column vector with _𝑚_ elements, there are 2 _[𝑚]_ possible types. Thus, the MAV has a length of 2 _[𝑚]_ . Mathematically, this step is equivalent to the I × X in Fig. 4 (c). Notably, non-zero entries in the MAV indicate multiple rows in a weight share the same addition operation. For instance, _𝑧_ 3 (Grouped index is 011) denotes the repetitive additions among rows 1 and 2, while _𝑧_ 0 represents activations multiplied by zero, which can be directly eliminated. With bit sparsity ratio _𝑏𝑠_ , this step consumes at most _𝐻_ × (1 − _𝑏𝑠_ ) additions, regardless of group size _𝑚_ . 

## **3.2 BS-Sparsity-enabled two-state Coding (BSTC)** 

While numerous studies [28, 29, 62, 63, 73] have explored coding techniques for sparse weight compression, they largely focus on value-level sparsity, limiting their effectiveness. In contrast, BSTC exploits the key insight that quantized weights exhibit Gaussianlike distribution [52], thus most non-zero weights own zero bits. To this end, BSTC encodes data at different BS matrices separately, to exploit the high sparsity in high-order bit plane. In addition, the encoding of BS matrices aligns with the computation granularity of BRCR, i.e., group size _𝑚_ , thus avoiding extra data conversion overhead. 

**2) Computation reconstruction** . As depicted in Fig.7 (c), this step is to reconstruct the GEMV results by multiplying the _Enumeration matrix_ with the MAV. It is noteworthy that for a Group matrix R _[𝑚]_[×] _[𝐻]_ , when _𝐻_ is very large, we can reasonably assume that all possible 2 _[𝑚]_ column vectors will appear. Thus, the Enumeration matrix contain all 2 _[𝑚]_ distinct column vectors. In this way, each row of the enumeration matrix can contain at most 2 _[𝑚]_[−][1] ones. Therefore, the computation reconstruction step requires at most _𝑚_ × 2 _[𝑚]_[−][1] additions for reconstructing _𝑚_ -row GEMV. 

Fig. 8 (a) illustrates BSTC’s design. To exploit bit-0 sparsity in high-order bits (near MSB part), we adopt the sign-magnitude (SM) format for all weights. Given varying sparsity across bit positions, only bit-slice matrices from bits 3-7 are compressed, while bits 1, 2, and 8 remain uncompressed. Despite redundant sparsity in high-order bits, naively encoding would result in irregular data re-assignment for computation, leading to severe overhead. To this end, we employ a _two-state_ encoding, which distinguishes only zero data and non-zero data. Zero is encoded as 1’b0, and non-zero is encoded as a ( _𝑚_ + 1)-b symbol: {1[′] _𝑏_ 1 _,𝑚_[′] _𝑏_ data}. For instance, in Fig. 8 (a), we have {0000}→{0} and {0001}→{10001}, where 1 is 

_𝑏𝑠_ ˜ Inand value sparsitysummary, for a _𝑣𝑠 𝑘_ -bit,, where _𝑚_ -row _𝑏𝑠_ ˜ is the average bit sparsity ratioGEMV with bit sparsity ratio across all (∈[1 _,𝑘_ ]) bit-slice matrices. The total additions required 

1597 

MCBP: A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness 

**==> picture [228 x 109] intentionally omitted <==**

**----- Start of picture text -----**<br>
The first round The second round<br>Load 1st bit (K) from HBM Load 2nd bit (K) from HBM<br>K K0 K1 K2 K3 K4 K5 K K1 K3 K5<br>(1 bit) (1 bit)<br><< <<<br>(4 bit) + (4 bit) +<br>Psum of 1 [st]<br>round …<br>RS Filter: Eq(1) RS Filter: Eq(1)<br>0 1 2 3 4 5 1 3 5<br>**----- End of picture text -----**<br>


**Figure 9: Bit-grained progressive top-** _𝑘_ **prediction (BGPP).** 

**==> picture [242 x 169] intentionally omitted <==**

**----- Start of picture text -----**<br>
Ctrl ❶ Efficient pipeline for MCBP`s Transformer workflow<br>BSTC Decode WQK Decode WV ❷ ❸❹❺ Decode WFFN<br>BRCR QK generation V generation Sparse Atten FFN Comp<br>BGPP QK Prediction<br>❻ ❼ ❽<br>Weight SRAM (768KB) Token SRAM (384KB) Temp SRAM (96KB)<br>❽<br>Bit-serial & Scheduling  CAM-based Bit-grained BRCR Unit<br>Computation Ctlr<br>❺ ❶ ❹ Adds  PE 0PE 7 Inter<br> Fetcher & DispatcherX, W, on demand KV ❸ CAM mergeAdd Merge reconst.Comp  Accu.stage<br>❷ ❼<br>Lightweight Parallel BSTC Clock-gated BGPP Unit<br> Encoder & Decoder ❻ << <<<br>Shift Adder<br>0 SIPO Unit<br>CMP CMP LUT << <<<br>APU Embedding Unit Special Function Unit Quantizer<br>External HBM<br>Sign decision Thre. update Clipping unit<br>**----- End of picture text -----**<br>


**Figure 10: High-level block diagram for MCBP accelerator.** 

an indicator that facilitates decoding. In this way, BSTC provides regularity at the bit-column level and achieves lossless compression. 

Since BSTC introduces a 1-bit indicator for each non-zero column vector, its applicability must be carefully evaluated; otherwise, the overhead may offset the encoding gains. Fig. 8 (b) illustrates the compression ratio (CR) of BSTC under varying sparsity ratio (SR) as the group size ( _𝑚_ ) changes. There are some interesting insights: First, an excessively large _𝑚_ may reduce the compression ratio due to fewer co-occurring zeros across data elements within larger groups. Second, when the SR is high, a larger group size _𝑚_ tends to yield a higher compression ratio, as it reduces the relative overhead of storing indicators. Last, we can figure that when SR exceeds 65%, BSTC can achieve positive benefits (i.e. CR>1). Further, Fig. 8 (c) analyzes the SR of bit-slice (BS) matrices across different bit positions in Llama7B and Qwen7B. It is observed that the SR for the 3rd to 7th BS matrices all exceed 65%. Thus, we apply BSTC compression to these BS matrices. By contrast, for BS matrices with low SR, such as the 1st BS matrix, no compression is applied 

## **3.3 Bit-grained Progressive Prediction (BGPP)** 

As introduced in §2.2, the core idea of top- _𝑘_ prediction is to estimate the attention matrix with a low-overhead paradigm, then pick up important Key indices. However, even utilizing the low-precision paradigm (e.g. 4bit with MSB only), the value-based strategy still 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

**==> picture [242 x 75] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) INT8 GEMM in MCBP (b) Theoretical Derivation*<br>ΔWΔX/ΔY  BRCR GEMM Bias 1 Yq =Yf/ΔY+ZY<br>Channel-wise 2     =(WfXf/ΔY+ZY)<br>Yq= Wq × + Xq + 34     =(Δ    =(ΔWWWΔXqWΔqX(XXqq-Δ-ZWX))/ΔΔXWYq+ZZXY)/ΔY+ZY<br>Yq= (ΔWΔX/ΔY) WqXq +ZY-ΔWΔXWq1ZX/ΔY 5     =(ΔWΔX/ΔY)WqXq+ZY-ΔWΔXWq1ZX/ΔY<br>Scale Bias 6     =Scale ⊙ (WqXq)+Bias<br>*For simplicity, we ignore the impact of dimension broadcast, so part of the result should multiply  1 .<br>**----- End of picture text -----**<br>


**Figure 11: Illustration for quantization process in MCBP.** 

**==> picture [242 x 63] intentionally omitted <==**

**----- Start of picture text -----**<br>
Bit slice Parallel across PEs Output tile Weight tile Input tile<br>1 2 MSBLSB 2 3 PE cluster #0 TN 3 For m=[0: M/TM )<br>3 = 3 1 × 1 2 TM TK 21 ForFor n=[0: k=[0: N/TK/TNK ))<br>TK BRCR-GEMM()<br>Output Weight Input<br>**----- End of picture text -----**<br>


**Figure 12: The tiling strategy for GEMM in MCBP.** 

causes unnecessary memory access and computation (Fig.5 (c)). Therefore, a more efficient prediction scheme is a must. 

BGPP addresses this by leveraging the relative nature of softmax: if an input’s gap from the current max exceeds a threshold, its softmax output will be near zero[72]. Thus, the gap (termed _radius_ ) with the current max value can be used to filter trivial Keys. 

We propose a bit-grained progressive filter mechanism to achieve this. _Progressive_ means: it performs multiple rounds of filtering, where in each round, incremental filtering is applied based on the Keys (Ks) selected in the previous round. Fig. 9 gives an illustration for this procedure. Assume the initial state consists of 6 Ks (K0K5). In the first round, we fetch the MSB of all Ks for computation with _𝑄𝑖_ (with 4 bit), and obtain the estimated Max attention value denoted as max( _𝐴_[ˆ] _𝑖_[1][). Then, based on Eq.][(1)][, a radius-calculated (RS)] filter obtains the filtering threshold for the current round. Then, it retains the indices ( _𝐾𝑖𝑑_ ) of the Ks (e.g. 1,3,5), whose attention values are greater than this threshold. In the next round, we only fetch the second bit of the {1 _,_ 3 _,_ 5}-th Ks from HBM. This process continues for the predetermined number of rounds. 

Instead of directly adopting a fixed value as the threshold, for round _𝑟_ , we set the filter threshold of the _𝑖_ -th row as _𝜃𝑖[𝑟]_[:] 

**==> picture [197 x 12] intentionally omitted <==**

where _𝐴_[ˆ] _[𝑟] 𝑖_[is the estimated attention of the] _[ 𝑖]_[-th row (During] _[ decoding]_ stage, _𝑖_ =0). Based on our experiments, we empirically set the default radius to 3 and use a parameter _𝛼𝑟_ ∈[0 _,_ 1] to control the threshold. By adjusting _𝛼𝑟_ , we can control the pruning ratio in each round. 

## **4 Architecture and Hardware Innovation 4.1 Architecture Overview** 

Fig. 10 illustrates the MCBP architecture (bottom) and its workflow (top) for efficient Transformer inference under attention sparsity (§2.2). The accelerator operates through eight key steps, with numbered markers on the timeline indicating their positions within the overall pipeline. First, the controller sends token indices and bit-slice (BS) weights to the data fetcher, which decodes physical addresses and loads data into on-chip SRAM ❶. The BSTC decoders 

1598 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Wang et al. 

**==> picture [242 x 197] intentionally omitted <==**

**----- Start of picture text -----**<br>
Bit-compressed matrix Data layout in HBM 1<br>0 0<br>H (hidden size) dim. 0 00 1<br>1 0<br>0 0 0 1 0 1 0 ... 0 01 0 ... [......... ......][ ...]<br>00 00 Prioritize interleaving along group size  0 Bank 0 0 HBM ... Row bufferRow bufferRow bufferRow buffer<br>0 0 dim. across HBM  Activation Row buffer<br>1 1 banks Row bufferareaRow buffer<br>Row buffer<br>0 [th] ,6 [th] ,7 [th]  BS<br>BSTS  Weight SRAM Input Act.  Output Act.<br>Dispacther Decoder 2*16*8 kB SRAM SRAM<br>Row 0-3 GBMC0 GBMC1 GBMC7 PE 0 <<<br>... ...<br>Row 4-7 GBM 0,0 GBM 0,1 GBM 0,7 PE 1 <<<br>... ...<br>GBM 1,0 GBM 1,1 GBM 1,7<br>Row 60-63<br>... ... PE 7 <<<br>GBM 15,0 GBM 15,1 GBM 15,7 PE Cluster<br>Bit slice Parallel<br>... ...<br>...<br>...<br>...<br>... ... ...<br>m (group size) dim.<br>Inter-PE Accumu.<br>64*1024 8-bit weight<br>**----- End of picture text -----**<br>


**Figure 13: The bit-grained computation dataflow of MCBP.** 

**==> picture [240 x 217] intentionally omitted <==**

**----- Start of picture text -----**<br>
Key (4bit) eg. 0001 Search  BS weight (m=4)Decompressed  Fetcher 16 Indices Activation Sram (2*8*24kB) 16 selected activations<br>16 16*5 16 PE 0PE 7<br>4 ❶ match (512B) CAM-based  512 32 bit 1001.. ❷ Index ConverterIndex Converter Index Convert Merge Unit ❸ Addition  300 ❹ Recon. Unit +<br>❶ Step 1: Address Orchestration for decompressed weight  CAM-based Fast Match Unit Converter ❷  Index  0<br>HO LO Data0 ... Data3 Data0 ... Data3 Position One<br>Data0 0 0 0 1 00 1 0 1 1 00 0 0 0 0 1001.. Detector 3 x64 6<br>Data1 1 0 0 1 01 0 0 0 0 01 1 1 0 1 32 bit 6<br>Data2 0 0 1 0 10 0 1 0 0 10 0 0 1 0<br>Data3 0 0 0 1 11 0 0 0 0 11 0 0 0 0 Selected 8 bit activation<br>Step 2: Search ❸ Addition Merge Unit<br>Data0 ... Data3 Data0 ... Data3 4*8 z1<br>Search 00 1 0 0 1 00 0 0 1 0<br>0001Key 01 10 0 0 0 1 1 0 0 0 0110 1 0 1 0 0 0 1 0 Data0 and 1001 20 x15<br>11 0 0 0 0 11 0 0 0 0 Data3 match 20 z15<br>Group Sum Buffer<br>❹ Reconstruction Unit z1 z2 [...] Group Sum Buffer (GSB) ...  z14 z15<br>yyy012 = zzz8 4 2 +z+z+ z 9 5   3 +z++z z 10  6   6  +z +z + z 11  77 +z+z+ z 12  10   12 +z +z +z13  13  11 +z+ +z z14 14   14    +z+z+z151515 Adder zzzz8421 0 zz z z95 3 3 z z zz10 6 65 Adder z zz z11 77 7 1 z zzz Adder 12 12109 2 z z z z 13 131111 zz zz 1414 1413 Adder zzz z 151515 15 3<br>y3 z1 +z3  +z5  +z7  +z9   +z11 +z13   +z15 y2 (cycle1)<br>y3 (cycle0)<br>Results<br>Search Key<br><<<br>...<br>...<br>...<br>**----- End of picture text -----**<br>


**Figure 14: A PE cluster of CAM-based fast-match BRCR unit.** 

then decompress the weight matrices ❷, forwarding them to the BRCR unit ❸, where a CAM-based module identifies repetitive BS weight entries.These indices are returned to the fetcher, translated into SRAM addresses, and used to fetch corresponding activations back to the BRCR unit ❹. Finally, the computed GEMM results are written back to off-chip DRAM ❺. 

To efficiently handle dynamic attention sparsity (§2.2) and hide prediction latency, BGPP operates concurrently with the BRCR unit. Its workflow begins by retrieving QK tensors from the _data fetcher_ ❻ and performs an initial prediction. The selected indices are then returned to the data fetcher ❼ to fetch the required Keys for the next round (as Fig. 9). This process continues iteratively until the preset number of iterations is reached, and the final KV indices are stored in Temp SRAM ❽. In this dedicated dataflow, once computation proceeds to the attention part, the BRCR unit can merely calculate attention scores with those vital KVs, based on these KV indices generated by BGPP. To fully support Transformer computation, MCBP integrates an Auxiliary Processing Unit (APU) that includes an embedding unit for generating input token embeddings via table lookup, a special function unit (SFU) implemented in FP16 using a combination of lookup tables and polynomial approximation [73] to compute non-linear functions such as GELU, softmax, and layer normalization, and a quantizer that handles data conversion between FP16 and INT8. Notably, the concatenation in MHA is performed during data movement. 

**Processing of scaling and zero point** . Fig. 11 (a) illustrates the quantization process in MCBP, where weights are quantized using per-channel symmetric quantization, and activations are quantized using per-tensor asymmetric quantization, as [38, 98, 101]. Taking activations as an example, the quantized input activation is computed as _𝑋𝑞_ = _𝑋𝑓_ /Δ _𝑥_ + _𝑍𝑥_ , where Δ _𝑥_ is the scaling factor and _𝑍𝑥_ is the zero-point offset. Notably, Δ _𝑤_ , Δ _𝑥_ , Δ _𝑦_ , _𝑍𝑥_ and _𝑍𝑦_ can be preknown by the calibration dataset. Based on the derivation in Fig. 11 (b), the final output is expressed as _𝑌𝑞_ = _𝑆𝑐𝑎𝑙𝑒_ ⊙( _𝑊𝑞𝑋𝑞_ ) + _𝐵𝑖𝑎𝑠_ , where the INT GEMM ( _𝑊𝑞𝑋𝑞_ ) is accelerated by the BRCR unit via 

efficient bit-slice processing and shift-accumulation. The results are then processed by the quantizer with the scaling and bias terms. 

**Tiling Strategy.** Fig. 12 illustrates the tiling strategy of MCBP and its corresponding loop representation for the output-stationary dataflow. To maximize weight reuse, MCBP stores slices included in the _𝑇𝑀_ × _𝐾_ weight tile into the weight SRAM at once, if possible. Then the BRCR unit assigns a _𝑇𝑀_ × _𝑇𝐾_ weight tile together with a _𝑇𝐾_ × _𝑇𝑁_ activation tile to each PE cluster, where 8 PEs concurrently process each bit-slice of the weight tile in parallel. In this work, we set _𝑇𝑀_ = 64, _𝑇𝐾_ = 256, and _𝑇𝑁_ = 32. 

## **4.2 Bit Dataflow of MCBP** 

To optimize bit-level memory access, we orchestrate the BS weight matrix layout in off-chip HBM. Fig. 13 shows an example using the compressed weight from Fig. 8. Given HBM’s read-write characteristics, we prioritize storing the bits along the group size dimension at the same address across all banks. Once filled, allocation moves sequentially to the next address until the entire BS matrix is stored. Before computation, the BS data is loaded into the on-chip weight SRAM. Given the one-row-per-cycle access feature of SRAM banks, we prioritize storing the BS matrix within a single bank. Once the data in the weight SRAM is ready, the sparse BS matrices are sent to the BSTC decoder for decompression. During the process, the controller will check whether decoding is required. BS matrices at the 1st, 2nd, and 8th bit positions are not decoded, as they were not encoded due to their low sparsity (see Fig. 8). 

## **4.3 CAM-based BRCR Unit** 

The BRCR mechanism requires quickly identifying and consolidating identical elements in the group matrix (§3.1). To this end, we design a content-addressable memory (CAM) based fast match unit, which can identify identical elements in one cycle. 

1599 

MCBP: A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

**==> picture [241 x 102] intentionally omitted <==**

**----- Start of picture text -----**<br>
BSTC Encoder 0 (a) (c) 16bit SRAM Bank 0SRAM Bank 1<br>4 0000 441 CMP 15 1 10 1/5 Starting address varied-length Compressed sub weight  for each  0x00x4 ❶ 003100110 3 0 [rd] 10001  row, 0-th column 0 0 0564b...0 04210100... ......<br>weight<br>0 0 0 ...<br>(b) SIPO 5 Lead one  Row-wise readout 0 0 0 10001 ❷ … 0 10011 0<br>1 01 1 CMPEn BSTC Decoder eliminator1 0000 10 4 0000, 0000, 0000, 0001,  m =4 Sub weight 0 ❸ 1 k … BSTC decoder, 0011, 0000 Decompressed weight matrix<br>**----- End of picture text -----**<br>


**Figure 15: (a)(b) Architectures for lightweight BSTC encoder/decoder. (c) Parallel-friendly segmented data layout.** 

As depicted in Fig. 14 ❶, we adopt a group size _𝑚_ = 4. Initially, each 4-bit column vector of the decoded weights is orchestrated in CAM. Higher-order (HO) two bits and lower-order (LO) two bits of each 4-bit data are managed separately. As two bits correspond to four possible values, four entries are needed to store the orchestrated data. Then, for the search step, if an entry address matches the search key, the content of that entry is set to 1, while the other entries are set to 0. Taking searching 0001 (search key) as an example, the HO two bits read the row at address ‘00’ of the MSB bank, while the LO two bits read the row at address ‘01’ of the LSB bank. Then readout bits from both banks are ANDed to match both high and low 2 bits with the search key. The generated bitmap ‘1001’, indicates _𝑥_ 0 and _𝑥_ 3 match the 0001. The controller enumerates all possible search keys for _𝑚_ =4 (0000 to 1111). If the search key is 4’b0000, the CAM will be clock-gated to save power. The CAM, with a 2-bit matching length as its basic block, is designed to be reconfigured by re-matching the outputs of multiple basic blocks, to support adaptation to different group sizes. 

The CAM-generated bitmap identifies activations to be merged and added. Sixteen index converters then translate the bitmap (e.g., 1001) into corresponding activation indices (❷). Next, the search key (0001) and the fetched activations ( _𝑥_ 0, _𝑥_ 3) are together sent to the addition merge unit (AMUs) (❸). The AMU first adds _𝑥_ 0 + _𝑥_ 3, then put the psum to the first register _𝑧_ 1, based on the search key 0001. For more fetched activations, data in one register is read out by the MUX and added to the psum, then the result is written back to the same register in group sum buffer (GSB) by the deMUX. 

Next, a reconstruction unit (RU) reorganizes partial sums stored in GSB into correct results. Inspired by the fixed re-construct formula as Fig. 14 (❹) left, we design a low-power RU with a fixed data path. _Fixed_ means: we bind specific registers to inputs of each adder. By reordering the computation sequence, we extend the data lifecycle in adders. For example, computing _𝑦_ 3 first, followed by _𝑦_ 2 down to _𝑦_ 0, allows _Adder 3_ to read _𝑧_ 15 only once, reducing its switching activity by up to 75%. Given that the reconstruction workload is much lighter than the addition merging, one RU is time-multiplexed to serve 16 AMUs, improving resource utilization. 

## **4.4 Lightweight BSTC CODEC and Data Layout** 

We first design a lightweight encoder-decoder that enhances parallelism within the same area budget. Then, we introduce a segmented interleaved data layout in SRAM to support parallel en/decoding. 

**==> picture [239 x 99] intentionally omitted <==**

**----- Start of picture text -----**<br>
Bit-serial based IP Unit Clock-gated Progressive Filter<br><<br>1 min A<br>8 Clipping<br>x64 Threshold Updating module<br>1 + max<br>1 8 > A[0] A[1] A[3] A[63]<br>MSB? > > > ... ><br>Sign decision unit 0 1 MSB? <<1 Vital K indices: [0,2,6,1 0 1 0 0 0 1 1 0 1 0 ...1 0 1 1 … 60,62,63]<br>psum<br>(bit)W0<br>X0<br>Neg Vital K indices<br>... com<br>Adder  Tree<br>32-b<br>Neg<br>X63 Radius (3)<br>(bit)<br>63<br>W<br>**----- End of picture text -----**<br>


**Figure 16: Threshold-aware clock-gated BGPP unit.** 

Fig.15 (a)(b) depicts the lightweight BSTC en/decoder architectures. The encoder comprises a 4 bit comparator (CMP) and a MUX. If the input is non-zero, it adds a 1-bit ‘1’ ahead the MSB and outputs the result; Otherwise, it outputs a 1-bit ‘0’. The decoder includes a 1-bit CMP, a 5-bit serial-in parallel-out (SIPO) register (for _𝑚_ =4), and a leading one eliminator. When a ‘0’ is detected in the bit stream, it outputs four consecutive 0’s. Otherwise, it buffers received bits in the SIPO, which outputs the buffered content once full. 

Fig.15 (c) illustrates the segmented weight layout during a decompression process. To enable parallel decoding, the weight matrix is partitioned along the hidden dimension into multiple subweights, each stored independently in separate banks. Given variable compression ratios, the starting address of each sub-weight is recorded. Before decompression, the controller fetches these starting addresses from the address area ❶. Based on the retrieved addresses ❷, sub-weight data is accessed row-wise and sent to the BSTC decoder ❸ for decompression. Each bank has 64 columns and 1024 rows, we use a 6-bit column address and a 10-bit row address to locate the first data of each sub-matrix. One row can store the address of four 1k-length sub-matrices, and three rows suffice to index up to 12 sub-matrices, covering the weight size of most LLMs. 

## **4.5 Threshold-aware clock-gated BGPP Unit** 

Fig. 16 shows the architecture of BGPP unit. First, 16 bit-serial inner product units compute Q (1*64) × K (64*16), each with a 64-input adder tree. The generated summations are passed to the progressive filter (PF) unit. Next, a _threshold updating_ (TU) module serially identifies the Max and Min values from these data. After completing the statistics for one row of the estimated attention, the TU subtracts the _𝛼𝑟_ × _𝑟𝑎𝑑𝑖𝑢𝑠_ from the Max value, which stands for the filter threshold (Eq.(1)). Next, a _clipping_ module compares all attention entries with this threshold, then produces a binary mask signal, where “1" signifies the index of Ks eligible to proceed to the subsequent filtering round. After a fixed number of rounds, the final set of key indices is selected. To save power, if the threshold falls below the observed minimum, the clipping module is clock-gated and BGPP immediately proceeds to the next round. Additionally, to enable bit-wise computation under the SM format, we design a _sign decision unit (SDU)_ and place it before the adder tree. 

1600 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Wang et al. 

**Table 2: Accuracy of Different Language Models with FP16, INT8 and MCBP Optimization (S: Standard, A: Aggressive).** 

**==> picture [506 x 268] intentionally omitted <==**

**----- Start of picture text -----**<br>
Model LlaMa7B LlaMa13B OPT1B3 Bloom1B7 Qwen7B<br>Task [‡] MMLU Wikiling. MBPP Wiki2 Winogran. Cola MNLI SST2 MMLU Wikiling. MBPP Wiki2 Winogran. Cola MNLI SST2 Wikiling. MBPP Wikiling. MBPP Wikiling. MBPP<br>FP16 35.1% 39.3 17 . 8% 5.68 70.1% 80.3% 84.6% 92.5% 41.2% 43.3 22% 5.09 74.2% 82.5% 85.5% 93.8% 36.2 12% 44.3 16% 46.6 30%<br>INT8 34.7% 38.9 17.2% 5.73 69.3% 80.2% 84.4% 92.5% 40.9% 42.7 21.6% 5.13 73.7% 82.3% 85.3% 93.7% 35.9 11.6% 44.1 15.7% 46.4 29.2%<br>MCBP (S) 34.6% 38.8 17.1% 5.75 69.2% 80.2% 84.4% 92.4% 40.7% 42.6 21.5% 5.15 73.4% 82.3% 85.3% 93.7% 35.8 11.5% 44.0 15.6% 46.3 29.1%<br>MCBP (A) 34.1% 38.4 16.5% 5.80 68.7% 80.0% 84.1% 92.1% 40.2% 42.0 21.0% 5.19 72.8% 82.0% 85.1% 93.4% 35.3 11.0% 43.6 15.2% 45.9 28.4%<br>‡ MMLU, WinoGrande, MBPP, Cola, MNLI, SST2 are evaluated by accuracy. Wikitext2 is evaluated by perplexity, where lower is better. Wikilingua is evaluated by ROUGE-1, where higher is better.<br>1.2 1.2<br>1.0 (Norm Comp.) SOFA Spatten FACT Bitwave FuseKNA MCBP SOFA Spatten FACT Bitwave FuseKNA MCBP (Norm Mem. Access) 1.0<br>0.8 0.8<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2<br>0 0<br>Llama7B Llama13B OPT1B3 Bloom1B7 Qwen7B<br>Figure 17: Normalized computation ( prefill  stage) and memory access ( decoding stage) of LLM inference.<br>30 4 (a) Breakdown for the union effect of BRCR, BSTC, BGPP  (b) Separate effect of BRCR, BSTC, BGPP<br>25 Min comp reduce Max comp reduce Compression rate 1.2 Baseline +BRCR +BSTC +BGPP 1.2 Baseline BRCR BSTC BGPP<br>20 Optimal Balance 3 1 1<br>0.8 0.8<br>15 2 0.6 0.6<br>10 1 0.4 0.4<br>5 0.2 0.2<br>0 1 2 3 4 5 6 7 8 9 0 0 0 0 1k 4k 1k 4k<br>Group size ( m ) # Prompt length # Decoding length<br>Dolly (decoding  MBPP (promot<br>Llama Llama OPT Bloom Qwen ~48 tokens) ~48 tokens)<br>7B 13B 1B3 1B7 7B Llama 7B (Batch size=8)<br>Norm computation Norm Mem. access<br>30%<br>25% 0.49<br>0.36 0.37<br>19% 0.26<br>Comp Reduction (CPR) Compression Rate (CR) Normalized latency<br>**----- End of picture text -----**<br>


**Figure 18: Design space exploration of the optimal group size** _𝑚_ **, for computation reduction and compression rate.** 

**Figure 19: Latency reduction for BRCR, BSTC and BGPP.** 

## **5 Evaluation** 

## **5.1 Experimental Setup** 

**Baseline comparisons** : We compare MCBP with two SOTA bit accelerators: FuseKNA [103], Bitwave [81], and three Transformer accelerators: Spatten [94], SOFA [92], FACT [72]. For fair comparison, FuseKNA and Bitwave are adapted from convolution to GEMV using im2col. All designs are normalized to a 28nm process and evaluated under identical conditions: PE arrays occupy the same area as MCBP and work in 1GHz, on-chip SRAM is set to 1248kB, and HBM bandwidth is fixed at 512-bit/cycle, with 4 pj/bit [67]. 

**Benchmarks** : We evaluate MCBP on several LLM models of varying sizes, including Llama7B/13B [88], Qwen7B [6], Bloom1B7 [42] and OPT1B3 [111], across nine tasks. These tasks includes Cola (S=0.25k), MNLI (S=0.5k), SST2 (S=0.25k) from GLUE [91], language modeling (Wikitext-2 (S=2k) [61], Wikilingua (S=2k) [18], Winogrande (S=0.25k) [77]), Multitask Language Understanding (MMLU, S=0.5k) [32], code generation MBPP (S=1k) [5], long context processing dolly (S=8k) [13]. 

**Quantization Accuracy** . All pre-trained models are sourced from Pytorch [69] and HuggingFace [99]. INT8 baselines derived via post-training quantization, where only the GEMMs are quantized to INT8, while non-linear operators (e.g., softmax) remain in FP16 precision. As shown in Table. 2, **the INT8 baseline incurs less than a** 1% **average accuracy drop from FP16, confirming its validity** . Notably, for reasoning tasks such as MMLU and Winogrande, the accuracy degradation caused by INT8 quantization 

is negligible, typically below 0 _._ 5%. This observation is consistent with prior works [36], which suggests that classification and reasoning tasks, due to their discrete output space and robustness to quantization noise, exhibit a high tolerance for low precision. 

**Simulation** : We implement the RTL design for MCBP and utilize Synopsys DC on TSMC 28nm CMOS technology to estimate the logic area and power. The CAM cell is designed using Cadence Virtuoso at the schematic level, then integrated with Verilog-based digital peripherals. The power, area, and read/write bandwidth of on-chip SRAM buffers are estimated through CACTI [64]. Off-chip HBM modeling involves simulating row activation and access patterns under various data layouts, capturing HBM’s burst behavior. We derive memory latency from Ramulator [41], and estimate IO power following the methodology in [3, 8, 96]. We extract each stage’s cycles by simulating the RTL with Verilator [83], and use a custom cycle-level simulator to evaluate end-to-end performance. 

**GPU comparison** We run benchmarks on Nvidia A100 with SOTA TensorRT-LLM [66]. To exclude the software overhead, we measure execution time with _cudaEvent_ , isolating GPU execution from CPU interference. The GPU is dedicated during tests, and large batch sizes are used to amortize data transfer costs. We employ _nvprof_ to exclude non-computational phases. Power is measured via _nvidia-smi_ ; dynamic power is computed as the difference between active and idle states. Each experiment is run 2k times, discarding the top and bottom 15% before averaging. 

1601 

MCBP: A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

## **5.2 Algorithm Performance** 

**Algorithm settings** : We regard INT8 models as the accuracy baseline, and adjust the value of _𝛼𝑟_ in 0.1 increments to evaluate the accuracy and overhead for each benchmark. This yields two MCBP configurations: standard (0% loss), aggressive (1% loss), representing the minimal and maximal performance optimizations, respectively. 

**Optimal Group Size** . We determine the optimal group size _𝑚_ by comparing computation reduction (CPR) and compression rate (CR) against dense models. Considering the varying sparsity levels, both the max and min CPRs are reported. Fig. 18 shows that CPR raises from _𝑚_ =1 to _𝑚_ =5, as more weight rows are merged, but declines beyond _𝑚_ =5, due to the exponential growth (2 _[𝑚]_ , Fig. 7) in additions required by computation reconstruction. For CR, _𝑚_ =1 results in a CR of less than 1, while _𝑚_ =4 maximizes CR by capturing all-zero columns. Beyond this point, fewer all-zero columns, in turn, negatively impact the CR. Considering the balance CPR and CR, and that 4 is the common divisor of most Transformer hidden dimensions, we select _𝑚_ =4 for this work. 

**Computation Reduction** . Fig. 17 compares the computation reduction of LLM inference across different accelerators. SOFA, which focuses solely on attention and adopts coarse-grained valuelevel sparsity, yields the lowest reduction and is used as the baseline. Bitwave enhances performance by exploiting bit-level sparsity, achieving a 32% reduction, and outperforming value-sparsity-based accelerators like FACT and Spatten. However, it does not capitalize on bit-repetition. FuseKNA utilizes bit-repetition but fails to exploit attention sparsity, limiting its reduction to 49%. By contrast, MCBP achieves up to 72 _._ 4% reduction by exploiting fine-grained bit-repetition, sparsity and attention dynamic sparsity. 

**Memory Access Reduction** . FuseKNA, which only exploits value compression by run-length coding, serves as the baseline. FACT and Spatten utilize IO-intensive top- _𝑘_ to speculate trivial KV tokens, which successfully reduces computation but leading to redundant IO traffic. SOFA exclusively targets KV memory traffic in the attention module via cross-stage tiling, but does not mitigate weight traffic during the _decoding_ stage. Thus, it shows comparable memory reduction to Bitwave in long-sequence tasks (e.g., Dolly, Wikilingua), but performs less effectively with short sequences like Cola. This is because, in short-sequence tasks, the memory bottleneck lies in the weight traffic, which SOFA fails to mitigate. In contrast, MCBP achieves an average memory reduction of 75 _._ 8% across both long- and short-sequence tasks, attributed to BSTC and BGPP, which respectively reduce weight and KV cache traffic. 

## **5.3 Architecture Evaluation** 

We first set an ablation study to evaluate the latency reduction of BRCR, BSTC and BGPP against a baseline, which is assumed to be vanilla bit computation + value-level Huffman compression + valuelevel top- _𝑘_ prediction. Latency is evaluated by mapping various workloads on the MCBP accelerator. As shown in Fig.19 (a), BRCR reduces average latency by 30% over the baseline, by eliminating redundant computation in the prefill stage. Further, BSTC and BGPP achieve a further 44% latency reduction by significantly reducing I/O traffic from weights and KV cache during decoding. 

**==> picture [242 x 111] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Throughput and (b) energy efficiency gain of MCBP and GPU 1.4 Latency breakdown for Llama7B (Dolly)<br>108 GPU(B=8) GPU(B=128) MCBP Standard(B=8) MCBP Aggressive(B=8)8.729.43 1.20.81 Compute Memory access Bit Shift3x<br>6 0.6<br>4 0.4<br>20 1.032.13 0.20 Base MCBP Base MCBP Base MCBP<br>353025 GPU (B=8) GPU (B=128) MCBP Standard (B=8) MCBP Aggressive (B=8)29.231.1 Value-level computation(Baseline) Dolly Wikilingua MCBP Geo Mean<br>2015 17.1%<br>105 1.355.59 47% 53% 32.9% 50%<br>0<br>Compute Memory access Bit shift<br>(c) Latency comparison for MCBP and<br>Llama7B Llama13B OPT1B3 Bloom1B7 [Qwen7B] value-based execution (baseline)<br>Norm latency<br>(a) Throughput gain<br>(b) Efficiency gain<br>**----- End of picture text -----**<br>


**Figure 20: (a) Throughput and (b) energy efficiency gain of MCBP. (c) Breakdown for the overhead of bit shifting.** 

Fig.20 (c) profiles the latency overhead between typical value level INT8 computation and MCBP (bit-level). Despite a 17% bitshifting overhead, the overall 3× latency reduction proves that the gain achieved through bit sparsity effectively covers this overhead. 

Fig.19 (b) shows the individual contributions of BRCR, BSTC, and BGPP across two LLaMA-7B tasks. For the Dolly long-text summarization task, we maintain a decoding length of 48 tokens and test different schemes with varying prompt lengths. In this case, BRCR delivers the primary speedup, achieving 3 _._ 9× and 2 _._ 8× latency reduction for 1k and 4k prompts, respectively, while BSTC and BGPP achieve only 1 _._ 6× and 1 _._ 2× acceleration at 1k prompt. This is because GEMM computation dominates 55% of total latency in prompt-driven long-text summarization, making BRCR the most effective. With 4k prompts, BGPP outperforms BSTC due to increased KV cache memory access. For code generation task MBPP, BRCR only reduces latency by 1 _._ 2×, as the serial autoregressive decoding stage dominates latency. With 1k decoding length, BSTC achieves 2 _._ 7× latency reduction for weight traffic reduction, and BGPP achieves 1 _._ 4× for KV cache reduction. With 4 _𝑘_ decoding length, BGPP increases to 2 _._ 1×, while BSTC drops to 1 _._ 6×. 

_Throughput Gain:_ Fig. 20 (a) compares the throughput of MCBP with A100 GPU on all benchmarks with batch sizes of 8 and 128. Given the INT8 compute power of A100 is 624 TOPS, we use 148 MCBP processors (total with 622TOPS@INT8) with data and model parallelism for performance comparison. First, we observe that B=128 provides an average 2 _._ 1× throughput gain over B=8, primarily due to amortized memory access. However, this benefit saturates, as a 16× increase in batch size results in only a 2× throughput gain. In contrast, MCBP achieves an 8 _._ 72× speedup over A100 with the same batch size. Second, we can see naively applying MCBP’s algorithm on GPU yields only 1.03× speed up, as GPUs cannot exploit bit-slice repetition and fine-grained dataflow and progressive sparsity prediction. In contrast, MCBP accelerator achieves 78% average utilization due to its Transformer-oriented workflow, which fully pipelines Parallel BSTC en/decoders, BRCR acceleration, BGPP predictor, leading to nearly 8× higher sparsity utilization than GPU. Overall, MCBP standard and aggressive achieve an average 8 _._ 72×/9 _._ 43× inference speed up, respectively. 

Fig. 21 (a) gives the throughput gain breakdown, where **software gain** refers to the improvements achieved by directly deploying 

1602 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Wang et al. 

**==> picture [242 x 68] intentionally omitted <==**

**----- Start of picture text -----**<br>
10 9.43 35 31.1<br>8 Software gainHardware gain 6.311.48x 3025 Software gainHardware gain 2.44x<br>6 2.19x 20 12.6<br>4 2.88 15 2.98x<br>2.88x 10 4.24<br>20 1.2x 1.44x 1.23x 50 4.24x 1.39x 1.58x 1.42x<br>GPU +BRCR +BSTC +BGPP GPU +BRCR +BSTC +BGPP<br>Throughput gain breakdown Baseline (a)  Efficiency gain breakdown Baseline (b)<br>**----- End of picture text -----**<br>


**Figure 21: Throughput and energy efficiency gain breakdown.** 

**Table 3: Hardware Configurations of MCBP.** 

**==> picture [243 x 194] intentionally omitted <==**

**----- Start of picture text -----**<br>
Main Modules Parameters<br>CAM-based BRCR Unit 20 PE Clusters (160 PEs)<br>One 512B CAM unit; 16 index converters<br>Processing Element (PE) 16 Add merge units; 1 Reconstruction unit<br>BSTC CODEC Unit 20 × 4 decoders; 10 × 4 encoders<br>64 64-input AND-based Adder-trees<br>Clock-gated BGPP Unit 4 Clock-gated Progressive Filters<br>Auxiliary Processing Unit 1 Embedding unit; 1 Special function unit; 1 Quantizer<br>384KB Token SRAM; 768KB Weight SRAM<br>On chip Buffer 96KB Temp SRAM<br>Main Memory HBM2, 8×128-bit HBM channels @2GHz, 8GB<br>(a) Area Breakdown (9.52 mm [2] ) (b) Overall power breakdown (2.395W)<br>BGPP Unit4.5% SRAM SRAM Scheduler 4.1%<br>BSTC Unit6.2% 19.1% Scheduler 13.4% DRAM 47.6% Core part  BGPP Unit8.2% 22.0% 11.7%APU<br>APU  37.3% BSTC Unit<br>BRCR Unit 18.4% 10.2% BRCR Unit<br>38.2% Memory  44.7%<br>interface<br>15.1%<br>Off-chip DRAM: HBM2, 8 x 128-bit HBM channels<br>**----- End of picture text -----**<br>


**Figure 22: Area/Power of MCBP at TSMC 28nm, 1GHz.** 

software optimizations on the GPU. Although the bit repetitionleveraged BRCR theoretically reduces computation by 5 _._ 7×, practical throughput improves by only 1 _._ 2×. This discrepancy arises from the GPU’s inefficiency in fine-grained bit-level operations and merging redundant elements, resulting in exposed latency bottlenecks when identifying repetitive elements. After adding the dedicated CAM-based BRCR engine, the performance jumps by 2 _._ 88×. Similarly, directly applying bit-sparsity BSTC scheme and bit-prediction BGPP scheme yields only 1 _._ 44× and 1 _._ 23× gain, as the value-to-bit reorder cost and mismatched computation granularity, which lead to severe underutilization of GPU resources. By contrast, employing tailored engines can further bring 2 _._ 19× and 1 _._ 48× acceleration effects. Interestingly, although BSTC achieves smaller performance gains (2 _._ 19×) than BRCR (2 _._ 88×) on ASICs, it yields greater improvements (1 _._ 44× than 1 _._ 2×) on GPUs. This is primarily because BSTC significantly reduces memory access overhead, allowing GPUs, despite lacking dedicated encoding/decoding support, to still benefit from the optimization. A similar trend is observed with BGPP, which reduces memory access through token sparsification, thus enabling performance gains on GPUs as well. 

_Area, Power and Energy:_ Table 3 summarizes the hardware configuration of MCBP and Fig. 22 shows its area and power breakdown. Here, we scale up the MCBP accelerator to contain 16 PE clusters to match the HBM I/O interface. The total power includes the 

**Table 4: Summary and comparison with SOTA works.** 

|**Acceleration for**<br>**Optimization level**‡<br>**Technology [nm]**<br>**Area [mm**2**]**<br>**Throughput [GOPS]**<br>**Energy Ef. [GOPS/W]**|Prefll<br>Prefll<br>Prefll<br>**P**&**D**<br>(attention)<br>(whole model)<br>(attention) **whole model**<br>Value C.<br>Value G.C.<br>Value C.<br>**Bit G.W.C.**<br>40<br>28<br>28<br>28<br>1_._55<br>6_._03<br>4_._29<br>9_._52<br>360<br>1153<br>24_,_423<br>54_,_463<br>382<br>4388<br>7183<br>22_,_740|
|---|---|



> ‡ G: GEMM, W: Weight access. C: KV cache access. And optimizing at the value or bit-level. 

core logic, memory interface [44], and external HBM. It has a total area of 9 _._ 52 mm[2] and 2.395W power consumption. Benefted by the lightweight design of BSTC encoders and decoders, CODEC part accounts for merely 6 _._ 2% and 10% of area and core part power. Despite average 75 _._ 8% reduction in IO traffic, DRAM power still accounts for approximately 48% of total power consumption, due to the autoregressive nature of LLMs. Fig. 20 (b) shows the overall energy-efficiency gain of MCBP over the A100 GPU. On average, MCBP standard/aggressive achieves 29 _._ 2×/31 _._ 1× greater efficiency than running dense benchmarks on GPU. Compared to naively running algorithm mechanism on GPU, MCBP standard/aggressive realizes 21 _._ 6×/23 _._ 1× gain. Fig. 21 (b) also shows the efficiency gain breakdown. Software-hardware co-design BRCR, TSBC and BGPP bring 4 _._ 24×, 2 _._ 98× and 2 _._ 44× efficiency gain, respectively. 

## **5.4 Comparison with SOTA Accelerators** 

Fig. 23 compares the throughput and energy of various accelerators during prefill and decoding. Energy is broken down into compute, bit reordering, and off-chip memory. In the prefill stage (Fig. 23(a)), computation consistently accounts for over 30% of total energy across all designs. Bit-reordering overhead is significant in FuseKNA (30%) and BitWave (18%) due to their value- or multi-bit compression schemes, which misalign with bit-serial processing. In contrast, MCBP limits this overhead to 3% via a bit-slice–first encoding strategy. In terms of throughput, for long-sequence tasks like Dolly with high token-level sparsity, traditional Transformer accelerators like SOFA, Spatten, and FACT achieve notable speedup. In this case, MCBP offers a smaller advantage over Energon (3 _._ 8×). However, in short-sequence tasks like MBPP, where token sparsity diminishes, bit-level redundancy becomes more exploitable. FuseKNA gains 3 _._ 7× speedup via bit-repetition but suffers from high-latency serial matching. In contrast, MCBP achieves the best acceleration, with an average speedup of 6 _._ 2×. 

In the _decoding_ stage (Fig.23 (b)), speedup mainly comes from reduced memory access for weights and KV cache. For long-text tasks like Dolly, where the KV cache size exceeds the weight size, attention optimization in SOFA, Spatten, and FACT yields a 3 _._ 7× speedup. Bitwave, optimizing only weights, achieves just 1 _._ 3× speedup. As sequence length shortens (e.g., Wikilingua), the KV cache shrinks proportionally, leading to performance degradation in traditional Transformer accelerators. For code generation tasks (MBPP), Bitwave benefits from its weight-centric design. Across all workloads, MCBP achieves the highest performance, averaging 4 _._ 8× speedup. 

Table 4 summarizes the specifications for SpAtten, FACT, and SOFA. They are the SOTA accelerators that exploit the attention 

1603 

MCBP: A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness 

**==> picture [242 x 235] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Throughput and energy comparison in  Prefill  stage of Llama7B<br>12 1.2<br>Energy: Computing Bit reorder Off chip mem Speed up<br>10 Dolly Wikilingua MBPP Mean 1<br>8 0.8<br>6 0.6<br>4 0.4<br>2 0.2<br>0 0<br>SO S FA B FU M SO S FA B FU M SO S FA B FU M SO S FA B FU M<br>(b) Throughput and energy comparison in  Decoding  stage of Llama7B<br>1.2<br>8 Speed up Energy: Computing Bit reorder Off chip mem 1<br>6 Dolly Wikilingua Mean 0.8<br>MBPP<br>4 0.6<br>0.4<br>2 0.2<br>0 0<br>FU FA S SO B M FU FA S SO B M FU FA S SO B M FU FA S SO B M<br>FU: FuseKNA;  FA: FACT;  S: Spatten;  SO: SOFA;  B: Bitwave;  M:MCBP<br>Figure 23: Speedup on Llama7B (a) Prefill (b) Decoding.<br>(a) Impact of      on accuracy and attention sparsity  (b) Ablation for the proposed three techniques<br>4540 10080 1.20.81 Area CAM overhead Energy EfficiencyPower Throughput 1086<br>35 60 0.60.4 4<br>3025 0.8 Acc-MMLUS0.7pa-MMLU0.6 0.5 Acc-MBPPSpa-MBPP0.4 0.3 40 0.20 Systolic BRCR +BSTC +BGPP 20<br>Average  array<br>Atten Sparsity (%) Norm Throughput<br>Norm. Area and Power and Energy efficiency<br>Norm Energy<br>Norm Throughput<br>Norm Energy<br>Norm Throughput<br>Accuracy (%)<br>**----- End of picture text -----**<br>


**Figure 24: (a) Evaluation of MCBP’s optimization impact on inference accuracy. (b) Ablation study of the hardware overhead introduced by the three optimizations.** 

mechanism to improve the energy efficiency of Transformer inference. Spatten and SOFA are both optimized for the attention. FACT focus on the whole model computation acceleration via top- _𝑘_ token pruning. However, these works are all designed for prefill stage by finding redundancy of attention or linear computation, making them unsuitable for decoding stage in LLM. In addition, their optimizations all remain at value level, missing abundant opportunities at bit level. To the best of our knowledge, MCBP is so far the first work that uses bit-level strategies for LLM inference, reducing both memory and computation effort for both prefill and decoding stages of LLM. The energy efficiency of MCBP (with bitlevel GEMM, weight & KV cache optimizations) is 22740 GOPS/W, which is 35×, 5 _._ 2× and 3 _._ 2× greater than the three counterparts, with different technology normalized to 28nm for fair comparison. The average energy efficiency is evaluated using the metric from each respective paper. However, SOFA experiences over a 4× efficiency degradation when applied to autoregressive LLMs, as it is tailored for parallel processing of attention and fails to address the memory access bottleneck in LLMs. In contrast, MCBP achieves a 12 _._ 8× efficiency gain over SOFA when processing LLMs. 

## **6 Discussion** 

Among the three optimizations in MCBP, BRCR and BSTC are lossless, since they leverage intrinsic data redundancy and sparsity for acceleration. In contrast, BGPP introduces a hyperparameter _𝛼[𝑟]_ to select vital KVs (a.k.a., attention sparsity, §3.3), which may affect accuracy. Fig. 24 (a) evaluate the impact of _𝛼[𝑟]_ on accuracy and attention sparsity using LLaMA-7B on two tasks: MMLU (reasoning) and MBPP (generation). Overall, a smaller _𝛼[𝑟]_ results in more aggressive pruning, which decreases model accuracy but increases 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

**==> picture [242 x 103] intentionally omitted <==**

**----- Start of picture text -----**<br>
100k80k (a) Data distribution of Llama13B weights 0.81 (b) Bit sparsity vs value sparsity in PTQ8 and QAT8PTQ INT8 72.2% QAT INT8 71.1%<br>60k 0.6<br>40k 0.4<br>20k 0.2 6.3% 5.2%<br>0 -100 -50 0 50 100 0<br>Quantitative values<br>(c) Bit sparsity vs value sparsity in PTQ4 (d) Acceleration of BRCR and BSTC in llama13B dolly<br>0.81 PTQ INT4 66.3% 1.21 BRCR BSTC 1.21<br>0.6 0.8 0.8<br>0.4 0.6 0.6<br>0.2 16.4% 0.40.2 0.40.2<br>0 3rd BS 2nd BS 1st BS mean Value 0 ValueValue PTQ INT8 QAT INT8 Value PTQ INT4 0<br>7th BS 6th BS 5th BS 4th BS 3rd BS 2nd BS 1st BS mean Value 7th BS 6th BS 5th BS 4th BS 3rd BS 2nd BS 1st BS mean Value<br>count<br>Sparsity ratio<br>Sparsity ratio<br>Norm computation Norm mem access<br>**----- End of picture text -----**<br>


**Figure 25: The bit sparsity for diverse quantization scenarios.** 

sparsity. There are some key observations from Fig. 24 (a): For generation tasks (MBPP), accuracy drops noticeably when _𝛼[𝑟] <_ 0 _._ 6. In contrast, for reasoning tasks (MMLU), the model is more tolerant to pruning, with performance degrading significantly only when _𝛼[𝑟] <_ 0 _._ 5. This may be because reasoning tasks rely on key tokens for inference, resulting in higher token redundancy. On the other hand, the sparsity gains begin to diminish when _𝛼[𝑟] <_ 0 _._ 5, this may be because overly aggressive pruning hurts some critical tokens. Therefore, to strike a well balance between accuracy and sparsity, we set _𝛼[𝑟]_ in the range of 0.5–0.6 in MCBP. 

Fig. 24 (b) presents an ablation exploration for BRCR, BSTC and BGPP, in terms of area and energy overhead, using a systolic array (SA) that provides the same throughput as the baseline. Although CAM adds 25% area and 47% power overhead to the BRCR unit, BRCR still reduces overall area and power by 45% and 72%, respectively, while boosting energy efficiency by 3 _._ 6×. These gains stem from BRCR’s efficient use of bit-level redundancy to eliminate redundant computations (§3.1). Additionally, the integration of CAM enhances pipeline efficiency, contributing to overall performance gains. Building upon this, BSTC applies bit sparsity optimization, achieving a 2 _._ 2× throughput gain with only 16% area and 20% energy overhead, driven by significantly reduced memory access. Finally, BGPP achieves a further 1 _._ 48× throughput gain with just 9% area and 13% energy overhead, owing to the reduction in attention computation and associated memory access operations. 

To explore bit-level sparsity across various quantization strategies, we profile Llama13B’s weights under QAT INT8, PTQ INT8, and PTQ INT4, as shown in Fig. 25(a). The weight distributions for QAT and PTQ INT8 are similar, likely due to LLMs’ fault tolerance enabling effective INT8 quantization. In contrast, PTQ INT4 exhibits a more concentrated distribution due to its lower bit width. 

We compare bit and value sparsity across the three quantization strategies. The 7th BS denotes the highest bit-slice matrix (excluding the sign bit), while the 1st BS stands for the lowest. Fig. 25 (b) shows the average bit sparsity for PTQ and QAT INT8 is about 11× higher than value sparsity. In contrast, PTQ INT4 notably increases value sparsity to ∼ 16% (Fig.25(c)), but bit sparsity remains higher at 66% (∼ 4× higher). Fig. 25(d) shows that BRCR reduces computation by 80%, 79 _._ 45%, and 51% for PTQ INT8, QAT INT8, and PTQ INT4, respectively, while BSTC cuts memory accesses by 71%, 70 _._ 5%, and 41%. These results highlight MCBP’s broad effectiveness, driven by the greater prevalence of bit-level sparsity. 

Cambricon-C (Cam-C) [11] is a SOTA INT4 accelerator that achieves computational acceleration by efficiently looking up all 

1604 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Wang et al. 

**==> picture [242 x 68] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Prefill stage of the Dolly Task (b) Decoding stage of the Dolly Task<br>2 Llama13B Llama7B Bloom1B7 1.2 3 Llama13B Llama7B Bloom1B7 1.2<br>1 2.5 1<br>1.5<br>0.8 2 0.8<br>1 0.6 1.5 0.6<br>0.4 1 0.4<br>0.5<br>0.2 0.5 0.2<br>0 0 0 0<br>Computing core On-chip mem Off chip mem Speed up<br>Norm Speed Up Norm Energy Norm Speed Up Norm Energy<br>**----- End of picture text -----**<br>


**Figure 26: Compared with SOTA INT4 accelerator.** 

256 product results (INT4×INT4, W4A4), avoiding explicit computation. We reproduce Cam-C with the same PE array area (4.65mm2) and 1248kB of on-chip SRAM as used in MCBP. Considering that W4A4 quantization is too aggressive and typically results in a 4–6% accuracy degradation [34] on modern LLMs, we adopt a more conservative W4A8 quantization for comparisons. Accordingly, we extend Cam-C to support W4A8, while retaining its core optimization technique—Quarter Square Multiplication. Using the QLLM framework [53], we quantize BloomB7, Llama7B/13B to W4A8, ensuring an accuracy loss of less than 1% compared to FP16. We evaluate the performance of Cam-C and MCBP on Dolly dataset of the three models and draw the following three key observations: 

First, Cam-C suffers from significant look-up overhead. This is because Cam-C relies on the look-up of all possible results. When the activation is extended to INT8, the cost of look-up increases dramatically, limiting Cam-C’s acceleration. This limitation is particularly evident with small models, e.g. Bloom1B7, where valuelevel redundancy cannot be guaranteed as their small hidden sizes. As shown in Fig. 26 (a), compared to Cam-C, MCBP achieves 1 _._ 5× speedup and 33% energy savings on LLaMA-13B, and 1 _._ 8× speedup with 50% lower energy consumption on Bloom-1B7. This benefits from MCBP maximizes redundancy utilization unexploited at the **bit level** instead of value level. Second, Cam-C fails to leverage the inherent bit sparsity of INT4 and attention sparsity for memory optimization, leading to poor performance during decoding stage. In contrast, MCBP utilizes BSTC to exploit the sparsity of INT4 and BGPP for KV cache traffic reduction, reducing memory access and achieving an average 2 _._ 4× speedup, as depicted in Fig. 26 (b). 

Overall, MCBP demonstrates an evident performance advantage over the SOTA INT4 accelerator, thanks to its comprehensive optimization of the LLM inference bottleneck at the bit level. 

## **7 Related Works** 

**Transformer accelerator** . Numerous works [7, 19, 20, 22, 25, 26, 33, 46, 50, 54, 59, 72, 74, 75, 80, 92, 94, 97, 104, 106, 113] have been proposed to improve efficiency of Transformer-based LLMs. Given the quadratic complexity of attention in long sequences [90], many focus on accelerating attention via static sliding windows [7, 19, 46, 80, 108] or dynamic top- _𝑘_ prediction [25, 26, 50, 54, 59, 75, 92, 106]. Other works extend token sparsity to linear layers [19, 22, 46, 72, 74, 94, 97, 104, 113]. However, due to the autoregressive nature of LLMs, weight and KV cache memory access dominate latency during the decoding stage—an aspect largely overlooked by prior designs. In contrast, MCBP addresses these bottleneck holistically. In the _prefill_ stage, MCBP optimizes computation with bit-slice (BS) repetition, while in the _decoding_ stage, it minimizes memory access through bit sparsity and bit-grained early termination. 

**Value sparsity accelerator** . Numerous accelerators [16, 21, 22, 24, 28–30, 39, 45, 49, 55, 56, 60, 62, 63, 71, 82, 86, 89, 100, 102, 107, 108, 110] exploit value sparsity to improve NNs performance. EIE [28] utilizes dynamic input and static weight sparsity to accelerate CNNs and RNNs, while S2TA [56] leverages structured sparsity in weights and activations to accelerate CNNs. Recent efforts [21, 22, 55, 86, 89, 107] have extended this idea to LLMs, e.g., EdgeBERT [86] uses masks to skip zeros in weights to reduce unnecessary computation. However, value sparsity in LLMs is highly limited (Llama13B 6.3%). In contrast, MCBP uses extremely fine-grained bit sparsity, which is average 10 _._ 1× higher than value sparsity, bit repetition, and bit prediction to remove redundant memory access. 

**Bit-serial computing accelerators** . Prior works [2, 14, 15, 23, 27, 35, 35, 37, 38, 38, 43, 47, 48, 57, 58, 68, 78, 103, 105] accelerates neural networks by exploiting bit-level sparsity within individual BS vector [2, 15, 58, 78, 103] or dynamically reducing bit-width [23, 47, 68, 105]. However, such techniques fall short for LLMs, which are both memory- and computation-intensive. Besides, they often target irregular activation sparsity [15, 78, 103], which doesn’t address LLM bottlenecks like weight and KV cache access. In contrast, MCBP eliminates redundancy across BS vectors and uses BS sparsity and fine-grained bit prediction to reduce weight and KV cache memory accesses. 

## **8 Conclusion** 

We propose MCBP, a software-hardware co-design to accelerate the computation, weight access and KV cache access for LLM inference. Utilizing the bit-level repetition, sparsity and reduced prediction traffic, MCBP achieves 31 _._ 1×, 35×, 5 _._ 2× and 3 _._ 2× energy saving than A100 GPU, SOTA accelerators SpAtten, FACT and SOFA. 

## **Acknowledgments** 

This work was supported in part by the National Science and Technology Major Project under Grant 2022ZD0115200; the NSFC under Grant 62125403, and Grant 92164301; Beijing S&T Project Z221100007722023; in part by the project funding for the 2022 Special Project on Industrial Foundation Reconstruction and High Quality Development of Manufacturing Industry CEIEC-2022-ZM020245; in part by the Beijing National Research Center for Information Science and Technology; and in part by the Beijing Advanced Innovation Center for Integrated Circuits. 

## **References** 

- [1] Miklós Ajtai. 1994. The complexity of the pigeonhole principle. _Combinatorica_ 14 (1994), 417–433. 

- [2] Jorge Albericio, Alberto Delmás, Patrick Judd, Sayeh Sharify, Gerard O’Leary, Roman Genov, and Andreas Moshovos. 2017. Bit-pragmatic deep neural network computing. In _Proceedings of the 50th annual IEEE/ACM international symposium on microarchitecture_ . 382–394. 

- [3] Renzo Andri, Lukas Cavigelli, Davide Rossi, and Luca Benini. 2016. YodaNN: An ultra-low power convolutional neural network accelerator based on binary weights. In _Proceedings of the IEEE Computer Society Annual Symposium on VLSI (ISVLSI)_ . 236–241. 

- [4] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023. Palm 2 technical report. _arXiv preprint arXiv:2305.10403_ (2023). 

- [5] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. 2021. Program synthesis with large language models. _arXiv preprint arXiv:2108.07732_ (2021). 

1605 

MCBP: A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness 

- [6] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. 2023. Qwen technical report. _arXiv preprint arXiv:2309.16609_ (2023). 

- [7] Zhenyu Bai, Pranav Dangi, Huize Li, and Tulika Mitra. 2024. SWAT: Scalable and efficient window attention-based transformers acceleration on FPGAs. In _Proceedings of the 61st ACM/IEEE Design Automation Conference_ . 1–6. 

- [8] Lukas Cavigelli and Luca Benini. 2016. Origami: A 803-GOp/s/W convolutional network accelerator. _IEEE Transactions on Circuits and Systems for Video Technology_ 27, 11 (2016), 2461–2475. 

- [9] Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, and Xing Xie. 2024. A survey on evaluation of large language models. _ACM Transactions on Intelligent Systems and Technology_ 15, 3 (2024), 1–45. 

- [10] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ (2021). 

- [11] Yi Chen, Yongwei Zhao, Yifan Hao, Yuanbo Wen, Yuntao Dai, Xiaqing Li, Yang Liu, Rui Zhang, Mo Zou, Xinkai Song, Xing Hu, Zidong Du, Huaping Chen, Qi Guo, and Tianqi Chen. 2024. Cambricon-C: Efficient 4-Bit Matrix Unit via Primitivization. In _2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . IEEE, 538–550. 

- [12] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. 2023. Vicuna: An open-source chatbot impressing GPT-4 with 90%* ChatGPT quality. _See https://vicuna. lmsys. org (accessed 14 April 2023)_ 2, 3 (2023), 6. 

- [13] Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. 2023. Free Dolly: Introducing the world’s first truly open instruction-tuned LLM. _Company Blog of Databricks_ (2023). 

- [14] Alberto Delmas, Patrick Judd, Sayeh Sharify, and Andreas Moshovos. 2017. Dynamic stripes: Exploiting the dynamic precision requirements of activation values in neural networks. _arXiv preprint arXiv:1706.00504_ (2017). 

- [15] Alberto Delmas Lascorz, Patrick Judd, Dylan Malone Stuart, Zissis Poulos, Mostafa Mahmoud, Sayeh Sharify, Milos Nikolic, Kevin Siu, and Andreas Moshovos. 2019. Bit-tactical: A software/hardware approach to exploiting value and bit sparsity in neural networks. In _Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems_ . 749–763. 

- [16] Chunhua Deng, Yang Sui, Siyu Liao, Xuehai Qian, and Bo Yuan. 2021. GoSPA: An energy-efficient high-performance globally optimized sparse convolutional neural network accelerator. In _Proceedings of the ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)_ . 1110–1123. 

- [17] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. 2022. GPT3.Int8 (): 8-bit matrix multiplication for Transformers at scale. _Advances in Neural Information Processing Systems_ 35 (2022), 30318–30332. 

- [18] Claire Cardie Faisal Ladhak, Esin Durmus and Kathleen McKeown. 2020. WikiLingua: A new benchmark dataset for multilingual abstractive summarization. In _Findings of EMNLP, 2020_ . 

- [19] Hongxiang Fan, Thomas Chau, Stylianos I Venieris, Royson Lee, Alexandros Kouris, Wayne Luk, Nicholas D Lane, and Mohamed S Abdelfattah. 2022. Adaptable butterfly accelerator for attention-based NNs via hardware and algorithm co-design. In _Proceedings of the 55th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . 599–615. 

- [20] Zichen Fan, Qirui Zhang, Pierre Abillama, Sara Shoouri, Changwoo Lee, David Blaauw, Hun-Seok Kim, and Dennis Sylvester. 2023. Taskfusion: An efficient transfer learning architecture with dual delta sparsity for multi-task natural language processing. In _Proceedings of the 50th Annual International Symposium on Computer Architecture_ . 1–14. 

- [21] Chao Fang, Shouliang Guo, Wei Wu, Jun Lin, Zhongfeng Wang, Ming Kai Hsu, and Lingzhi Liu. 2022. An efficient hardware accelerator for sparse Transformer 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

   - neural networks. In _2022 IEEE International Symposium on Circuits and Systems (ISCAS)_ . IEEE, 2670–2674. 

- [22] Chao Fang, Aojun Zhou, and Zhongfeng Wang. 2022. An algorithm-hardware co-optimized framework for accelerating N:M sparse Transformers. _IEEE Transactions on Very Large Scale Integration (VLSI) Systems_ 30, 11 (2022), 1573–1586. 

- [23] Ashish Gondimalla, Noah Chesnut, Mithuna Thottethodi, and TN Vijaykumar. 2019. SparTen: A sparse tensor accelerator for convolutional neural networks. In _Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture_ . 151–165. 

- [24] Sumanth Gudaparthi, Sarabjeet Singh, Surya Narayanan, Rajeev Balasubramonian, and Visvesh Sathe. 2022. CANDLES: Channel-aware novel dataflowmicroarchitecture co-design for low energy sparse neural network acceleration. In _Proceedings of the IEEE International Symposium on high-performance computer architecture (HPCA)_ . 876–891. 

- [25] Tae Jun Ham, Sung Jun Jung, Seonghak Kim, Young H Oh, Yeonhong Park, Yoonho Song, Jung-Hun Park, Sanghee Lee, Kyoung Park, Jae W Lee, and DeogKyoon Jeong. 2020. A[3] : Accelerating attention mechanisms in neural networks with approximation. In _Proceedings of the IEEE International Symposium on High Performance Computer Architecture (HPCA)_ . 328–341. 

- [26] Tae Jun Ham, Yejin Lee, Seong Hoon Seo, Soosung Kim, Hyunji Choi, Sung Jun Jung, and Jae W Lee. 2021. ELSA: Hardware-software co-design for efficient, lightweight self-attention mechanism in neural networks. In _Prceedings of the 48th ACM/IEEE Annual International Symposium on Computer Architecture (ISCA)_ . 692–705. 

- [27] Meng Han, Liang Wang, Limin Xiao, Hao Zhang, Tianhao Cai, Jiale Xu, Yibo Wu, Chenhao Zhang, and Xiangrong Xu. 2024. BitNN: A bit-serial accelerator for k-nearest neighbor search in point clouds. In _Proceddings of the ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)_ . 1278–1292. 

- [28] Song Han, Xingyu Liu, Huizi Mao, Jing Pu, Ardavan Pedram, Mark A Horowitz, and William J Dally. 2016. EIE: Efficient inference engine on compressed deep neural network. _ACM SIGARCH Computer Architecture News_ 44, 3 (2016), 243– 254. 

- [29] Song Han, Huizi Mao, and William J Dally. 2015. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. _arXiv preprint arXiv:1510.00149_ (2015). 

- [30] Edward Hanson, Shiyu Li, Hai‘Helen’ Li, and Yiran Chen. 2022. Cascading structured pruning: Enabling high data reuse for sparse DNN accelerators. In _Proceedings of the 49th Annual International Symposium on Computer Architecture_ . 522–535. 

- [31] Kartik Hegde, Jiyong Yu, Rohit Agrawal, Mengjia Yan, Michael Pellauer, and Christopher Fletcher. 2018. UCNN: Exploiting computational reuse in deep neural networks via weight repetition. In _2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA)_ . IEEE, 674–687. 

- [32] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. _arXiv preprint arXiv:2009.03300_ (2020). 

- [33] Seongmin Hong, Seungjae Moon, Junsoo Kim, Sungjae Lee, Minsub Kim, Dongsoo Lee, and Joo-Young Kim. 2022. DFX: A low-latency multi-FPGA appliance for accelerating Transformer-based text generation. In _Proceedings of the 55th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . 616–630. 

- [34] Yuxuan Hu, Xiaodong Chen, Cuiping Li, Hong Chen, and Jing Zhang. 2025. QUAD: Quantization and Parameter-Efficient Tuning of LLM with Activation Decomposition. _arXiv preprint arXiv:2503.19353_ (2025). 

- [35] Dongseok Im, Gwangtae Park, Zhiyong Li, Junha Ryu, and Hoi-Jun Yoo. 2023. Sibia: Signed bit-slice architecture for dense dnn acceleration with slice-level sparsity exploitation. In _2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . IEEE, 69–80. 

- [36] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. 2018. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ . 2704–2713. 

- [37] Patrick Judd, Jorge Albericio, Tayler Hetherington, Tor M Aamodt, and Andreas Moshovos. 2016. Stripes: Bit-serial deep neural network computing. In _Proceedings of the 49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . 1–12. 

- [38] Dongyun Kam, Myeongji Yun, Sunwoo Yoo, Seungwoo Hong, Zhengya Zhang, and Youngjoo Lee. 2024. Panacea: Novel DNN Accelerator using AccuracyPreserving Asymmetric Quantization and Energy-Saving Bit-Slice Sparsity. _arXiv preprint arXiv:2412.10059_ (2024). 

- [39] Sanghoon Kang, Donghyeon Han, Juhyoung Lee, Dongseok Im, Sangyeob Kim, Soyeon Kim, Junha Ryu, and Hoi-Jun Yoo. 2021. GANPU: An energy-efficient multi-DNN training processor for GANs with speculative dual-sparsity exploitation. _IEEE Journal of Solid-State Circuits_ 56, 9 (2021), 2845–2857. 

- [40] Majeed Kazemitabaar, Runlong Ye, Xiaoning Wang, Austin Zachary Henley, Paul Denny, Michelle Craig, and Tovi Grossman. 2024. Codeaid: Evaluating a classroom deployment of an LLM-based programming assistant that balances 

1606 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Wang et al. 

   - student and educator needs. In _Proceedings of the CHI Conference on Human Factors in Computing Systems_ . 1–20. 

- [41] Yoongu Kim, Weikun Yang, and Onur Mutlu. 2015. Ramulator: A fast and extensible DRAM simulator. _IEEE Computer architecture letters_ 15, 1 (2015), 45–49. 

- [42] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. 2022. Bloom: A 176B-parameter open-access multilingual language model. _arXiv preprint arXiv:2211.05100_ (2022). 

- [43] Jinmook Lee, Changhyeon Kim, Sanghoon Kang, Dongjoo Shin, Sangyeob Kim, and Hoi-Jun Yoo. 2018. UNPU: A 50.6 TOPS/W unified deep neural network accelerator with 1b-to-16b fully-variable weight bit-precision. In _Proceedings of IEEE International Solid-State Circuits Conference-(ISSCC)_ . 218–220. 

- [44] Brian Leibowitz, Robert Palmer, John Poulton, Yohan Frans, Simon Li, John Wilson, Michael Bucher, Andrew M Fuller, John Eyles, Marko Aleksic, Trey Greer, and Nhat M Nguyen. 2010. A 4.3 GB/s mobile memory interface with power-efficient bandwidth scaling. _IEEE Journal of Solid-State Circuits_ 45, 4 (2010), 889–898. 

- [45] Jonathan S Lew, Yunpeng Liu, Wenyi Gong, Negar Goli, R David Evans, and Tor M Aamodt. 2022. Anticipating and eliminating redundant computations in accelerated sparse training. In _Proceedings of the 49th Annual International Symposium on Computer Architecture_ . 536–551. 

- [46] Bingbing Li, Santosh Pandey, Haowen Fang, Yanjun Lyv, Ji Li, Jieyang Chen, Mimi Xie, Lipeng Wan, Hang Liu, and Caiwen Ding. 2020. FTRANS: Energyefficient acceleration of Transformers using FPGA. In _Proceedings of the ACM/IEEE International Symposium on Low Power Electronics and Design_ . 175– 180. 

- [47] Gang Li, Weixiang Xu, Zhuoran Song, Naifeng Jing, Jian Cheng, and Xiaoyao Liang. 2022. Ristretto: An atomized processing architecture for sparsitycondensed stream flow in CNN. In _Proceedings of the 55th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . 1434–1450. 

- [48] Guoyu Li, Shengyu Ye, Chunyun Chen, Yang Wang, Fan Yang, Ting Cao, Cheng Liu, Mohamed M Sabry, and Mao Yang. 2025. LUT-DLA: Lookup Table as Efficient Extreme Low-Bit Deep Learning Accelerator. _arXiv preprint arXiv:2501.10658_ (2025). 

- [49] Shiyu Li, Edward Hanson, Xuehai Qian, Hai" Helen" Li, and Yiran Chen. 2021. ESCALATE: Boosting the efficiency of sparse CNN accelerator with kernel decomposition. In _Proceedings of the 54th Annual IEEE/ACM International Symposium on Microarchitecture_ . 992–1004. 

- [50] Zheng Li, Soroush Ghodrati, Amir Yazdanbakhsh, Hadi Esmaeilzadeh, and Mingu Kang. 2022. Accelerating attention through gradient-based learned runtime pruning. In _Proceedings of the 49th Annual International Symposium on Computer Architecture_ . 902–915. 

- [51] Bin Lin, Chen Zhang, Tao Peng, Hanyu Zhao, Wencong Xiao, Minmin Sun, Anmin Liu, Zhipeng Zhang, Lanbo Li, Xiafei Qiu, Li Shen, Zhigang Ji, Tao Xie, Yong Li, and Wei Lin. 2024. Infinite-LLM: Efficient LLM service for long context with distattention and distributed kvcache. _arXiv preprint arXiv:2401.02669_ (2024). 

- [52] Fangxin Liu, Ning Yang, Haomin Li, Zongwu Wang, Zhuoran Song, Songwen Pei, and Li Jiang. 2024. SPARK: Scalable and precision-aware acceleration of neural networks via efficient encoding. In _Proceedings of the IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . 1029–1042. 

- [53] Jing Liu, Ruihao Gong, Xiuying Wei, Zhiwei Dong, Jianfei Cai, and Bohan Zhuang. 2023. QLLM: Accurate and efficient low-bitwidth quantization for large language models. _arXiv preprint arXiv:2310.08041_ (2023). 

- [54] Siqin Liu, Prakash Chand Kuve, and Avinash Karanth. 2024. HSCONN: Hardware-Software Co-Optimization of Self-Attention Neural Networks for Large Language Models. In _Proceedings of the Great Lakes Symposium on VLSI 2024_ . 736–741. 

- [55] Shiwei Liu, Peizhe Li, Jinshan Zhang, Yunzhengmao Wang, Haozhe Zhu, Wenning Jiang, Shan Tang, Chixiao Chen, Qi Liu, and Ming Liu. 2023. 16.2 A 28nm 53.8 TOPS/W 8b sparse Transformer accelerator with in-memory butterfly zero skipper for unstructured-pruned NN and CIM-based local-attention-reusable engine. In _2023 IEEE International Solid-State Circuits Conference (ISSCC)_ . IEEE, 250–252. 

- [56] Zhi-Gang Liu, Paul N Whatmough, Yuhao Zhu, and Matthew Mattina. 2022. S2TA: Exploiting structured sparsity for energy-efficient mobile CNN acceleration. In _Proceedings of the IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . 573–586. 

- [57] Yun-Chen Lo and Ren-Shuo Liu. 2023. Bit-serial cache: Exploiting input bit vector repetition to accelerate bit-serial inference. In _Proceedings of the 60th ACM/IEEE Design Automation Conference (DAC)_ . 1–6. 

- [58] Hang Lu, Liang Chang, Chenglong Li, Zixuan Zhu, Shengjian Lu, Yanhuan Liu, and Mingzhe Zhang. 2021. Distilling bit-level sparsity parallelism for general purpose deep learning acceleration. In _MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture_ . 963–976. 

      - using reconfigurable architecture. In _Proceedings of the 54th Annual IEEE/ACM International Symposium on Microarchitecture_ . 977–991. 

   - [60] Mostafa Mahmoud, Isak Edo, Ali Hadi Zadeh, Omar Mohamed Awad, Gennady Pekhimenko, Jorge Albericio, and Andreas Moshovos. 2020. TensorDash: Exploiting sparsity to accelerate deep neural network training. In _Proceedings of the 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . 781–795. 

   - [61] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer sentinel mixture models. In _Proceedings of the International Conference on Learning Representations_ . 

   - [62] Bert Moons, Roel Uytterhoeven, Wim Dehaene, and Marian Verhelst. 2017. 14.5 Envision: A 0.26-to-10TOPS/W subword-parallel dynamic-voltage-accuracyfrequency-scalable convolutional neural network processor in 28nm FDSOI. In _Proceddings of the IEEE International Solid-State Circuits Conference (ISSCC)_ . 246–247. 

   - [63] Bert Moons and Marian Verhelst. 2016. An energy-efficient precision-scalable ConvNet processor in 40-nm CMOS. _IEEE Journal of solid-state Circuits_ 52, 4 (2016), 903–914. 

   - [64] Naveen Muralimanohar, Rajeev Balasubramonian, and Norman P Jouppi. 2009. CACTI 6.0: A tool to model large caches. _HP laboratories_ 27 (2009), 28. 

   - [65] Daye Nam, Andrew Macvean, Vincent Hellendoorn, Bogdan Vasilescu, and Brad Myers. 2024. Using an LLM to help with code understanding. In _Proceedings of the IEEE/ACM 46th International Conference on Software Engineering_ . 1–13. 

   - [66] Nvidia. 2023. TensorRT-LLM. https://github.com/NVIDIA/TensorRT-LLM? tab=readme-ov-file. 

   - [67] Mike O’Connor, Niladrish Chatterjee, Donghyuk Lee, John Wilson, Aditya Agrawal, Stephen W Keckler, and William J Dally. 2017. Fine-grained DRAM: Energy-efficient DRAM for extreme bandwidth systems. In _Proceedings of the 50th Annual IEEE/ACM International Symposium on Microarchitecture_ . 41–54. 

   - [68] Angshuman Parashar, Minsoo Rhu, Anurag Mukkara, Antonio Puglielli, Rangharajan Venkatesan, Brucek Khailany, Joel Emer, Stephen W Keckler, and William J Dally. 2017. SCNN: An accelerator for compressed-sparse convolutional neural networks. _ACM SIGARCH computer architecture news_ 45, 2 (2017), 27–40. 

   - [69] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. 2017. Automatic differentiation in PyTorch. (2017). 

   - [70] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, and Ricardo Bianchini. 2024. Splitwise: Efficient generative LLM inference using phase splitting. In _2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)_ . IEEE, 118–132. 

   - [71] Eric Qin, Ananda Samajdar, Hyoukjun Kwon, Vineet Nadella, Sudarshan Srinivasan, Dipankar Das, Bharat Kaul, and Tushar Krishna. 2020. Sigma: A sparse and irregular GEMM accelerator with flexible interconnects for DNN training. In _Proceedings of the IEEE International Symposium on High Performance Computer Architecture (HPCA)_ . 58–70. 

   - [72] Yubin Qin, Yang Wang, Dazheng Deng, Zhiren Zhao, Xiaolong Yang, Leibo Liu, Shaojun Wei, Yang Hu, and Shouyi Yin. 2023. Fact: FFN-attention co-optimized Transformer architecture with eager correlation prediction. In _Proceedings of the 50th Annual International Symposium on Computer Architecture_ . 1–14. 

   - [73] Yubin Qin, Yang Wang, Jiachen Wang, Zhiwei Lin, Yushu Zhao, Shaojun Wei, Yang Hu, and Shouyi Yin. 2025. 23.8 An 88.36 TOPS/W Bit-Level-WeightCompressed Large-Language-Model Accelerator with Cluster-Aligned INT-FPGEMM and Bi-Dimensional Workflow Reformulation. In _2025 IEEE International Solid-State Circuits Conference (ISSCC)_ , Vol. 68. IEEE, 420–422. 

   - [74] Yubin Qin, Yang Wang, Zhiren Zhao, Xiaolong Yang, Yang Zhou, Shaojun Wei, Yang Hu, and Shouyi Yin. 2024. MECLA: Memory-compute-efficient LLM accelerator with scaling sub-matrix partition. In _Proceedings of the 51st ACM/IEEE Annual International Symposium on Computer Architecture (ISCA)_ . 1032–1047. 

   - [75] Zheng Qu, Liu Liu, Fengbin Tu, Zhaodong Chen, Yufei Ding, and Yuan Xie. 2022. DOTA: Detect and omit weak attentions for scalable Transformer acceleration. In _Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems_ . 14–26. 

   - [76] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al. 2023. Code LLaMa: Open foundation models for code. _arXiv preprint arXiv:2308.12950_ (2023). 

   - [77] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial Winograd schema challenge at scale. _Commun. ACM_ 64, 9 (2021), 99–106. 

   - [78] Sayeh Sharify, Alberto Delmas Lascorz, Mostafa Mahmoud, Milos Nikolic, Kevin Siu, Dylan Malone Stuart, Zissis Poulos, and Andreas Moshovos. 2019. Laconic deep learning inference acceleration. In _Proceedings of the 46th International Symposium on Computer Architecture_ . 304–317. 

   - [79] Hardik Sharma, Jongse Park, Naveen Suda, Liangzhen Lai, Benson Chau, Joon Kyung Kim, Vikas Chandra, and Hadi Esmaeilzadeh. 2018. Bit Fusion: 

- [59] Liqiang Lu, Yicheng Jin, Hangrui Bi, Zizhang Luo, Peng Li, Tao Wang, and Yun Liang. 2021. Sanger: A co-design framework for enabling sparse attention 

1607 

MCBP: A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness 

   - Bit-level dynamically composable architecture for accelerating deep neural network. In _Proceedings of the ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA)_ . 764–775. 

- [80] Guan Shen, Jieru Zhao, Quan Chen, Jingwen Leng, Chao Li, and Minyi Guo. 2022. SALO: An efficient spatial accelerator enabling hybrid sparse attentionmechanisms for long sequences. In _Proceedings of the 59th ACM/IEEE Design Automation Conference_ . 571–576. 

- [81] Man Shi, Vikram Jain, Antony Joseph, Maurice Meijer, and Marian Verhelst. 2024. BitWave: Exploiting Column-Based Bit-Level Sparsity for Deep Learning Acceleration. In _Proceedings of IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . 732–746. 

- [82] Jong Hoon Shin, Ali Shafiee, Ardavan Pedram, Hamzah Abdel-Aziz, Ling Li, and Joseph Hassoun. 2022. Griffin: Rethinking sparse optimization for deep learning architectures. In _Proceedings of the IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . 861–875. 

- [83] Wilson Snyder. 2004. Verilator and SystemPerl. In _North American SystemC Users’ Group, Design Automation Conference_ . 

- [84] Benjamin Spector and Chris Re. 2023. Accelerating LLM inference with staged speculative decoding. _arXiv preprint arXiv:2308.04623_ (2023). 

- [85] Salmonn Talebi, Elizabeth Tong, and Mohammad RK Mofrad. 2023. Beyond the Hype: Assessing the Performance, Trustworthiness, and Clinical Suitability of GPT3. 5. _arXiv preprint arXiv:2306.15887_ (2023). 

- [86] Thierry Tambe, Coleman Hooper, Lillian Pentecost, Tianyu Jia, En-Yu Yang, Marco Donato, Victor Sanh, Paul Whatmough, Alexander M Rush, David Brooks, and Gu-Yeon Wei. 2021. Edgebert: Sentence-level energy optimizations for latency-aware multi-task NLP inference. In _Proceedings of the 54th Annual IEEE/ACM International Symposium on Microarchitecture_ . 830–844. 

- [87] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https://crfm.stanford.edu/2023/03/13/alpaca. html. 

- [88] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ (2023). 

- [89] Shikhar Tuli and Niraj K Jha. 2023. AccelTran: A sparsity-aware accelerator for dynamic inference with Transformers. _IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems_ 42, 11 (2023), 4038–4051. 

- [90] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. _Advances in neural information processing systems_ 30 (2017). 

- [91] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. 2018. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In _Proceedings of the International Conference on Learning Representations_ . 

- [92] Huizheng Wang, Jiahao Fang, Xinru Tang, Zhiheng Yue, Jinxi Li, Yubin Qin, Sihan Guan, Qize Yang, Yang Wang, Chao Li, Yang Hu, and Shouyi Yin. 2024. SOFA: A compute-memory optimized sparsity accelerator via cross-stage coordinated tiling. _arXiv preprint arXiv:2407.10416_ (2024). 

- [93] Huizheng Wang, Weihong Xu, Zaichen Zhang, Xiaohu You, and Chuan Zhang. 2021. An efficient stochastic convolution architecture based on fast FIR algorithm. _IEEE Transactions on Circuits and Systems II: Express Briefs_ 69, 3 (2021), 984–988. 

- [94] Hanrui Wang, Zhekai Zhang, and Song Han. 2021. SpAtten: Efficient sparse attention architecture with cascade token and head pruning. In _Proceedings of the IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . 97–110. 

- [95] Huizheng Wang, Zaichen Zhang, Xiaohu You, and Chuan Zhang. 2018. Lowcomplexity Winograd convolution architecture based on stochastic computing. In _2018 IEEE 23rd International Conference on Digital Signal Processing (DSP)_ . IEEE, 1–5. 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

      - _Language Processing: System Demonstrations_ . 38–45. 

   - [100] Yannan Nellie Wu, Po-An Tsai, Saurav Muralidharan, Angshuman Parashar, Vivienne Sze, and Joel Emer. 2023. HighLight: Efficient and flexible DNN acceleration with hierarchical structured sparsity. In _Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture_ . 1106–1120. 

   - [101] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. 2023. Smoothquant: Accurate and efficient post-training quantization for large language models. In _International Conference on Machine Learning_ . PMLR, 38087–38099. 

   - [102] Dingqing Yang, Amin Ghasemazar, Xiaowei Ren, Maximilian Golub, Guy Lemieux, and Mieszko Lis. 2020. Procrustes: A dataflow and accelerator for sparse deep neural network training. In _Proceedings of the 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . 711–724. 

   - [103] Jianxun Yang, Zhao Zhang, Zhuangzhi Liu, Jing Zhou, Leibo Liu, Shaojun Wei, and Shouyi Yin. 2021. FuseKNA: Fused kernel convolution based accelerator for deep neural networks. In _Proceddings of the IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . 894–907. 

   - [104] Tao Yang, Fei Ma, Xiaoling Li, Fangxin Liu, Yilong Zhao, Zhezhi He, and Li Jiang. 2022. DTATrans: Leveraging dynamic token-based quantization with accuracy compensation mechanism for efficient Transformer architecture. _IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems_ 42, 2 (2022), 509–520. 

   - [105] Yifan Yang, Joel S Emer, and Daniel Sanchez. 2023. ISOSceles: Accelerating sparse CNNs through inter-layer pipelining. In _2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . IEEE, 598–610. 

   - [106] Amir Yazdanbakhsh, Ashkan Moradifirouzabadi, Zheng Li, and Mingu Kang. 2022. Sparse attention acceleration with synergistic in-memory pruning and on-chip recomputation. In _Proceedings of the 55th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . 744–762. 

   - [107] Eunji Yoo, Gunho Park, Jung Gyu Min, Se Jung Kwon, Baeseong Park, Dongsoo Lee, and Youngjoo Lee. 2023. TF-MVP: Novel sparsity-aware transformer accelerator with mixed-length vector pruning. In _2023 60th ACM/IEEE Design Automation Conference (DAC)_ . IEEE, 1–6. 

   - [108] Haoran You, Zhanyi Sun, Huihong Shi, Zhongzhi Yu, Yang Zhao, Yongan Zhang, Chaojian Li, Baopu Li, and Yingyan Lin. 2023. ViTCoD: Vision Transformer acceleration via dedicated algorithm and accelerator co-design. In _Proceedings of the IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . 273–286. 

   - [109] Ofir Zafrir, Guy Boudoukh, Peter Izsak, and Moshe Wasserblat. 2019. Q8BERT: Quantized 8bit BERT. In _2019 Fifth Workshop on Energy Efficient Machine Learning and Cognitive Computing-NeurIPS Edition (EMC2-NIPS)_ . IEEE, 36–39. 

   - [110] Shijin Zhang, Zidong Du, Lei Zhang, Huiying Lan, Shaoli Liu, Ling Li, Qi Guo, Tianshi Chen, and Yunji Chen. 2016. Cambricon-X: An accelerator for sparse neural networks. In _2016 49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . IEEE, 1–12. 

   - [111] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022. OPT: Open pre-trained transformer language models. _arXiv preprint arXiv:2205.01068_ (2022). 

   - [112] Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy, Tianqi Chen, and Baris Kasikci. 2024. Atom: Lowbit quantization for efficient and accurate llm serving. _Proceedings of Machine Learning and Systems_ 6 (2024), 196–209. 

   - [113] Zhe Zhou, Junlin Liu, Zhenyu Gu, and Guangyu Sun. 2022. Energon: Toward efficient acceleration of Transformers using dynamic sparse attention. _IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems_ 42, 1 (2022), 136–149. 

- [96] Yizhi Wang, Jun Lin, and Zhongfeng Wang. 2017. An energy-efficient architecture for binary weight convolutional neural networks. _IEEE Transactions on Very Large Scale Integration (VLSI) Systems_ 26, 2 (2017), 280–293. 

- [97] Yang Wang, Yubin Qin, Dazheng Deng, Jingchuan Wei, Yang Zhou, Yuanqi Fan, Tianbao Chen, Hao Sun, Leibo Liu, Shaojun Wei, and Shouyi Yin. 2022. An energy-efficient Transformer processor exploiting dynamic weak relevances in global attention. _IEEE Journal of Solid-State Circuits_ 58, 1 (2022), 227–242. 

- [98] Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu, and Xianglong Liu. 2022. Outlier suppression: Pushing the limit of low-bit transformer language models. _Advances in Neural Information Processing Systems_ 35 (2022), 17402–17414. 

- [99] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. In _Proceedings of the Conference on Empirical Methods in Natural_ 

1608 

