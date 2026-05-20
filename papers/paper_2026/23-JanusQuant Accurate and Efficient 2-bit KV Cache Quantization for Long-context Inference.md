## **JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-Context Inference** 

Chengyu Sun Yaqi Xia School of Computer Science, Wuhan School of Computer Science, Wuhan University University Wuhan, China Wuhan, China chengyusun@whu.edu.cn yaqixia@whu.edu.cn 

## Hulin Wang 

School of Computer Science, Wuhan University Wuhan, China wonghulin@whu.edu.cn 

Donglin Yang Nvidia Corporation Santa Clara, USA dongliny@nvidia.com 

Xiaobo Zhou Dazhao Cheng[∗] IOTSC & Department of CIS, School of Computer Science, Wuhan University of Macau University Macau, China Wuhan, China waynexzhou@um.edu.mo dcheng@whu.edu.cn 

## **Abstract** 

_**Keywords:**_ Long-context Inference, KV Cache Quantization 

Long-context large language models (LLMs) have seen widespread adoption in recent years. However, during inference, the key-value (KV) cache—which stores intermediate activations—consumes significant memory, particularly as sequence lengths grow. Quantization offers a promising path to compress KV cache, but existing 2-bit approaches fall short of achieving optimal inference efficiency due to hardwareunfriendly algorithms and system implementations. 

## **ACM Reference Format:** 

Chengyu Sun, Yaqi Xia, Hulin Wang, Donglin Yang, Xiaobo Zhou, and Dazhao Cheng. 2026. JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-Context Inference. In _Proceedings of the 31st ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP ’26), January 31 – February 4, 2026, Sydney, NSW, Australia._ ACM, New York, NY, USA, 14 pages. https://doi.org/10.1145/3774934.3786428 

We present JanusQuant, a 2-bit KV cache quantization system that achieves both high accuracy and end-to-end efficiency through algorithm–system co-design for longcontext generation tasks. At its core is RtSmooth, a novel runtime smoothing quantization algorithm that mitigates outlier-induced accuracy loss via adaptive transformation. Building on RtSmooth, JanusQuant further enhances quantized inference with a series of optimizations: a fast absmax positioning technique for lightweight quantization, a memory-efficient data structure for managing recent tokens, and a custom mixed-precision attention kernel to accelerate computation. Across representative LLMs, JanusQuant preserves 99% of FP16 accuracy, reduces KV cache memory usage by up to 5.3×, and delivers up to 4.45× faster decoding throughput compared to state-of-the-art methods, while scaling efficiently to long-context inference. 

## **1 Introduction** 

Large language models (LLMs) [1, 8, 38] have demonstrated exceptional performance across a wide range of advanced applications, significantly impacting our daily lives [22, 29, 31, 45, 49]. With the growing demand for tasks such as document summarization and code assistance, LLMs are increasingly expected to handle longer input contexts. Recent models like Deepseek-R1 [9] and Llama-3.3 [28], for example, support sequences up to 128K tokens. However, this extended context capability introduces significant deployment challenges. As sequence lengths grow, the key-value (KV) cache—a transient structure generated during autoregressive inference—begins to dominate memory usage. For instance, while the Llama213B model has a fixed weight size of 24.5GB, serving a single request with 128K tokens in FP16 requires over 100GB of KV cache—far exceeding the capacity of a typical A10040GB GPU. Moreover, the attention mechanism must access all stored KV pairs during decoding, making the process highly memory-bound. As KV cache memory consumption increases, this memory bottleneck becomes a dominant limiter of inference performance. 

## _**CCS Concepts:**_ • **Computing methodologies** → **Parallel algorithms** ; **Natural language processing** . 

∗Corresponding author. 

Quantization is a direct and effective technique for reducing the memory footprint of the KV cache. By converting FP16 activations into lower-precision formats, significant memory savings can be achieved. Prior works such as FlexGen [35] and Atom [53] have incorporated 4-bit KV cache quantization into inference optimization pipelines, 

This work is licensed under a Creative Commons Attribution 4.0 International License. _PPoPP ’26, Sydney, NSW, Australia_ 

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2310-0/2026/01 https://doi.org/10.1145/3774934.3786428 

301 

Chengyu Sun, Yaqi Xia, Hulin Wang, Donglin Yang, Xiaobo Zhou, and Dazhao Cheng 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

**==> picture [181 x 128] intentionally omitted <==**

**Figure 1.** Efficiency and accuracy of quantization methods. 

demonstrating promising gains in efficiency. As serving systems strive for even higher compression ratios to support longer context lengths on resource-constrained GPUs, 2-bit KV cache quantization emerges as a compelling alternative. However, our analysis reveals a critical limitation: existing 2-bit quantization methods struggle to achieve an effective balance between accuracy and inference efficiency. 

Several efficiency-oriented quantization systems, such as Atom [53] and QServe [24], aim to enhance inference efficiency by using offline calibration to determine quantization parameters and integrating these parameters with weights to minimize runtime overhead. However, these methods exhibit unacceptable accuracy degradation when applied at 2-bit precision. We attribute this to two key factors. First, while per-token quantization is straightforward and efficient, it leads to interference between channels: outlier values in one channel can significantly distort the quantization of neighboring values. Second, offline calibration is unable to adapt to dynamic runtime variations across different input requests and sequence lengths, making it ill-suited for managing outliers during generation. 

To preserve model quality, accuracy-oriented quantization systems—SKVQ [12], KVQuant [15], and KIVI [27]—adopt advanced techniques including channel reordering, denseand-sparse quantization, and recent token reservation. However, our analysis shows that these methods often incur higher latency than FP16 baselines without quantization. We attribute this inefficiency to three key sources of overhead. First, these systems require explicit outlier detection and extraction prior to quantization, introducing additional computation and global memory access overhead. Second, the caching strategies used for recent token reservation are costly, introducing memory concatenation overhead during the decoding process. Third, they rely on a separate dequantization kernel prior to attention computation, adding further latency due to dequantization overhead. 

To this end, we propose JanusQuant, a 2-bit KV cache quantization system tailored for long-context LLM inference. As shown in Figure 1, JanusQuant leverages a co-design of algorithm and system to simultaneously improve accuracy 

and efficiency, thereby realizing the practical potential of 2-bit quantization in LLM inference. 

On the algorithmic front, JanusQuant introduces RtSmooth, a runtime per-token smoothing transformation that mitigates the impact of outliers. RtSmooth narrows intra-channel value gaps, enabling a more balanced representation across tokens. Unlike static calibration methods, RtSmooth dynamically adapts to KV cache variations across requests and sequence lengths, improving accuracy under diverse workloads. From a system perspective, JanusQuant builds on a key observation: the channel holding the absolute maximum (absmax) value per token is typically sparse and predictable. Leveraging this, we propose a fast absmax positioning technique that accesses fewer than 2% of channels to compute smoothing factors at runtime, significantly lowering outlier handling overhead. To further enhance memory efficiency, we design a new data structure for recent token reservation, which minimizes memory concatenation overhead during decoding. We also develop a custom mixed-precision attention kernel that fuses dequantization and attention computation, reducing kernel launch and memory access overhead. 

We implement JanusQuant on top of PyTorch [32] and Transformers [41], supporting LLMs such as the Llama, Mistral, Vicuna and Qwen families. JanusQuant is lightweight and easy to integrate into other models. To our knowledge, JanusQuant is the first system to fully leverage 2-bit KV cache quantization to achieve both high accuracy and efficient inference in practical long-context LLM deployment. Our contributions are summarized as follows: 

- Characterization of design limitations: We analyze the limitations of existing 2-bit KV cache quantization systems and identify key sources of overhead, including inefficient token caching, hardware-unfriendly outlier handling, and separate dequantization operations. 

- Quantization algorithm: We propose RtSmooth, a runtime per-token smoothing algorithm that adaptively mitigates outliers to reduce quantization difficulty. To minimize overhead, we introduce a fast absmax positioning technique that exploits the sparsity and regularity of per-token maxima. 

- System-level optimizations: We design a memoryefficient token cache and a custom mixed-precision attention kernel to address overheads in KV token management and computation. These system components co-optimize with RtSmooth to deliver high-speed inference. 

Experimental results on several representative LLMs—across varying model sizes and context lengths—show that JanusQuant delivers decoding speedups of 5.64× over FP16, 5.84× over KIVI, 4.45× over QServe, and 2.50× over DuoAttention. Additionally, JanusQuant reduces long-context KV cache memory consumption by 5.3× 

302 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

JanusQuant 

compared to FP16, enabling more efficient deployment on resource-constrained hardware. 

**Table 1.** Existing efficiency-oriented quantization systems fail to maintain models’ accuracy when extending their solutions to 2-bit precision. 

## **2 Background and Motivation** 

## **2.1 Generative Inference and KV Cache** 

LLMs typically adopt a decoder-only transformer architecture optimized for next-token prediction. The inference process consists of two stages: prefill and decoding. In the prefill stage, the model processes the entire input prompt to compute contextual representations and generate the first output token. In the decoding stage, LLMs iteratively generate one token at a time, using the most recent token along with previously generated tokens as input. Central to this process is the self-attention mechanism, which enables each token to incorporate contextual information from preceding tokens. The core computation is multi-head attention, defined as: 

**==> picture [213 x 26] intentionally omitted <==**

where the Query ( _𝑄_ ), Key ( _𝐾_ ), and Value ( _𝑉_ ) matrices are computed by projecting input tokens through corresponding learned weight matrices, and _ℎ_ denotes the number of attention heads. During the prefill stage, _𝑄_ , _𝐾_ , and _𝑉_ have the same dimensions, proportional to the prompt length. In the decoding stage, however, the attention computation involves only a single _𝑄_ (from the current token) and all previously computed _𝐾_ and _𝑉_ matrices. To avoid recomputing _𝐾_ and _𝑉_ at each decoding step, these matrices are cached in memory—known as the KV cache. This cache significantly accelerates inference but introduces memory and bandwidth challenges, especially for long sequences. 

## **2.2 Quantization Technique** 

While the KV cache reduces computation during decoding, its memory usage grows linearly with sequence length, becoming a major bottleneck for deploying LLMs on resource-constrained hardware. Quantization is an effective technique to compress the KV cache by mapping high-precision floating-point values to _𝑛_ -bit low-precision discrete integers. A typical quantization process involves the following steps: 

1. Compute quantization parameters—specifically, the scaling factor _𝑠_ and the zero point _𝑧_ : 

**==> picture [193 x 22] intentionally omitted <==**

2. Quantize the original tensor _𝑋_ to a low-precision _𝑄𝑥_ : 

**==> picture [181 x 21] intentionally omitted <==**

3. Dequantize _𝑄𝑥_ to an approximate high-precision _𝑋_[ˆ] : 

**==> picture [155 x 12] intentionally omitted <==**

Quantization can be applied at different levels of granularity. Per-tensor quantization uses a single scaling factor 

|Methods|Accuracy (Llama2-7Bperplexity) ↓|
|---|---|
|FP16|5.47|
|Atom[53]|4-bit: 5.93 / 2-bit: 103.05|
|QServe[24]|4-bit: 5.70 / 2-bit: 11.36|



**==> picture [217 x 101] intentionally omitted <==**

**Figure 2.** Magnitude of the KV cache on Llama2-13B. 

and zero point for the entire tensor. Per-channel quantization assigns distinct parameters to each column, while pertoken quantization does so for each row. A more fine-grained approach, per-group quantization, further subdivides each channel or token into groups, applying separate quantization parameters per group. We denote the group size as _𝑔_ . 

## **2.3 Challenges in 2-bit KV Cache Quantization** 

Numerous studies [3, 24, 35, 53] have employed 4-bit KV cache quantization as part of inference optimization strategies. However, as LLMs scale to longer context lengths, 4-bit quantization increasingly falls short on resource-constrained GPUs. For example, applying 4-bit group quantization with a group size of _𝑔_ = 128 to a 128K-token sequence results in 26.6GB of KV cache memory usage—still exceeding the capacity of an A100-40GB GPU when serving the Llama2-7B model, which already occupies 14GB of weights. 

To achieve higher compression, 2-bit quantization presents a compelling alternative. It not only enables support for longer contexts, but also offers greater theoretical speedups due to the memory-bound nature of attention operations. However, existing 2-bit KV cache quantization systems struggle to balance accuracy and efficiency, thereby hindering their practical deployment. 

**2.3.1 Accuracy Limitations of Prior Efficiencyoriented Methods.** Efficiency-oriented methods such as Atom [53] and QServe [24] apply per-token group quantization to the KV cache, leveraging row-major memory access patterns for faster execution. To reduce accuracy degradation caused by outliers, QServe introduces a per-channel smoothing transformation for the _𝐾_ cache, adapted from SmoothQuant [43]. Specifically, channels in 

303 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia Chengyu Sun, Yaqi Xia, Hulin Wang, Donglin Yang, Xiaobo Zhou, and Dazhao Cheng 

**==> picture [241 x 92] intentionally omitted <==**

**Figure 3.** The absmax value per channel varies with different requests and different sequence lengths. 

the _𝐾_ cache are scaled using smoothing factors _𝛾_ , such that _𝑄_ × _𝐾[𝑇]_ = ( _𝑄_ × Δ) × ( _𝐾_ × Δ[−][1] ) _[𝑇]_ , where Δ = _𝑑𝑖𝑎𝑔_ ( _𝛾_ ) and _𝛾 𝑗_ = _𝑚𝑎𝑥_ (| _𝐾𝑗_ |) _[𝛼] ,_ 0 ≤ _𝑗 < ℎ𝑖𝑑𝑑𝑒𝑛_  𝑠𝑖𝑧𝑒_ . To reduce runtime overhead, QServe precomputes Δ via offline calibration and integrates it into the preceding layer’s weights, modifying them as _𝑊𝑄_ = Δ _𝑊𝑄_ and _𝑊𝐾_ = Δ[−][1] _𝑊𝐾_ . While these methods maintain reasonable accuracy under 4-bit quantization, they fail to generalize to 2-bit precision. As shown in Table 1, the perplexity of Atom on Llama2-7B rises from 5.93 (4-bit) to 103.05 (2-bit). Although QServe performs better than Atom, its accuracy at 2-bit precision remains insufficient for practical use. Through further analysis, we identify two primary causes of accuracy loss in these systems under 2-bit quantization, as discussed next. 

**(1) Outlier channels amplify quantization error in grouped quantization.** As shown in Figure 2, the _𝐾_ cache exhibits strong outliers concentrated in some channels, whereas the _𝑉_ cache does not. In per-token group quantization, outliers in one channel can significantly increase the quantization error of neighboring values. To quantify this effect, we compute the mean square error (MSE) of the _𝐾_ cache under 2-bit quantization in three settings: (a) Atom, (b) QServe, and (c) an ideal case where the most extreme outlier channel in each group is replaced with a non-outlier channel before applying QServe. On randomly selected WikiText2 sentences, the MSEs are 1.0352 (Atom), 0.5552 (QServe), and 0.3734 (ideal case). These results show that while QServe’s smoothing transformation reduces the error gap caused by outliers compared to Atom, it does not fully mitigate the interference—outlier channels continue to inflate the quantization error of other values in the same group. 

**(2) Offline calibration fails to adapt to runtime dynamics of the KV cache.** QServe relies on offline calibration to compute per-channel smoothing factors using a predefined dataset. However, because these factors are derived solely from static channel-wise absmax values, they cannot adapt to variations across input requests or sequence lengths. To evaluate this, we sample random sentences of varying lengths and measure the per-channel absmax values, as shown in Figure 3. It is observed that these absmax values fluctuate significantly—sometimes by more than 4× 

**==> picture [241 x 62] intentionally omitted <==**

**Figure 4.** Workflow of accuracy-oriented 2-bit KV cache quantization methods. 

across requests and sequence lengths. Because the smoothing factors are fixed at calibration time, they fail to reflect these runtime shifts, resulting in reduced quantization accuracy, especially at 2-bit precision. Notably, although our results are shown for fixed-length sequences, this mismatch becomes more severe with increasing context lengths, where KV cache variance is even more pronounced. 

**2.3.2 Efficiency Limitations of Prior Accuracyoriented Methods.** Several studies have explored strategies to preserve accuracy in 2-bit KV cache quantization. For example, SKVQ [12] uses channel reordering to group channels with similar distributions. KVQuant [15] adopts a dense-and-sparse quantization approach, isolating outliers for separate sparse attention computation. KIVI [27], on the other hand, stores a fixed number of recent tokens in FP16 precision. As illustrated in Figure 4, these systems generally follow a similar multi-step pipeline: (1) retain recent tokens in FP16; (2) detect and extract outliers; (3) quantize the remaining KV cache; (4) dequantize before attention; and (5) perform attention computation. Despite their accuracy benefits, these methods suffer significant performance bottlenecks, which we attribute to three sources. 

**(1) Outlier handling overhead.** Outliers must be detected and handled prior to quantization. Systems like SKVQ [12] and KVQuant [15] first identify outliers among generated tokens, then reorder channels to group them by value range into variable-sized segments. These operations introduce non-trivial overhead—and because these steps are memory-bound, the cost grows linearly with sequence length. 

**(2) Caching overhead.** Reserving recent tokens in FP16 is critical for preserving accuracy, especially in long-context inference where attention prioritizes recent information. Prior works such as SKVQ [12] and KIVI [27] adopt a slidingwindow caching strategy that appends new tokens to the end and removes the oldest from the front. The evicted token is then quantized and added to the low-precision KV cache. However, this approach incurs extra memory overhead, as each append and remove operation during decoding requires tensor concatenation, which is costly and frequent. 

**(3) Dequantization overhead.** SKVQ [12] performs outlier extraction and rearrangement across hidden dimensions, which conflicts with the multi-head attention mechanism. Since attention heads are computed independently but may 

304 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

JanusQuant 

**==> picture [241 x 88] intentionally omitted <==**

**Figure 5.** Runtime breakdown of Llama2-7B attention layer. 

**==> picture [241 x 102] intentionally omitted <==**

**Figure 6.** An example of RtSmooth algorithm, suppose _𝑔_ = 4. 

share quantization groups, the dequantization kernel cannot be fused with attention computation. This results in a separate dequantization step and adds memory-bound overhead that scales linearly with sequence length. 

To further illustrate the overhead, we analyze the attention layer kernels of the SKVQ baseline [12]. Figure 5 shows the runtime breakdown for the Llama2-7B attention layer. The results reveal that the combined overhead from outlier handling, recent token reservation and separate dequantization accounts for over 85% of total runtime in the prefill stage (20% / 20% / 45%) and over 97% in the decoding stage (2% / 15% / 80%). These costs significantly limit end-to-end inference efficiency, especially during decoding. 

**Summary.** Previous attempts at 2-bit KV cache quantization face fundamental trade-offs. Efficiency-oriented methods avoid runtime outlier handling through offline calibration but suffer significant accuracy loss, while accuracyoriented methods rely on recent token reservation and complex outlier handling to preserve accuracy but remain less efficient than FP16 inference. These challenges call for runtimeaware approaches that mitigate outlier effects across diverse requests. Moreover, advancing end-to-end performance requires lightweight outlier detection, memory-efficient data structures for the cache, and fused mixed-precision attention kernel to reduce deployment costs. 

## **3 JanusQuant Design and Implementation** 

We introduce JanusQuant, a system that unifies an outlieraware quantization algorithm with a set of system-level optimizations to enable accurate and efficient 2-bit KV cache quantization. JanusQuant is co-designed across algorithm and system: its core component, RtSmooth, dynamically adapts to outlier patterns at runtime, while the supporting architecture minimizes memory and compute overhead through lightweight data structures and fused computation. Together, these components enable JanusQuant to deliver both high compression and high accuracy, with practical performance gains for long-context LLM inference. 

## **3.1 RtSmooth: An Outlier-Aware Algorithm** 

As discussed in our motivation analysis, accurate outlier handling in the _𝐾_ cache is essential for maintaining model accuracy under 2-bit quantization. Our design is based on 

three key insights: (1) Using per-channel quantization for the _𝐾_ cache helps localize outlier effects, since outliers are typically concentrated in some channels—unlike per-token quantization, where grouped values from different channels are affected. (2) To adapt to dynamic runtime conditions (e.g., different requests or sequence lengths), outlier handling must occur at runtime. (3) Due to the locality bias in the attention mechanism, preserving a small number of recent KV tokens in FP16 is important during decoding. Based on these observations, we propose RtSmooth, an outlieraware quantization algorithm. Specifically, RtSmooth applies per-token smoothing transformation alongside per-channel group quantization for the _𝐾_ cache, and uses per-token group quantization for the _𝑉_ cache. During decoding, it maintains FP16 precision for newly generated KV tokens and performs quantization on the buffered FP16 tokens every _𝑔_ steps. 

RtSmooth dynamically determines smoothing factors at runtime for each request to improve accuracy. As illustrated in Figure 6, it first computes the smoothing factor for each token as _𝑚𝑎𝑥_ (| _𝐾𝑖_ |) _[𝜆]_ , where _𝜆_ = 0 _._ 5 is selected empirically based on experimental results. The computed factors are then applied to scale each token’s values. This runtime smoothing transformation reduces the value range within each quantization group, narrowing the gap between the maximum and minimum values, thereby reducing quantization error. Let _𝜖𝑔𝑝_ denote the quantization error for values in a group _𝑔𝑝_ . Its upper bound is defined by the quantization scale: 

**==> picture [186 x 22] intentionally omitted <==**

Reducing the quantization scale _𝑠𝑔𝑝_ leads to smaller quantization error, meaning dequantized values more closely approximate their original floating-point values. After smoothing, the value range shrinks, resulting in a lower error bound: 

**==> picture [185 x 20] intentionally omitted <==**

While RtSmooth maintains high accuracy in 2-bit quantization, it introduces additional processing overhead during both quantization and dequantization. The JanusQuant system integrates lightweight architectural optimizations to hide this overhead and enable efficient deployment. 

305 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia Chengyu Sun, Yaqi Xia, Hulin Wang, Donglin Yang, Xiaobo Zhou, and Dazhao Cheng 

**==> picture [217 x 113] intentionally omitted <==**

**Figure 7.** Workflow of JanusQuant. 

**==> picture [229 x 69] intentionally omitted <==**

**Figure 8.** Left: the absmax value and outlier value distribution of the 2-nd layer _𝐾_ cache of Llama2-13B model. Right: the ratio of the calibrated channel set size relative to the total number of channels. The input data comes from WikiText2. 

## **3.2 System-level Optimizations** 

Leveraging the RtSmooth algorithm, JanusQuant synergizes three key attention-layer optimizations (Figure 7). For quantization, it introduces a fast absmax positioning technique to reduce the overhead of runtime outlier handling. For caching, it employs a memory-efficient data structure that transitions seamlessly from FP16 KV buffering to low-bit quantization. For dequantization, it implements a custom mixed-precision attention kernel that fuses dequantization and attention computation to minimize processing overhead. 

**3.2.1 Lightweight Quantization via Fast Absmax Value Positioning.** Figure 8 (left) presents the distribution of absmax and outlier[1] values. Outliers exhibit dynamic variation across requests and decoding steps, incurring substantial overhead for direct identification. In contrast, the absmax values exhibit a layer-wise pattern of sparsity and regular concentration: a small subset of channels consistently accounts for the absmax values across tokens. This insight motivates our fast absmax value positioning technique (FAVP), which combines offline calibration with lightweight runtime computation to reduce overhead. With a one-time calibration (only a few minutes) before deployment, we identify the most likely absmax channels for each layer. Figure 8 (right) shows that these channels remain stable and sparse across 128 random 8K-length WikiText2 samples, with over 90% of layers involving fewer than 2% of all channels. 

> 1Following prior works [6], values greater than 6 are identified as outliers. 

**==> picture [242 x 139] intentionally omitted <==**

= **Figure 9.** Two methods of recent token reservation ( _𝑔_ 4). 

At runtime, we restrict the smoothing factor computation to only these channels, dramatically reducing memory access. Without this technique, computing the absmax across all channels incurs substantial overhead. In fact, as shown in Figure 15a, more than 80% of the quantization kernel’s overhead stems from absmax calculation. Our technique significantly mitigates this bottleneck. To further improve efficiency, we implement a fused quantization operator that integrates the smoothing transformation with quantization parameter (scaling factor and zero point) computation and reorganization, and KV cache quantization. Note that the parameter reorganization—an optimization for attention calculation—is detailed in Section 3.2.3. 

**3.2.2 Token Reservation via Memory-efficient Data Structure.** While prior works such as KIVI [27] preserve model accuracy by reserving several recent tokens in FP16 precision, they often rely on inefficient caching mechanisms that degrade inference speed. As illustrated in Figure 9, these implementations use memory-intensive tensor concatenation to sequentially store recent tokens, quantizing the entire group once a fixed size _𝑔_ is reached[2] . This approach incurs global memory movement at each decoding step, as concatenation triggers memory reallocation and data copying. 

To mitigate the associated overhead, a straightforward approach involves maintaining a fixed cache capable of accommodating _𝑔_ tokens, thereby reducing the frequency of memory allocation and copying. While this approach improves efficiency, it may not adequately account for the effects on accuracy. In the current implementation, not all decoding steps benefit from the accuracy improvements associated with recent token reservation. For example, the cached recent tokens are reset to zero after each quantization, means that the next decoding will only have one newly generated token with FP16 precision; while ideally, reserving at least 32 tokens for each decoding would improve accuracy [14]. 

The above observations motivate the development of a novel data structure that optimizes the trade-off between 

> 2Following prior works [12, 27], we set _𝑔_ = 32 for 2-bit quantization. 

306 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

JanusQuant 

**==> picture [241 x 212] intentionally omitted <==**

**Figure 10.** Kernel timeline and optimization strategies. 

memory usage and accuracy by appropriately increasing memory allocation. The design centers on piecewise quantization, which separates the tokens involved in the quantization process from at least the most recent _𝑔_ tokens. As shown in Figure 9, in this example, the cache is fixed to hold 2 _𝑔_ tokens, with newly generated tokens alternately assigned to the first or second half of the cache. Upon reaching full capacity, the system automatically quantizes the older half and transfers them to the quantized KV cache. 

Despite this cache size being sufficient to improve accuracy in experiments, JanusQuant implements efficient caching based on a ring buffer data structure, enhanced with pointers to manage critical information, such as the location of the next token, the decision to perform quantization, and the identification of the next segment to be quantized. This approach facilitates user-defined cache size as any integer multiples of _𝑔_ , thereby accommodating diverse accuracy requirements. Specifically, when the cache accommodates _𝑛𝑔_ tokens, each decoding process retains at least ( _𝑛_ − 1) _𝑔_ tokens. 

Furthermore, the cache introduces no additional overhead since it is pre-allocated prior to inference, and the extra memory consumption remains negligible, especially in longcontext reasoning scenarios. For example, when processing 128K tokens, the additional memory reserved for full precision (assuming the cache holds 2 _𝑔_ tokens) amounts to only 0.05% of the total token count. 

**3.2.3 Mixed-Precision Attention Kernel Adapted for RtSmooth.** The RtSmooth algorithm preserves the positional alignment of outliers, facilitating the fusion of dequantization and attention for the 2-bit KV cache, while maintaining full-precision attention for recent KV tokens. To exploit this property, we design a custom mixed-precision 

attention kernel that hides dequantization overhead by processing both quantized and FP16 KV entries in a single pass. As illustrated in Figure 10a, the kernel leverages task parallelism, assigning different workloads to separate CUDA thread blocks. We incorporate asynchronous execution to overlap computation and memory access, improving runtime efficiency. Additionally, we apply two levels of optimization to reduce compute intensity and increase memory bandwidth utilization in the fused kernel. 

First, using the Roofline model generated by Nsight Compute, we observe that the attention kernel—typically memory-bound during decoding—becomes compute-bound due to the high arithmetic intensity of dequantization. Addressing this requires reducing the computational cost of integer-to-floating-point conversion. Inspired by prior work on 8-bit and 4-bit dequantization [17], we implement an efficient INT2-to-FP16 unpacking and type-casting method. The approach leverages a key insight: FP16 values between 1024 and 2047 share an exponent of 1024, while their mantissa encodes the integer offset by 1024. Thus, for integers in the range [0, 1023], we can synthesize FP16 representations by placing the integer directly in the mantissa and setting the exponent to 1024. As shown in Figure 10b, we use two 32-bit registers, R1 and R2, where R1 holds 16 2-bit quantized values. Each unpacking operation processes two 2-bit values, which are extracted into R2 and interpreted as INT16. A bitwise OR operation is applied: _𝑅_ 2 = _𝑅_ 2|0 _𝑥_ 64006400. This operation sets the exponent field to 1024, effectively treating the INT16 values as FP16 numbers. Finally, subtracting 1024 from each number yields the final FP16 values. Repeating this process across all data in R1 completes the conversion. This method uses only three instructions—lop3, or, and sub—to unpack and type-cast two values. By contrast, a naive implementation requires at least four instructions per value. 

Second, we identify that parameter loading during dequantization suffers from inefficient memory bandwidth utilization. As illustrated in Figure 10c, a naive layout stores different parameter types (e.g., scaling factor and zero point) separately, with varying shapes determined by the quantization dimensions of the KV cache. This fragmentation leads to redundant memory transactions. For example, loading 64 bytes (32 × _𝑠𝑖𝑧𝑒𝑜𝑓_ ( _𝐹𝑃_ 16)) of scaling factors for the _𝑉_ cache may require up to eight memory transactions, despite only 8 bytes being valid in each. Given the minimum transaction width of 32 bytes, this results in 75% bandwidth waste. To address this, JanusQuant introduces a unified parameter block layout, in which four types of parameters are coalesced into a single structure aligned with the memory access patterns of each CUDA thread block. The size of each parameter block is matched to the KV cache segment processed by the thread block. This reorganization significantly improves bandwidth efficiency; in the example shown, the number of memory transactions is reduced from 20 to 8. 

307 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia Chengyu Sun, Yaqi Xia, Hulin Wang, Donglin Yang, Xiaobo Zhou, and Dazhao Cheng 

**Table 2.** Accuracy results on LongBench long-context multitasks for LLMs. Higher values indicate better performance. 

|Model|Method|Task|Task|Task|Task|Task|Task|Task|Task|Task|
|---|---|---|---|---|---|---|---|---|---|---|
|||LCC|TriviaQA|RepoBench-P|QMSum|SAMSum|MultiNews|Qasper|TREC|Avg|
|Llama<br>2-7B<br>-32K|FP16|51.75|86.21|51.78|22.02|45.18|27.42|29.04|72.00|48.18|
||Atom-4bit|50.29|84.74|50.13|23.34|44.89|26.77|29.52|70.50|47.52|
||Atom-2bit|4.09|0.00|3.58|0.17|0.00|0.07|0.01|0.00|0.99|
||SKVQ|50.28|85.99|50.05|23.97|43.88|24.73|24.94|71.00|46.86|
||KIVI|50.25|83.21|49.40|19.75|45.59|26.85|25.15|72.00|46.53|
||JanusQuant|51.66|85.32|51.20|23.43|44.75|26.79|27.08|71.50|47.72|
|Llama<br>3-8B<br>-1048K|FP16|38.06|87.70|38.25|25.18|42.49|28.34|27.90|71.00|44.87|
||Atom-4bit|36.72|86.43|38.60|24.49|41.83|27.75|27.96|70.00|44.22|
||Atom-2bit|2.43|0.00|2.91|0.00|0.00|0.00|0.00|0.00|0.67|
||SKVQ|34.06|85.88|36.24|25.55|40.26|27.69|26.28|70.76|43.34|
||KIVI|34.96|87.43|36.78|24.95|42.47|27.78|27.67|71.50|44.19|
||JanusQuant|35.32|87.23|39.23|25.34|42.19|28.31|27.44|71.50|44.57|



By combining these architectural optimizations, JanusQuant effectively hides system-level overhead and enables efficient inference, addressing the challenges in our analysis. 

## **3.3 Implementation** 

JanusQuant is implemented with approximately 3,500 lines of CUDA/ C++ and 2,500 lines of Python, supporting practical LLM inference. Specialized CUDA kernels are developed for: (1) processing the memory-efficient token cache, (2) performing fused smoothing and quantization, and (3) executing the fused dequantization-attention operation. These kernels can be compiled into a standalone shared library (.so), and are also wrapped as Python extensions using Pybind and FlashInfer [7, 46], enabling compatibility with Python-based serving frameworks such as PyTorch and Transformers. The Python interface includes a custom attention module that inherits from the PyTorch neural network API. This design supports mainstream LLMs—including Llama, Mistral, Vicuna and Qwen—and allows seamless integration of additional models. By exposing a PyTorch-compatible attention module, JanusQuant integrate seamlessly into multi-stream inference engines, improving GPU utilization and decoding throughput. The artifact will be released as open-source repository to support further research and adoption. 

## **4 Evaluation** 

## **4.1 Experimental Setup** 

**Testbed.** All experiments are conducted on a machine equipped with four NVIDIA A100-PCIE-40GB GPUs. Note all efficiency-related experiments, including both end-to-end and kernel-level evaluations, are performed on a single A100 GPU. The software stack includes PyTorch 2.4.0 and CUDA 12.6. 

**Baselines.** We compare JanusQuant against several opensourced, state-of-the-art KV cache quantization methods, 

including KIVI [27], SKVQ [12], KVQuant [15], Atom [53] and QServe [24]—as well as DuoAttention [44], a leading KV cache selection method, to comprehensively assess performance improvements by JanusQuant. 

**Models, datasets and metrics.** We evaluate on a variety of models, including Llama, Mistral, Vicuna and Qwen families [16, 28, 37]. Throughout our experiments, we adhere to the native data types defined in each model’s official release. Specifically, we use torch.float16 for models such as Llama30B, Llama-2-7B/13B, and Vicuna-7B/13B, while employing torch.bfloat16 for Llama-3-8B, Mistral-7B, and Qwen-2.532B. The evaluation datasets include WikiText2 and C4, and the KV tokens are generated through actual model inference using these datasets. Following established methodologies from prior studies [3, 12, 23, 27, 53], we evaluate model accuracy using the perplexity metric and the LongBench benchmark [4]. For efficiency, we assess both kernel-level performance and end-to-end model execution. Our primary metrics are kernel runtime and serving latency. For kernellevel experiments, results are averaged over 10,000 runs after 100 warm-up runs. For end-to-end evaluation, results are averaged over 100 runs following 10 warm-up runs. Note that we select baselines based on the primary focus of each metric. For accuracy evaluations, we compare against SKVQ, KVQuant, and KIVI to cover state-of-the-art 2-bit schemes. However, for latency benchmarks, we exclude SKVQ and KVQuant as their current implementations prioritize theoretical algorithm over end-to-end inference speed. Instead, we compare against KIVI and the 4-bit optimized system, QServe, to demonstrate our method’s practical efficiency. 

## **4.2 Accuracy Evaluation** 

**Settings.** We compare our method against KIVI and SKVQ on LongBench tasks. Note that KVQuant is evaluated only on perplexity benchmarks, as its official implementation is 

308 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

JanusQuant 

**Table 3.** Perplexity results on WikiText2 with sequence length 2048. Lower values indicate better performance. 

|Model<br>Method|FP16|RTN|KVQuant|JanusQuant|
|---|---|---|---|---|
|Llama-30B|4.10|4.88|OOM|4.27|
|Llama2-7B|5.47|8.31|5.99|5.80|
|Llama2-13B|4.88|6.43|5.31|5.12|
|Llama3-8B|6.14|17.14|7.42|6.93|
|Mistral-7B|5.94|7.55|6.32|6.21|



**Table 4.** Perplexity on WikiText2 with sequence length 2048 under different methods at 2-bit precision. 

|Method|Llama2-7B|Llama2-13B|
|---|---|---|
|QServe|11.36|7.08|
|RtSmooth w/o. FAVP<br>Ofine calibration|7.41|6.21|
|RtSmooth w/o. FAVP<br>Runtime calculation|5.85|5.16|
|RtSmooth w/. FAVP|5.85|5.16|
|RtSmooth w/. FAVP<br>+ Recent token reservation|5.80|5.12|



**==> picture [241 x 65] intentionally omitted <==**

**Figure 11.** Evaluation of predicted-to-actual absmax ratios on 8K decoded C4 tokens. 

designed for fixed-length contexts (requiring consistent calibration and testing lengths), which is incompatible with the dynamic sequence lengths characteristic of LongBench. To enable a fairer comparison, we implement a 2-bit fake quantization variant of Atom, adapted from its original KV cache quantization strategy. We also include a Round-To-Nearest (RTN) baseline, which applies per-token group quantization to the KV cache, to assess the incremental gains from our outlier-aware design. The group size parameter _𝑔_ is set to 128 for 4-bit quantization and 32 for 2-bit quantization. The cache size of recent tokens is fixed to hold 2 _𝑔_ tokens while this size is user-customizable. For calibration, we use 128 samples of 8K-length sequences drawn from the WikiText2 training set. 

**Main results.** To evaluate long-context understanding and reasoning, we report LongBench results on eight commonsense tasks (Table 2). JanusQuant outperforms all 2-bit baselines; for example, it improves Qasper accuracy by 6.65% and 7.37% over KIVI and SKVQ, respectively. Relative to 

**Table 5.** Average bit-width of different methods. 

|Method(2-bit)|KVQuant|SKVQ|KIVI|
|---|---|---|---|
|Avg. Num. Bits|2.320|3.000|3.000|
|Method(2-bit)|Atom|QServe|JanusQuant|
|Avg. Num. Bits|3.000|3.001|3.008|
|Method(4-bit)|Atom|QServe|-|
|Avg. Num. Bits|4.250|4.251|-|



**Table 6.** Memory usage (GB) of the KV cache. 

|Model<br>Method|FP16|DuoAttention|JanusQuant|
|---|---|---|---|
|Llama2-7B<br>L=128K|62.50|15.74|11.75|
|Llama3-8B<br>L=448K|54.69|13.70|10.29|
|Llama2-13B<br>L=64K|48.83|12.39|9.20|



FP16, JanusQuant incurs only minor accuracy loss (0.95% on Llama2-7B-32K and 0.66% on Llama3-8B-1048K). For the Llama2-7B-32K model, JanusQuant delivers a 0.2 increase in average scores compared to the 4-bit Atom, while Atom’s 2- bit variant incurs unacceptable accuracy degradation. These results indicate that JanusQuant can maintain over 99% of the accuracy achieved compared to FP16. Table 3 presents a comparison of perplexity across JanusQuant and other baselines. Note that OOM (out-of-memory) indicates that the KVQuant method exceeds available GPU memory during quantization. JanusQuant consistently outperforms KVQuant, achieving perplexity reductions of up to 0.49. 

**Breakdown analysis.** We conduct a breakdown analysis to evaluate the contribution of different quantization techniques to model accuracy. As shown in Table 4, we report perplexity on the WikiText2 dataset at each analysis stage. QServe is included as a reference baseline. Our findings show that combining offline per-token smoothing transformation with per-channel quantization significantly improves model accuracy, reducing perplexity by 3.95 on the Llama27B model. Applying smoothing transformation at runtime yields further gains. Moreover, introducing FAVP technique maintains accuracy comparable to full runtime computation. Finally, when combined with recent token reservation, the approach achieves an additional perplexity improvement. 

We further assess the effect of FAVP on accuracy with different types of LLMs. Figure 11 shows the ratio between the predicted and actual absmax values per token using the fast positioning technique. Although exact matches are not always achieved, deviations occur in only a minimal fraction of tokens. As reflected in end-to-end perplexity, this deviation has no negative impact on overall model accuracy. 

309 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia Chengyu Sun, Yaqi Xia, Hulin Wang, Donglin Yang, Xiaobo Zhou, and Dazhao Cheng 

**==> picture [505 x 153] intentionally omitted <==**

**Figure 12.** The latency under different sequence lengths with different models. Dotted lines represent OOM (out-of-memory), and the results are linearly extrapolated from measured data. 

**==> picture [241 x 72] intentionally omitted <==**

**Figure 13.** The end-to-end GPU time to finish all inference requests in the workload. 

## **4.3 End-to-end Evaluation** 

**End-to-end latency.** We evaluate the latency of five methods: Transformers baseline with FP16 FlashAttention-2 (FA2) backend, KIVI (2-bit), QServe (4-bit), DuoAttention (KV selection), and JanusQuant (proposed 2-bit), across models with 7B, 8B, and 13B parameters. To highlight the performance advantages of JanusQuant, we report latencies for the prefill and decoding stages separately. Note that during the decoding stage, we preallocate the KV cache to eliminate the overhead associated with dynamic memory allocations. 

As shown in the top row of Figure 12, JanusQuant does not achieve significant speedup over FA2 during the prefill stage. This is expected, as JanusQuant utilizes original full-precision KV tokens for attention in this phase to preserve accuracy. In contrast, QServe benefits from weightactivation quantization in the linear layers, and DuoAttention avoids quantization entirely. Nonetheless, JanusQuant delivers speedups of 1.11×, 1.38×, and 1.17× over KIVI during prefill across the three model sizes. 

In the bottom row of Figure 12, we present the total latency for decoding 100 tokens. JanusQuant consistently outperforms all baselines in this stage. Importantly, as sequence length increases, FA2 and QServe encounter out-of-memory (OOM) errors, while JanusQuant continues to operate reliably on memory-constrained hardware. Overall, JanusQuant 

achieves up to 5.64× decoding speedups over FA2, with additional gains of 5.84×, 4.45×, and 2.50× over KIVI, QServe, and DuoAttention, respectively. 

**GPU time.** Figure 13 reports the end-to-end GPU time for completing all real inference requests on LongBench. In this experiment, we employ Vicuna-7B and Vicuna-13B to demonstrate the versatility of JanusQuant across different models. Given that the model sizes are comparable to those in our previous experiments, we anticipate consistent speedup trends. Unlike the comprehensive benchmarks in Figure 12, this experiment serves as a supplementary study illustrating JanusQuant’s seamless integration into the serving engine (Transformers) and the resulting throughput improvements for real-world requests. As such, several baselines lack implementation support for such serving scenarios and are consequently omitted. 

The evaluation covers multi-doc QA, single-doc QA, summarization, and synthetic tasks. Each task consists of 200 requests with input lengths ranging from 2K to 44K and output lengths up to 128, where both methods are controlled to generate outputs of the same length. JanusQuant delivers notable speedups of 1.52× and 1.70× on Vicuna-7B and Vicuna-13B, respectively, over FA2, demonstrating substantial practical benefits. 

**Memory usage.** Table 5 summarizes the average bitwidth of each quantization scheme. Note that the group size is set to 128 for 4-bit quantization and 32 for 2-bit quantization. KVQuant exhibits an advantage in memory efficiency by employing a sophisticated non-uniform quantization scheme that reduces quantization parameters via table lookups. In contrast, our method, JanusQuant, incurs a slight storage overhead compared to other uniform integer quantization methods due to the additional smoothing factors stored by RtSmooth. Nevertheless, JanusQuant still achieves a 1.41× memory reduction compared to 4-bit schemes. 

To further evaluate the benefits of 2-bit quantization over token selection methods, Table 6 presents the total KV cache 

310 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

JanusQuant 

memory usage. JanusQuant reduces memory consumption by 5.30× relative to FP16, facilitating longer sequence inference within the same GPU memory budget. By comparison, DuoAttention achieves a 3.95× memory reduction relative to FP16, but pushing beyond that limit results in significant accuracy loss. These results highlight JanusQuant’s ability to achieve substantial memory savings without compromising accuracy, making it well-suited for long-context inference on resource-constrained hardware. 

## **4.4 Kernel Evaluation** 

**Kernel performance.** We evaluate the performance of attention kernels during decoding across various methods, including SKVQ, KIVI, QServe, FA2, and our proposed method, JanusQuant. Figure 14 shows the relative performance of each kernel, normalized to JanusQuant. The results show that JanusQuant consistently outperforms all baselines across a range of settings. For example, with a sequence length of 128K, hidden size of 4096, and 32 KV heads, JanusQuant achieves speedups of 6.17× and 1.69× over the KIVI and QServe kernels, respectively. Compared to FA2, JanusQuant demonstrates an average improvement of 1.99×. SKVQ, in contrast, shows limited gains due to its reorganization of KV channels, which prevents fusing dequantization into the attention kernel—thereby introducing inefficiencies. 

**Breakdown analysis.** For the quantization kernel, we evaluate the performance improvement introduced by the fast positioning technique. A naive quantization kernel without smoothing is used as a baseline for comparison. For the attention kernel, we assess the impact of two optimizations: (1) efficient unpacking and INT2-to-FP16 type-casting, and (2) parameter reorganization. We include FA2 and a naive mixed-precision kernel as baselines. 

Figure 15a reports the runtime of three quantization kernels, highlighting that adding runtime smoothing introduces significant overhead. For instance, at a sequence length of 64K, it increases the quantization kernel’s runtime by 4.43×. However, the optimized RtSmooth kernel, equipped with the FAVP technique, effectively mitigates this overhead while maintaining accuracy. 

Figure 15b presents the runtime of the naive mixedprecision attention kernel alongside two optimized variants. The unoptimized version shows no performance gain over FA2. In contrast, integrating efficient unpacking and INT2-to-FP16 type-casting (to reduce compute intensity), along with parameter reorganization (to improve memory bandwidth utilization), yields substantial speedups. These two optimizations achieve average improvements of 1.99× and 3.05×, respectively, relative to the naive baseline. 

## **5 Related Work** 

**KV cache quantization** has been explored to reduce memory consumption and improve inference throughput. 

**==> picture [242 x 90] intentionally omitted <==**

**Figure 14.** Attention kernel performance. 

**==> picture [117 x 71] intentionally omitted <==**

**==> picture [117 x 71] intentionally omitted <==**

**==> picture [174 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)  Quantization (b)  Attention<br>**----- End of picture text -----**<br>


**Figure 15.** Breakdown analysis of the quantization and attention kernels on Llama2-7B model. 

Prior works such as Atom [53], FlexGen [35], QuaRot [3], and QServe [24] demonstrate effective 4-bit quantization, but suffer severe accuracy degradation at sub-4-bit precision. Recent efforts, including KIVI [27], SKVQ [12], KVQuant [15], and WKVQuant [48], investigate 2-bit quantization to achieve higher compression, yet remain inefficient compared to FP16, limiting practical deployment. In contrast, JanusQuant achieves both high accuracy and efficiency, while remaining compatible with other optimizations such as weight quantization [10, 13, 23, 52] and weight–activation quantization [6, 25, 40, 42, 47, 51] in linear layers for efficient inference. 

**KV cache selection** reduces memory usage by identifying and evicting or offloading less important tokens. Methods such as H2O [50] and Scissorhands [26] rely on attention scores to retain a fixed set of critical tokens while discarding others. FastGen [14], SnapKV [21], and DuoAttention [44] exploit head-specific attention patterns to statically prune unused cache segments. More recently, InfiniGen [20] adopts a dynamic strategy, offloading the cache to DRAM and prefetching tokens to the GPU as needed during decoding. 

**KV cache management** has been a central focus of serving systems for transformer architectures [2, 5, 11, 19, 30, 36, 39], aiming to reduce memory usage. vLLM [18] alleviates fragmentation via PagedAttention, while vAttention [33] offers a simpler alternative using CUDA virtual memory for coherence. Mooncake [34] expands cache management with a global scheduler spanning CPU, DRAM, SSD, and NIC resources. Yet token capacity remains limited by physical memory. We plan to explore integrating JanusQuant into these systems to mitigate this constraint. 

311 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

Chengyu Sun, Yaqi Xia, Hulin Wang, Donglin Yang, Xiaobo Zhou, and Dazhao Cheng 

## **6 Conclusion** 

We present JanusQuant, an accurate and efficient 2-bit KV cache quantization system for long-context LLMs. Our approach combines the RtSmooth runtime smoothing algorithm with three system-level optimizations—a fast absmax positioning method, a memory-efficient token cache structure, and a custom mixed-precision attention kernel. Through this algorithm–system co-design, JanusQuant delivers up to 5.3× KV cache compression and outperforms FP16, KIVI, QServe, and DuoAttention in decoding speed, while preserving long-context accuracy. 

## **Acknowledgments** 

We would like to thank the anonymous reviewers and our shepherd, Tim Harris, for their constructive comments and guidance which significantly improved the quality of this paper. This work was supported by National Key Research and Development Program of China (2023YFE0205700), National Natural Science Foundation of China (62341410), and the Science and Technology Development Fund, Macao S.A.R (FDCT) projects 0078/2023/AMJ and 001/2024/SKL. 

## **References** 

- [1] OpenAI (2023). 2023. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ (2023). 

- [2] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, and Ramachandran Ramjee. 2024. Taming Throughput-Latency tradeoff in LLM inference with Sarathi-Serve. In _18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)_ . 117–134. 

- [3] Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. 2025. Quarot: Outlier-free 4-bit inference in rotated llms. _Advances in Neural Information Processing Systems (NeurIPS)_ 37 (2025), 100213–100240. 

- [4] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2024. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL) (Volume 1: Long Papers)_ . 3119–3137. 

- [5] Branden Butler, Sixing Yu, Arya Mazaheri, and Ali Jannesari. 2024. PipeInfer: Accelerating LLM Inference using Asynchronous Pipelined Speculation. In _International Conference for High Performance Computing, Networking, Storage and Analysis (SC)_ . IEEE, 1–19. 

- [6] Yidong Chen, Chen Zhang, Rongchao Dong, Haoyuan Zhang, Yonghua Zhang, Zhonghua Lu, and Jidong Zhai. 2024. MixQ: Taming Dynamic Outliers in Mixed-Precision Quantization by Online Prediction. In _International Conference for High Performance Computing, Networking, Storage and Analysis (SC)_ . IEEE, 1–15. 

- [7] Tri Dao. 2024. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. In _The Twelfth International Conference on Learning Representations (ICLR)_ . 

- [8] DeepSeek-AI. 2024. Deepseek-v3 technical report. _arXiv preprint arXiv:2412.19437_ (2024). 

- [9] DeepSeek-AI. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. _arXiv preprint arXiv:2501.12948_ (2025). 

- [10] Tim Dettmers, Ruslan A Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, and Dan Alistarh. 2024. SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression. In _The Twelfth International Conference on Learning Representations (ICLR)_ . 

- [11] Jiangsu Du, Jinhui Wei, Jiazhi Jiang, Shenggan Cheng, Dan Huang, Zhiguang Chen, and Yutong Lu. 2024. Liger: Interleaving intra-and inter-operator parallelism for distributed large model inference. In _Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP)_ . 42–54. 

- [12] Haojie Duanmu, Zhihang Yuan, Xiuhong Li, Jiangfei Duan, Xingcheng Zhang, and Dahua Lin. 2024. Skvq: Sliding-window key and value cache quantization for large language models. _Conference on Language Modeling (COLM)_ (2024). 

- [13] Elias Frantar, Roberto L Castro, Jiale Chen, Torsten Hoefler, and Dan Alistarh. 2025. Marlin: Mixed-precision auto-regressive parallel inference on large language models. In _Proceedings of the 30th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP)_ . 239–251. 

- [14] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. 2024. MODEL TELLS YOU WHAT TO DISCARD: ADAPTIVE KV CACHE COMPRESSION FOR LLMS. In _12th International Conference on Learning Representations (ICLR)_ . 

- [15] Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W Mahoney, Sophia Shao, Kurt Keutzer, and Amir Gholami. 2025. Kvquant: Towards 10 million context length llm inference with kv cache quantization. _Advances in Neural Information Processing Systems (NeurIPS)_ 37 (2025), 1270–1303. 

- [16] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2024. Mixtral of experts. _arXiv preprint arXiv:2401.04088_ (2024). 

- [17] Young Jin Kim, Rawn Henry, Raffy Fahim, and Hany Hassan Awadalla. 2022. Who Says Elephants Can’t Run: Bringing Large Scale MoE Models into Cloud Scale Production. In _Proceedings of The Third Workshop on Simple and Efficient Natural Language Processing (SustaiNLP)_ . 36–43. 

- [18] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In _Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)_ . 611–626. 

- [19] Malgorzata Lazuka, Andreea Anghel, and Thomas Parnell. 2024. LLMPilot: Characterize and Optimize Performance of your LLM Inference Services. In _International Conference for High Performance Computing, Networking, Storage and Analysis (SC)_ . IEEE, 1–18. 

- [20] Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. 2024. InfiniGen: Efficient generative inference of large language models with dynamic KV cache management. In _18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)_ . 155–172. 

- [21] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024. Snapkv: Llm knows what you are looking for before generation. _Advances in Neural Information Processing Systems (NeurIPS)_ 37 (2024), 22947–22970. 

- [22] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. 2024. Video-LLaVA: Learning United Visual Representation by Alignment Before Projection. In _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ . 5971– 5984. 

312 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

JanusQuant 

- [23] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, WeiChen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. 2024. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. _Proceedings of Machine Learning and Systems (MLSys)_ 6 (2024), 87–100. 

- [24] Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, and Song Han. 2025. Qserve: W4a8kv4 quantization and system co-design for efficient llm serving. _Proceedings of Machine Learning and Systems (MLSys)_ (2025). 

- [25] Lian Liu, Haimeng Ren, Long Cheng, Zhaohui Xu, Yudong Pan, Mengdi Wang, Xiaowei Li, Yinhe Han, and Ying Wang. 2025. COMET: Towards Partical W4A4KV4 LLMs Serving. In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), Volume 2_ . 131–146. 

- [26] Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. 2023. Scissorhands: Exploiting the persistence of importance hypothesis for llm kv cache compression at test time. _Advances in Neural Information Processing Systems (NeurIPS)_ 36 (2023), 52342–52364. 

- [27] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. 2024. KIVI: A TuningFree Asymmetric 2bit Quantization for KV Cache. In _International Conference on Machine Learning (ICML)_ . PMLR, 32332–32344. 

- [28] AI@Meta Llama Team. 2024. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ (2024). 

- [29] Pierre-Emmanuel Mazare, Samuel Humeau, Martin Raison, and Antoine Bordes. 2018. Training Millions of Personalized Dialogue Agents. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ . 2775–2779. 

- [30] Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia. 2024. Specinfer: Accelerating large language model serving with tree-based speculative inference and verification. In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), Volume 3_ . 932–949. 

- [31] Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. 2022. CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis. In _The Eleventh International Conference on Learning Representations (ICLR)_ . 

- [32] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. Pytorch: An imperative style, high-performance deep learning library. _Advances in neural information processing systems (NeurIPS)_ 32 (2019). 

- [33] Ramya Prabhu, Ajay Nayak, Jayashree Mohan, Ramachandran Ramjee, and Ashish Panwar. 2025. vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention. In _Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), Volume 1_ . 1133–1150. 

- [34] Ruoyu Qin, Zheming Li, Weiran He, Jialei Cui, Feng Ren, Mingxing Zhang, Yongwei Wu, Weimin Zheng, and Xinran Xu. 2025. Mooncake: Trading More Storage for Less Computation—A KVCache-centric Architecture for Serving LLM Chatbot. In _23rd USENIX Conference on File and Storage Technologies (FAST)_ . 155–170. 

- [35] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang. 2023. Flexgen: High-throughput generative inference of large language models with a single gpu. In _International Conference on Machine_ 

_Learning (ICML)_ . PMLR, 31094–31116. 

- [36] Yixin Song, Zeyu Mi, Haotong Xie, and Haibo Chen. 2024. Powerinfer: Fast large language model serving with a consumer-grade gpu. In _Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles (SOSP)_ . 590–606. 

- [37] Qwen Team. 2024. Qwen2 technical report. _arXiv preprint arXiv:2407.10671_ (2024). 

- [38] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, MarieAnne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ (2023). 

- [39] Hulin Wang, Yaqi Xia, Donglin Yang, Xiaobo Zhou, and Dazhao Cheng. 2025. Harnessing inter-gpu shared memory for seamless moe communication-computation fusion. In _Proceedings of the 30th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP)_ . 170–182. 

- [40] Weihu Wang, Yaqi Xia, Donglin Yang, Xiaobo Zhou, and Dazhao Cheng. 2025. MXBLAS: Accelerating 8-bit Deep Learning with a Unified MicroScaled GEMM Library. In _Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC)_ . 1590–1603. 

- [41] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M Rush. 2020. Transformers: State-of-the-art natural language processing. In _Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations (EMNLP)_ . 38–45. 

- [42] Haojun Xia, Zhen Zheng, Xiaoxia Wu, Shiyang Chen, Zhewei Yao, Stephen Youn, Arash Bakhtiari, Michael Wyatt, Donglin Zhuang, Zhongzhu Zhou, Olatunji Ruwase, Yuxiong He, and Shuaiwen Leon Song. 2024. Quant-LLM: Accelerating the Serving of Large Language Models via FP6-Centric Algorithm-System Co-Design on Modern GPUs. In _2024 USENIX Annual Technical Conference (USENIX ATC)_ . 699–713. 

- [43] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. 2023. Smoothquant: Accurate and efficient post-training quantization for large language models. In _International Conference on Machine Learning (ICML)_ . PMLR, 38087–38099. 

- [44] Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao Fu, and Song Han. 2025. Duoattention: Efficient long-context llm inference with retrieval and streaming heads. In _The Thirteenth International Conference on Learning Representations (ICLR)_ . 

- [45] Haoran Xu, Young Jin Kim, Amr Sharaf, and Hany Hassan Awadalla. 2024. A Paradigm Shift in Machine Translation: Boosting Translation Performance of Large Language Models. In _The Twelfth International Conference on Learning Representations (ICLR)_ . 

- [46] Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, and Ceze Luis. 2025. Flashinfer: Efficient and customizable attention engine for llm inference serving. _Proceedings of Machine Learning and Systems (MLSys)_ (2025). 

- [47] Jeffrey Yu, Kartik Prabhu, Yonatan Urman, Robert M Radway, Eric Han, and Priyanka Raina. 2024. 8-bit Transformer Inference and Fine-tuning for Edge Accelerators. In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), Volume 3_ . 5–21. 

- [48] Yuxuan Yue, Zhihang Yuan, Haojie Duanmu, Sifan Zhou, Jianlong Wu, and Liqiang Nie. 2024. Wkvquant: Quantizing weight and key/value cache for large language models gains more. _arXiv preprint arXiv:2402.12065_ (2024). 

313 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

Chengyu Sun, Yaqi Xia, Hulin Wang, Donglin Yang, Xiaobo Zhou, and Dazhao Cheng 

- [49] Boyuan Zhang, Luanzheng Guo, Jiannan Tian, Jinyang Liu, Daoce Wang, Fanjiang Ye, Chengming Zhang, Jan Strube, Nathan R Tallent, and Dingwen Tao. 2025. High-performance Visual Semantics Compression for AI-Driven Science. In _Proceedings of the 30th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP)_ . 557–559. 

- [50] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. 2023. H2o: Heavy-hitter oracle for efficient generative inference of large language models. _Advances in Neural Information Processing Systems (NeurIPS)_ 36 (2023), 34661–34710. 

- [51] Zheng Zhang, Hulin Wang, Hongming Xu, Donglin Yang, Xiaobo Zhou, and Dazhao Cheng. 2025. HyTiS: Hybrid Tile Scheduling for 

   - GPU GEMM with Enhanced Wave Utilization and Cache Locality. In _Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC)_ . 1604–1618. 

- [52] Juntao Zhao, Borui Wan, Chuan Wu, Yanghua Peng, and Haibin Lin. 2024. LLM-PQ: Serving LLM on Heterogeneous Clusters with PhaseAware Partition and Adaptive Quantization. In _Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP)_ . 460–462. 

- [53] Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy, Tianqi Chen, and Baris Kasikci. 2024. Atom: Low-bit quantization for efficient and accurate llm serving. _Proceedings of Machine Learning and Systems (MLSys)_ 6 (2024), 196–209. 

Received 2025-08-29; accepted 2025-11-10 

314 

