2025 IEEE International Symposium on High Performance Computer Architecture (HPCA) 

**==> picture [29 x 29] intentionally omitted <==**

**==> picture [29 x 29] intentionally omitted <==**

**==> picture [28 x 28] intentionally omitted <==**

# BitMoD: Bit-serial Mixture-of-Datatype LLM Acceleration 

Yuzong Chen _[†]_ , Ahmed F. AbouElhamayed _[†]_ , Xilai Dai _[†]_ , Yang Wang _[‡]_ , Marta Andronic _[§]_ , George A. Constantinides _[§]_ , and Mohamed S. Abdelfattah _[†]_ 

> _†_ Computer Systems Lab, Cornell University 

> _‡_ Systems and Networking Research Group, Microsoft Research 

> _§_ Department of Electrical and Electronic Engineering, Imperial College London 

> _†{_ yc2367, afa55, xd44, mohamed _}_ @cornell.edu 

> _‡_ yang.wang92@microsoft.com 

> _§{_ marta.andronic18, g.constantinides _}_ @imperial.ac.uk 

_**Abstract**_ **—Large language models (LLMs) have demonstrated remarkable performance across various machine learning tasks. Yet the substantial memory footprint of LLMs significantly hinders their deployment. In this paper, we improve the accessibility of LLMs through BitMoD**[1] **, an algorithm-hardware co-design solution that enables efficient LLM acceleration at low weight precision. On the algorithm side, BitMoD introduces fine-grained data type adaptation that uses a different numerical data type to quantize a group of (e.g., 128) weights. Through the careful design of these new data types, BitMoD is able to quantize LLM weights to very low precision (e.g., 4 bits and 3 bits) while maintaining high accuracy. On the hardware side, BitMoD employs a bitserial processing element to easily support multiple numerical precisions and data types; our hardware design includes two key innovations: First, it employs a unified representation to process different weight data types, thus reducing the hardware cost. Second, it adopts a bit-serial dequantization unit to rescale the per-group partial sum with minimal hardware overhead. Our evaluation on six representative LLMs demonstrates that BitMoD significantly outperforms state-of-the-art LLM quantization and acceleration methods. For discriminative tasks, BitMoD can quantize LLM weights to 4-bit with** _<_ 0 _._ 5% **accuracy loss on average. For generative tasks, BitMoD is able to quantize LLM weights to 3-bit while achieving better perplexity than prior LLM quantization scheme. Combining the superior model performance with an efficient accelerator design, BitMoD achieves an average of** 1 _._ 69 _×_ **and** 1 _._ 48 _×_ **speedups compared to prior LLM accelerators ANT and OliVe, respectively.** 

## I. INTRODUCTION 

Large language models (LLMs) have achieved significant breakthroughs in natural language processing tasks [47], [57]. However, the growth of LLM size and complexity continues to outpace the scaling of compute performance and memory capacity in existing hardware platforms [22]. For example, the first generation of the GPT model, introduced in 2018, contains only 117 million parameters, while the second and third generations grew more than 10 _×_ and 1000 _×_ , respectively within two years [9]. This rapid increase in size necessitates significant memory capacity for model deployment, hindering their wide adoption, especially in edge scenarios with limited compute and memory resources. For instance, the state-of-theart (SOTA) open-source LLM family, Llama-3 [35], contains 

> 1Code is available at: https://github.com/yc2367/BitMoD-HPCA-25 

more than 8 billion parameters and requires more than 16GB of memory to store the model weights in 16-bit floatingpoint (FP16) format, which cannot fit in an edge GPU such as Jetson-TX2 with 8GB memory [37]. Therefore, designing novel LLM compression algorithms, together with accelerators co-designed for efficient deployment of the compressed models, presents a promising solution to enhancing the accessibility of LLMs on edge devices. 

Quantization serves as one of the most hardware-efficient methods to mitigate the computation and memory demands of LLMs. Generally, there are two types of quantization mechanisms. The first one is quantization-aware training (QAT), where retraining is needed to update model weights and quantization parameters (e.g., scaling factors) [26], [31]. The second approach is post-training quantization (PTQ), which does not require retraining [10], [19], [20], [25], [30], [42], [52]. Although QAT can achieve more competitive accuracy than PTQ, the prohibitive cost of retraining LLMs makes it less practical. As a result, PTQ is commonly adopted in existing LLM quantization studies. While some PTQ works quantize both weights and activations into low precision [25], [42], [52], weight-only quantization can offer a better trade-off between model accuracy and hardware efficiency for edge deployment of LLMs, where weights dominate the memory footprint [10], [19], [20], [30]. However, existing weight-only quantization works on GPUs suffer from poor computational efficiency since GPUs lack dedicated hardware to perform multiplication between integer weight and floating-point activation. Consequently, these methods must first dequantize the weight to FP16 and rely on the floating-point pipeline for computation. 

To achieve better computational efficiency for LLMs, a recent accelerator work, FIGNA [27], proposes a family of dedicated computing units for mixed-precision arithmetic between integer weights and floating-point activations. To further unleash the potential of quantization for improved hardware efficiency, several works have proposed algorithmhardware co-design solutions based on _custom_ low-precision data types [25], [26], [38], [40]. The microscaling format (MX) [38], [40], assigns 8-bit metadata as the shared exponent to a group of low-precision weights. ANT [26] introduces a 

979-8-3315-0647-6/25/$31.00 ©2025 IEEE DOI 10.1109/HPCA61900.2025.00084 

1082 

new data type that better adapts to the intra-tensor value distribution, thus reducing the quantization error. OliVe [25] proposes an outlier-victim-pair quantization mechanism, where an outlier value with a large magnitude is represented with an “Adaptive Biased Float” format and can be protected by pruning its adjacent victim value that has a small magnitude. 

In this paper, we propose _BitMoD_[2] , an algorithm-hardware co-design solution for efficient LLM acceleration at low weight precision. On the algorithm side, _BitMoD_ exploits the _per-group_ quantization [16], and modifies low-precision floating-point data types by repurposing the redundant zero value with a special value, which provides the ability to better adapt the data type itself to the numerical distribution of each weight group. Through careful choice of special values, _BitMoD_ is able to quantize LLM weights to very low precision (e.g., 4-bit and 3-bit) with tiny encoding overhead while maintaining good model accuracy. On the hardware side, _BitMoD_ employs the bit-serial computing paradigm with a unified representation for different low-precision data types to efficiently trade-off weight precision and hardware efficiency. The main contributions of this paper are summarized below: 

- 1) We propose _BitMoD_ , a hardware-efficient PTQ solution for LLM acceleration. _BitMoD_ introduces new data types that are tailored for per-group weight quantization at 4-bit and 3-bit precision with tiny encoding overhead. 

- 2) We demonstrate that the proposed data types can be seamlessly integrated with other quantization optimization techniques, achieving better model perplexity than SOTA software-only LLM quantization works. 

- 3) We propose an efficient accelerator design for _BitMoD_ , which adopts a unified bit-serial representation for multiple low-precision data types. This effectively reduces the hardware cost to perform computation between lowprecision weights and FP16 activations, and trades-off weight precision for improved hardware efficiency. 

- 4) Our evaluation on six representative LLMs shows that on average, _BitMoD_ achieves 2 _._ 2 _×_ speedup and 2 _._ 31 _×_ better energy efficiency compared to the baseline FP16 accelerator, _without_ loss in accuracy. Compared to SOTA accelerators ANT and OliVe, _BitMoD_ achieves an average speedup of 1 _._ 69 _×_ and 1 _._ 48 _×_ , respectively. 

## II. BACKGROUND AND MOTIVATION 

## _A. Why Weight Quantization for LLMs?_ 

To demonstrate the importance of LLM weight quantization for edge applications, we profile the total memory access footprint of weight and activation for four representative LLMs running both discriminative and generative tasks with a batch size of 1. For discriminative tasks, the LLM receives an input context and outputs a single token such as in sentiment analysis [46] and multiple-choice question answering [13]. For generative tasks, the LLM receive an input context and output multiple tokens. We set the input to output sequence length to 256 : 1 and 256 : 256 for discriminative and generative tasks, 

**==> picture [220 x 9] intentionally omitted <==**

**==> picture [230 x 107] intentionally omitted <==**

Fig. 1: Total memory access of weights and activations on discriminative tasks (with 256 input tokens and 1 output token) and generative tasks (with 256 input tokens and 256 generated tokens). Note the log scale on the y-axis. Note that the gap between weight and activation memory accesses increases for generative tasks at batch size 1 despite a much larger KV-cache than discriminative tasks. While prior work [44] has correctly reported a memory bottleneck caused by the KV-cache, this only occurs for 175B+ parameter models with a high batch size (e.g., 512) and a context lengths exceeding 512 tokens. This scenario is less relevant to our focus on low-batch edge LLM inference where the weights indeed dominate the total memory accesses. 

respectively, catering for edge applications as suggested by Lin _et al._ [30]. As shown in Fig. 1, the LLM weights access consumes orders of magnitude larger memory than the activations access. Although discriminative tasks only need to output a single token (e.g., “A”/“B”/“C” for multiple-choice question answering), the weight tensor dimension of an LLM (e.g., 2048 for OPT-1.3B) is much larger than the input token length, leading to memory access dominated by weights. Moreover, generative tasks necessitate repeated weight fetching for every new output token, resulting in significantly higher memory access for LLM weights. Thus, weight quantization is more effective for deploying LLMs in edge scenario where the batch size is small and the input token length is typically short. 

## _B. Quantization Basics_ 

One of the most popular quantization schemes is integer quantization, where a floating-point value is scaled and rounded to a low-precision integer. There are two widely used quantization modes – _symmetric_ and _asymmetric_ . Symmetric integer quantization can be expressed as follows: 

**==> picture [232 x 23] intentionally omitted <==**

where _Wf_ is the original floating-point tensor, _Wf_ max is the absolute maximum value, _b_ is the quantized integer precision, ∆ is the scaling factor, _Wq_ is the quantized integer value, and _Wqf_ is the floating-point value after performing dequantization (i.e., re-scaling). 

The symmetric quantization assumes that the minimum and maximum values of a tensor have the same absolute value (i.e., symmetric value range), but this is not always true. Hence, another popular mode of quantization is asymmetric quantization, which can be expressed as follows: 

**==> picture [221 x 53] intentionally omitted <==**

1083 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:30:15 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [228 x 88] intentionally omitted <==**

Fig. 2: Maximum value and value range for different quantization granularity. Results are normalized to the standard deviation ( _σ_ ) of the weight vector at the corresponding granularity, then averaged across all weight vectors. The per-group granularity has a group size of 128. 

where _Wf_ min is the absolute minimum value of _Wf_ , and _z_ represents the zero-point of the quantized tensor. 

## _C. Motivation_ 

We analyze several techniques that are widely adopted in recent quantization studies, which motivates our proposed _BitMoD_ framework. We mainly focus on weight quantization in our discussion. 

**Quantization Granularity Matters.** Consider a floatingpoint weight tensor _Wf[K][×][D]_ , where _K_ represents the number of output channels and _D_ is the channel size. There are three _granularities_ to quantize the model weight: per-tensor, perchannel, and per-group. The _per-tensor_ quantization uses the same scaling factor to quantize a whole weight tensor, while _per-channel_ quantization divides the weight tensor along the output channel into _K_ vectors, and quantizes every vector _Wf_[1] _[×][D]_ independently. However, given the large tensor size and hidden dimension of LLMs, these two granularities still lead to large quantization error. Specifically, the quantization error of a dequantized weight in Eq. 1 can be expressed as: 

**==> picture [202 x 24] intentionally omitted <==**

where ErrorRound is the rounding error during quantization, which has been shown to have an expected value of 0.25 [30]. Therefore, the quantization error is proportional to the scaling factor ∆, which is further proportional to the maximum value and range for symmetric (Eq. 1) and asymmetric (Eq. 2) quantization, respectively. 

In order to further reduce the quantization error, recent LLM quantization studies adopt the _per-group_ granularity [16], [19], [20], [30]. The per-group quantization further divides a weight channel _Wf_[1] _[×][D]_ into _D/G_ groups, each with a group size of _G_ . The group size introduces extra overhead to store the quantization parameters, i.e., scaling factor (and zero-point) for every group, and is usually set to 128 in SOTA quantization frameworks to balance accuracy and memory overhead [20], [30]. Fig. 2 demonstrates the benefits of per-group quantization by showing the maximum value and range in four representative LLMs at different granularity. The per-group granularity has the lowest maximum value and range, hence will have a lower quantization error compared to the other two granularity. Therefore, we focus on per-group quantization in this work. 

TABLE I. Wikitext-2 perplexity ( _↓_ ) under different quantization granularity and 4-bit data types. “PC” and “PG” stand for per-channel and per-group, respectively. The group size is 128. 

|Model|OPT-1.3B|Phi-2B|Llama-2-7B|Llama-2-13B|
|---|---|---|---|---|
|Granularity|PC<br>PG|PC<br>PG|PC<br>PG|PC<br>PG|
|FP16|1462<br>1462|971<br>971|547<br>547|488<br>488|
|INT4-Sym<br>INT4-Asym<br>FP4<br>Flint|.<br>.<br>36.05<br>16.04<br>48.41<br>15.41<br>16.07<br>**14.99**<br>**15.87**<br>16.23|.<br>.<br>13.03<br>11.15<br>12.08<br>**10.67**<br>**11.24**<br>10.68<br>11.71<br>11.23|.<br>.<br>12.92<br>5.84<br>8.89<br>**5.77**<br>8.07<br>**5.77**<br>**6.67**<br>6.09|.<br>.<br>5.47<br>5.07<br>5.27<br>**5.01**<br>**5.15**<br>5.05<br>5.31<br>5.29|



**Quantization Data Type Matters.** Numerous studies have proposed custom data types for quantization at the per-channel granularity [25], [26], [40]. We analyze the effects of adopting different data types for per-channel and per-group weight quantization. We explore four basic data types at 4-bit precision: integer with symmetric (INT4-Sym) and asymmetric (INT4-Asym) quantization, floating-point (FP4), and the Flint data type proposed by ANT [26]. Table I shows the resulting perplexity on the Wikitext-2 dataset [33]. We highlight two important observations. First, although Flint can achieve better perplexity at the per-channel granularity, it never outperforms other data types at the per-group granularity. Second, the per-group INT4-Asym and FP4 quantization achieve the best perplexity on some but not all studied LLMs, indicating that both asymmetry and FP data types are favorable for per-group quantization. The reason behind this is twofold. First, weight tensors typically exhibit Gaussian-like distribution that fits well to the floating-point data type [17], [51]. Second, while the effects of outliers are mitigated by pergroup quantization, a weight group can still contain outliers in an asymmetric pattern, being either solely positive or negative, as highlighted in previous studies [15], [16]. This characteristic benefits from asymmetric quantization. 

The above observation motivates us to explore new quantization data types that can combine the benefits of _asymmetry_ and FP formats to achieve better accuracy under per-group quantization. We notice that the basic FP data types have symmetric quantization values due to the inherent sign-magnitude binary representation that contains positive and negative zero values. Our key insight is that we can introduce additional asymmetry to FP by repurposing a redundant zero value with another _special value_ . This approach provides us with two key benefits. First, it allows to fully utilize the limited quantization levels. Although the redundant zero value does not affect highprecision formats such as FP16, it constitutes a large fraction of quantization levels at low precision (e.g., 12 _._ 5% at 3-bit precision). Second, we can tune the special value to make the extended FP data types better adapt to the per-group weight distribution, which we discuss in Section III-B. 

**Quantization Bit-width Matters.** While prior LLM accelerators mainly rely on bit-parallel architectures that support 8-bit and 4-bit precision [25]–[27], recent studies have shown that 6-bit floating-point weights exhibit negligible accuracy loss across various LLM models and tasks [49], [51]. Motivated by this, we analyze the effects of using different 6-bit data types for per-group LLM weight quantization. We consider four data 

1084 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:30:15 UTC from IEEE Xplore.  Restrictions apply. 

TABLE II. Wikitext-2 and C4 perplexity ( _↓_ ) under different 6-bit data types. We use per-group weight quantization with a group size of 128. 

|of 128.|||||
|---|---|---|---|---|
|Model|OPT-1.3B|Phi-2B|Llama-2-7B|Llama-2-13B|
|Dataset|Wiki<br>C4|Wiki<br>C4|Wiki<br>C4|Wiki<br>C4|
|FP16|14.62<br>14.72|9.71<br>12.74|5.47<br>6.97|4.88<br>6.47|
|INT6-Sym<br>INT6-Asym<br>FP6-E2M3<br>FP6-E3M2|**14.51**<br>14.80<br>14.61<br>14.78<br>14.59<br>**14.76**<br>14.81<br>14.81|9.85<br>12.82<br>**9.76**<br>**12.8**<br>9.85<br>**12.8**<br>9.81<br>12.87|**5.49**<br>**6.99**<br>**5.49**<br>**6.99**<br>5.52<br>**6.99**<br>**5.49**<br>7.02|**4.89**<br>**6.46**<br>**4.89**<br>**6.46**<br>4.92<br>6.49<br>**4.89**<br>6.50|



types: integer with symmetric (INT6-Sym) and asymmetric (INT6-Asym) quantization, floating-point with 2-bit exponent and 3-bit mantissa (FP6-E2M3), and floating-point with 3-bit exponent and 2-bit mantissa (FP6-E3M2). Table II compares the resulting perplexity of different quantization data types on Wikitext-2 [33] and C4 [18] datasets. On average, the studied 6-bit data types achieve similar and negligible perplexity loss compared to the FP16 baseline. For example, the average perplexity loss of INT6-Sym is less than 0.05, and its simple integer representation offers a promising solution to efficient LLM acceleration. Therefore, it is crucial for an accelerator to support diverse quantization bit-width to offer a better tradeoff between memory footprint and model accuracy. 

A natural solution for accommodating variable precision is to adopt bit-serial architectures [3], [12], [28], [45]. However, existing bit-serial accelerators mainly target the integer data type, which causes significant accuracy loss at 3-bit precision as we will show in Section V-B. Furthermore, these accelerators cannot leverage per-group quantization for improved accuracy. This is because per-group quantization assigns different scaling factors for different groups, necessitating a floatingpoint unit with large area overhead to dynamically dequantize the partial sum after computing the dot-product for every group. Thus, an efficient dequantization mechanism with low hardware cost is desirable. 

**Algorithm-Hardware Co-Design Matters.** Numerous frameworks have been proposed to accelerate LLM execution, as depicted in Table III. SOTA algorithmic solutions such as AWQ [30] quantize LLM weights to low-precision integer while preserving high accuracy. Nevertheless, AWQ is optimized for LLM acceleration on GPUs, which lack dedicated mixed-precision computing unit. As a result, it converts the low-precision weights to FP16 and relies on the GPU floatingpoint pipeline for computation, resulting in poor computational efficiency. 

In contrast, ANT [26], OliVe [25], and FIGNA [27] propose efficient bit-parallel accelerators for quantized model acceleration. But their precision is limited to 8-bit and 4-bit, which restricts the ability to utilize other precision (e.g., 6- bit) for a better accuracy-efficiency trade-off. Moreover, their accelerators do not natively support per-group quantization, which requires a floating-point unit to dynamically dequantize the per-group partial sum on the fly. While Microscaling [40] accommodates diverse precision, it necessities a floating-point pipeline to handle the shared micro-exponent of a weight group, leading to higher energy consumption compared to 

TABLE III. Comparison between _BitMoD_ and SOTA co-design frameworks for LLM acceleration 

|**Framework**|**Per-group**|**Supported**|**Accuracy @**|**Hardware**|
|---|---|---|---|---|
||**Quant?**|**Precision**|**3-bit Weight**|**Effciency**|
|AWQ [30]|Yes|**Limited**|**High**|**Low**|
|FIGNA [27]|No|**Limited**|**Low**|**High**|
|ANT [26]|No|**Limited**|**Low**|**High**|
|OliVe [25]|No|**Limited**|**Medium**|**High**|
|Microscaling [40]|Yes|**Many**|**Low**|**Medium**|
|**_BitMoD_ (Ours)**|Yes|**Many**|**High**|**High**|



other low-precision compute units. Furthermore, given the significant memory footprint of LLMs, it is desirable to explore sub-4-bit quantization while maintaining good model accuracy, which ANT, OliVe, and Microscaling do not address. As we will show in Section V-B, the custom quantization data types proposed by ANT, OliVe, and Microscaling fail to achieve better accuracy than the simple asymmetric integer quantization at 4-bit weight precision, and cause unacceptable accuracy loss at 3-bit weight precision under per-group quantization. The above limitation motivates us to propose an efficient LLM acceleration framework that supports a wide range of hardware-friendly bit-widths, while maintaining good accuracy at low weight precision. 

## III. BITMOD QUANTIZATION FRAMEWORK 

In this section, we present the _BitMoD_ quantization framework, which includes new data type families tailored for pergroup quantization at 3-bit and 4-bit precision. Section III-A describes our proposed data types that extend the basic floating-point data types at 3-bit and 4-bit precision. Section III-B presents an enhanced per-group LLM quantization strategy using the proposed data types. Section III-C describes the hardware-efficient per-group dequantization mechanism using integer scaling factors. 

## _A. Asymmetric FP3 and FP4 Data Types_ 

The basic floating-point formats contain a redundant quantization level due to the sign-magnitude representation that has both +0 and _−_ 0. We propose to replace this redundant zero with another _special value_ to fully utilize the available quantization levels and introduce additional asymmetry. We first use the basic FP3 format to derive our custom 3-bit data type, and then extend our idea to 4-bit precision. 

**FP3 Extension.** The basic FP3 data type contains seven distinct values _{_ 0 _, ±_ 1 _, ±_ 2 _, ±_ 4 _}_ . Our main idea is to extend FP3 and allows the redundant zero to be replaced by one of some pre-defined special values. Consequently, a weight group can be quantized by the basic FP3 data type together with a selected special value to minimize the quantization error. Ideally, the special values can have an arbitrary precision. But a high-precision (e.g., FP16) special value leads to more hardware overhead for computing, which offsets the efficiency of low-precision data types. Hence, we limit the special value to low-precision integers. Furthermore, given _N_ as the number of allowed special values, an encoding overhead of �log _N_ � 

1085 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:30:15 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [234 x 89] intentionally omitted <==**

Fig. 3: Normalized weight quantization error ( _↓_ ) with different special values (SV) for FP3. We use per-group quantization with a group size of 128. The special values _±_ 6 achieve the lowest overall quantization error, thus adopted in _BitMoD_ . 

bits is needed to specify which special value to be selected during computation. This selection also requires an _N_ -to-1 mux in the hardware implementation. To balance the encoding overhead and hardware complexity, we set _N_ = 4 which only requires 2-bit encoding per group. 

The choice of special values will affect the resulting quantization error because it changes the set of available quantization values. As discussed in Section II-C, both asymmetry and floating-point data types are crucial for good accuracy under per-group quantization. Since the scaling factor and quantized values are ultimately determined by the absolute maximum value of a data type [32], we establish the set of special values based on two principles. First, some special values should fall inside the numerical range of FP3 to ensure that they do not alter its original absolute maximum (i.e., 4). This is advantageous for quantizing weight groups exhibiting symmetric, Gaussian-like distribution. Second, some special values could fall outside the numerical range of FP3 to introduce additional asymmetry, i.e., the absolute maximum and minimum quantization values of the extended FP3 are different. This can benefit weight groups that exhibit asymmetric distribution. 

To satisfy the first property, the special values should be set to _±_ 3, which replace the redundant zero with + 3 and _−_ 3, respectively. We call this new data type FP3-ER since it adds extra resolution (ER) within the range of FP3. To satisfy the second property, there are an infinite number of values that can fall outside the FP3 range. Therefore, we determine the two remaining special values that can minimize the quantization error. We further reduce the search space by restricting these two special values to have the same absolute value, which results in _balanced_ asymmetry across all weight groups. This is desirable because, although an individual weight group may prefer asymmetric quantization, a whole weight tensor 

TABLE IV. Our proposed extended resolution (ER) and extended asymmetry (EA) FP3 and FP4 data types. 

|mmetry (|EA) FP3 and FP4|data types.||
|---|---|---|---|
|Dtype|Basic Values|Extended Dtype<br>S|pecial Value|
|FP3|0, _±_1, _±_2, _±_4|FP3-ER|_−_3 or +3|
|||FP3-EA|_−_6 or +6|
|FP4|0, _±_0_._5, _±_1, _±_1_._5<br>_±_2, _±_3, _±_4, _±_6|FP4-ER|_−_5 or +5|
|||FP4-EA|_−_8 or +8|



**Algorithm 1:** Fine-grained data type adaptation. 

|**A**|**lgorithm **|**1:** Fine-grained data type adaptation.|
|---|---|---|
||**Input**<br>**:**|Weight group: _W_; Quantization precision: _p_|
||**Output :**|Quantized weight group: _Wqout_;|
|||Selected special value: _vout_|
|**1 **|**Func** AdaptiveQuant(_W, p_)**:**||
||// Get basic and special quantization||
||values according to Table IV||
|**2**|basicValues = GetBasicValues(_p_)||
|**3**|specialValues = GetSpecialValues(_p_)||
||// Search for the best special value||
|**4**|minError = +_∞_||
|**5**|**for** _v_|in specialValues **do**|
|**6**|quantValues = basicValues _∪v_||
|**7**|_Wq_ = NonLinearQuantize(_W,_ quantValues)||
|**8**|newError = MeanSquareError(_W, Wq_)||
|**9**|**if** newError _<_ minError **then**||
|**10**||minError = newError|
|**11**||_Wqout_ = _Wq_|
|**12**||_vout_ =_v_|
||||
|**13**|**return** _Wqout_, _vout_||



usually exhibits symmetric, Gaussian-like distribution [26], [55]. Fig. 3 shows the normalized per-group quantization error on six LLMs when adding different special values to FP3. We observe that adding asymmetry significantly reduces the quantization error. In addition, the special values _±_ 6 have the lowest quantization error on most LLMs except for OPT-1.3B, and are therefore adopted in _BitMoD_ . We call the resulting new data type FP3-EA since it adds extra asymmetry (EA) to extend the range of FP3. 

**FP4 Extension.** Similar to FP3-ER and FP3-EA, we add extra resolution and asymmetry to FP4. We conduct experiments to measure the effects of different FP4 special values on the resulting quantization error, which leads to the best FP4-ER and FP4-EA that have special values _±_ 5 and _±_ 8, respectively. Table IV summarizes the extended FP3 and FP4 data types. Note that although we have fixed the four special values given that they can minimize the quantization error for the diverse set of LLMs that we evaluate, the proposed _BitMoD_ accelerator can flexibly accommodate other arbitrary special values that may perform well with different LLMs, which we discuss in Section IV-A. 

## _B. Fine-grained Data Type Adaptation_ 

The extended FP3 and FP4 data types contain four special values, but every weight group can only be quantized with one special value in addition to the basic values. Therefore, we propose a _fine-grained data type adaptation_ strategy that quantizes every group using a different special value to minimize the quantization error, as detailed in Algo. 1. First, the basic and special values are obtained from Table IV (Line 2 – 3). We iterate through all special values and add one special value to the set of basic values in every iteration (Line 5 – 6). Then we perform non-linear quantization (Line 7), which is commonly used in previous PTQ studies that map the floating- 

1086 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:30:15 UTC from IEEE Xplore.  Restrictions apply. 

TABLE V. Wikitext-2 and C4 perplexity ( _↓_ ) under different precision for the per-group scaling factor (SF). We use INT4-Asym for weight quantization with a group size of 128. 

|or the per-group scaling factor (SF). We useINT4-Asymfor weig<br>uantization with a group size of 128.|or the per-group scaling factor (SF). We useINT4-Asymfor weig<br>uantization with a group size of 128.|or the per-group scaling factor (SF). We useINT4-Asymfor weig<br>uantization with a group size of 128.|or the per-group scaling factor (SF). We useINT4-Asymfor weig<br>uantization with a group size of 128.|or the per-group scaling factor (SF). We useINT4-Asymfor weig<br>uantization with a group size of 128.|
|---|---|---|---|---|
|Model<br>OPT-1.3B<br>Phi-2B<br>Llama-2-7B<br>Llama-2-13B|||||
|SF Bits|Wiki<br>C4|Wiki<br>C4|Wiki<br>C4|Wiki<br>C4|
|FP16|15.41<br>15.84|10.68<br>13.66|5.77<br>7.31|5.01<br>6.62|
|INT8<br>INT6<br>INT4<br>INT2|15.41<br>15.84<br>15.43<br>15.84<br>15.52<br>15.93<br>18.46<br>18.61|10.68<br>13.66<br>10.74<br>13.71<br>10.76<br>13.73<br>15.68<br>18.32|5.77<br>7.31<br>5.77<br>7.31<br>5.77<br>7.36<br>8.41<br>10.63|5.01<br>6.62<br>5.01<br>6.62<br>5.03<br>6.64<br>6.19<br>8.12|



point tensor to a set of non-linear values (i.e., non-INT data types) [19], [25], [26]. Finally, we assign the special value that has the lowest mean-square error between the original and quantized weight (Line 8 – 11). 

Although Algo. 1 describes the quantization procedure for a single weight group, the algorithm can be vectorized on a GPU to find the best special value for all groups of a weight tensor simultaneously. For our implementation, the algorithm only takes _∼_ 10 second to quantize the whole Llama-2-7B model on a single A6000 GPU. Hence, our proposed quantization strategy exhibits high compression speed and efficiency. 

## _C. Efficient Per-group Dequantization_ 

While quantization allows computing the dot product in low precision, it still requires dequantization (i.e., re-scaling) after producing the output. Specifically, Eq. 1 indicates that the quantized low-precision weight should be multiplied by the scaling factor to obtain the actual floating-point weight. Per-channel quantization only needs re-scaling after producing the final output activation. Such per-channel re-scaling can be further fused into other element-wise operations such as layernorm before writing the output activation back to memory, reducing the data transfer cost [30], [52]. However, per-group quantization must dequantize the partial sum after computing every group of dot-products because different groups have different scaling factors. Furthermore, since _BitMoD_ maintains the input activation at FP16, the group partial sum will also have floating-point formats. As a result, performing dequantization on-the-fly necessitates a floating-point pipeline, which can diminish the potential hardware efficiency gained from using low-precision weights. 

In order to reduce the dequantization cost, we build upon prior work VS-Quant [14], which applies a second-level quantization that further quantizes the scaling factors to lowprecision integers. Given the weight channel size _D_ and group size _G_ , VS-Quant applies symmetric quantization (Eq. 1) to the _D/G_ scaling factors from the same channel, where the precision of per-group scaling factor is a design parameter. However, VS-Quant only targets small-scale neural networks and uses a small group size of 16. It is unclear how quantizing the scaling factors of a larger group will affect the accuracy of LLMs. Hence, we conduct experiments to find the best precision for per-group scaling factors. We use INT4-Asym quantization as an example, while other data types show the same trend. As shown in Table V, INT8 per-group scaling factors have no accuracy loss compared to using FP16 scaling 

**==> picture [236 x 153] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Bsig = 4 Bsig = 0<br>Truth Table of INT Booth Term<br>INT8 I7 I6 I5 I4 I3 I2 I1 I0 0 3b String Op Sign Exp Man<br>000 / 111 0 0 0 0<br>Bsig = 6 Bsig = 2 001 / 010 + 𝑥 0 0 1<br>Bsig = 4 Bsig = 0 110 / 101 −𝑥 1 0 1<br>011 +2 𝑥 0 1 1<br>INT6 I5 I4 I3 I2 I1 I0 0 100 − 2 𝑥 1 1 1<br>Bsig = 2<br>(b) SV_reg SV0 SV1 SV2 SV3<br>Bsig = 0 2 Exp<br>Sign LOD<br>S E FP4 1 E0 M FP 10 S I3 I2 I1 I0 F0 LOD 21 ManExp<br>Fixed 1 Bsig = 1 1 Man<br>− 0 e q<br>→<br>**----- End of picture text -----**<br>


Fig. 4: Unified bit-serial representation of (a) INT8, INT6, and (b) FP4. Every bit-serial term contains four parts: sign, exponent (exp), mantissa (man) and bit-significance (bsig). 

factors. This is expected since INT8 can even achieve no accuracy loss for per-channel weight quantization [52], [53], which has a much wider numerical range than scaling factors. Thus, we use INT8 per-group scaling factors in _BitMoD_ , which allows efficient per-group dequantization in a bit-serial manner as will be described in Section IV-B. 

**Memory Overhead Analysis.** The proposed _BitMoD_ quantization only needs an 8-bit scaling factor and 2-bit encoding metadata to select the special value for every group. Given a large group size such as 128, which is commonly used in SOTA software-only LLM quantization studies [19], [20], [30], the 10-bit extra memory per group incurs practically no overhead. Furthermore, prior software-only PTQ works mainly adopt asymmetric integer quantization [19], [20], [30], which requires a 16-bit scaling factor and an 8-bit zero-point per group. Hence, _BitMoD_ exhibits lower memory overhead compared to these works. 

## IV. BITMOD HARDWARE ACCELERATOR 

In this section, we describe the _BitMoD_ hardware design, which leverages the bit-serial computing paradigm to offer a good trade-off between weight precision, model accuracy and hardware efficiency. Section IV-A develops a unified bit-serial representation of different low-precision data types supported by _BitMoD_ . Section IV-B details the microarchitecture of the _BitMoD_ processing element (PE). Section IV-C presents the overall accelerator architecture. 

## _A. Unified Bit-serial Representation_ 

Prior studies have demonstrated that per-channel INT8 weight quantization shows no accuracy loss compared to using FP16 weights [27], [52], [53]. Moreover, as discussed in Section II-C, per-group INT6 quantization also has negligible accuracy loss. Hence, the design target of _BitMoD_ hardware is to support INT8, INT6, as well as the new FP4 and FP3 extensions in a unified architecture. However, the basic values of FP3 and FP4 use the floating-point format, which is not compatible with the integer representation. A naive approach 

1087 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:30:15 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [477 x 111] intentionally omitted <==**

**----- Start of picture text -----**<br>
❶ Exponent Alignment ❷ Bit-Serial Multiplication ❸ Group Accumulation ❹ Dequantization<br>ae0 5 eACC 6 MAX- 6 wa δ m0m0e0 111 >> 14 neg 15 >> 23 15 Δ i1<br>we0 2 + × 4 lanes 17 << 23 mACC 21 mGRP<br>ae3 - 6 δ e3 3<br>wwae3s0s0 1 + 1 ys0 wam3m3 111 >> 14 neg 15 - wbsig 6 4 6 0 3 < < 1<br>+ + N ormalize<br>was3s3 1 ys3 eMAX eACC 6 eGRP<br>... ... ... Normalize<br>...<br>+<br>+ +<br>**----- End of picture text -----**<br>


Fig. 5: The microarchitecture of _BitMoD_ PE. Every bit-serial weight term contains 1-bit sign ( _ws_ ), 2-bit exponent ( _we_ ), 1-bit mantissa ( _wm_ ), and a shared bit-significance ( _wbsig_ ). 

is to convert all data types to INT8, yet this cannot improve the computational efficiency for lower-precision weights. 

In order to trade-off lower weight precision for improved hardware efficiency, we propose a unified bit-serial representation, where every number is decomposed into a series of bitserial terms, each containing four parts: sign, exponent (exp), mantissa (man), and bit-significance (bsig). The value of a bit-serial term can be expressed as: 

**==> picture [183 x 12] intentionally omitted <==**

Fig. 4 describes the bit-serial representation for different data types supported by _BitMoD_ . For INT8 and INT6, we apply Booth encoding [8] to decompose their binary strings into four and three 3-bit Booth strings, respectively. The Booth encoding has been widely used in prior bit-serial accelerators to speed up computation [3], [5], [43]. Every two adjacent Booth strings have a difference of 2 in bit-significance. The sign, exponent, and mantissa depend on the content of a Booth string, which defines the desired operation when multiplying with another operand _x_ . 

For the extended FP4, we first convert it to fixed-point values in sign-magnitude format with 1 sign bit, 4 integer bits _{I_ 3 _, I_ 2 _, I_ 1 _, I_ 0 _}_ to handle the largest special value _±_ 8 of FP4-EA, and 1 fraction bit _{F_ 0 _}_ to handle _±_ 0 _._ 5 and _±_ 1 _._ 5 of the basic FP4 values. The fixed-point value is then compared with the redundant negative zero. If the comparison result is equal, the negative zero will be replaced by the assigned special value ( _SV_ ) of a particular weight group. The four allowed special values are stored in a register file ( _SV reg_ ), which only requires one-time programming before deploying an LLM. To obtain the bit-serial term, we observe that all values of the extended FP4 in Table IV contain at most two ‘1’ bits after converting to the fixed-point format. Hence, we use the simple leading-one detector ( _LOD_ ) to get two bit-serial terms from the first four bits _{I_ 3 _, I_ 2 _, I_ 1 _, I_ 0 _}_ and last four bits _{I_ 2 _, I_ 1 _, I_ 0 _, F_ 0 _}_ , respectively. Finally, since the extended FP3 values are a subset of FP4, it can be decoded into two bit-serial terms using the same hardware. Note that the bit-serial decoder is not limited to support the special values shown in Table IV. The special value register file can be programmed with other special values as needed, and the number of decoded bitserial terms can be minimized with simple modification to the decoder. For example, the special value 7 can be expressed as 

**==> picture [236 x 115] intentionally omitted <==**

**----- Start of picture text -----**<br>
Weight Buffer<br>BitMoD PE Column<br>Bit-serial Term Generator Bit-serial<br>sign exp man Bsig Weight Term Output<br>Buffer<br>PE Tile (0, 0) PE Tile (0, 1) ... PE Tile (0, 3) Input PE<br>Input PE<br>PE Tile (1, 0) PE Tile (1, 1) ... PE Tile (1, 3) Input PE ACC<br>4×4 ×8<br>PE Tile (3, 0) PE Tile (3, 1) ... PE Tile (3, 3) Input PE ×8<br>Off-Chip DRAM<br>... ... ... ...<br>Input Buffer<br>**----- End of picture text -----**<br>


Fig. 6: _BitMoD_ accelerator architecture. 

two bit-serial terms 2[3] and _−_ 2[0] instead of using a leading-one detector that emits three bit-serial terms. 

## _B. BitMoD Processing Element_ 

While _BitMoD_ is able to quantize weight to low precision, activation still remains in FP16. To address this challenge, we propose a mixed-precision bit-serial PE as shown in Fig. 5. In every cycle, the PE performs a 4-way dot product between four bit-serial weight terms ( _w_ ) and four FP16 activations ( _a_ ). Step 1 first aligns the sum of exponents ( _ae_ + _we_ ) to compute the delta exponent ( _δe_ ). It also generates the sign ( _ys_ ) of every product between a weight term and activation. Step 2 performs the bit-serial multiplication between the 1-bit weight mantissa ( _wm_ ) and 11-bit activation mantissa ( _am_ ) including the hidden bit. The multiplication result is aligned by a rightshifter that is controlled by the delta exponent. We reserve 3 extra bits in the shifter result to account for rounding to the nearest even as suggested by Awad _et al._ [5]. The bit-serial dot product of the mantissa is then computed using an adder tree. After producing the bit-serial dot product, Step 3 performs accumulation by first multiplying the dot product with the weight bit-significance ( _Wbsig_ ), followed by adding with the accumulator mantissa ( _mACC_ ). The accumulated mantissa is then normalized to update the accumulator exponent ( _eACC_ ). Since _BitMoD_ adopts per-group quantization, the accumulated group partial sum must be dequantized on the fly. To reduce this hardware cost, Step 4 performs dequantization in a bit-serial manner. Specifically, the accumulator mantissa is multiplied by one bit of the group scaling factor (∆ _i_ ) in every cycle, followed by shift-and-add to obtain the exponent ( _eGRP_ ) and mantissa ( _eGRP_ ) of the dequantized partial sum. 

1088 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:30:15 UTC from IEEE Xplore.  Restrictions apply. 

TABLE VI. Wikitext-2 and C4 perplexity ( _↓_ ) using different data types under per-group weight quantization. 

|**Precision**<br>**Datatype***<br>**OPT-1.3B**<br>**Phi-2B**<br>**Yi-6B**<br>**Llama-2-7B**<br>**Llama-2-13B**<br>**Llama-3-8B**<br>Wiki<br>C4<br>Wiki<br>C4<br>Wiki<br>C4<br>Wiki<br>C4<br>Wiki<br>C4<br>Wiki<br>C4|**Mean**∆**PPL**|
|---|---|
|16-bit<br>FP16<br>14.62<br>14.72<br>9.71<br>12.74<br>5.84<br>8.91<br>5.47<br>6.97<br>4.88<br>6.47<br>6.13<br>8.88|0|
|4-bit<br>ANT<br>16.23<br>16.08<br>11.23<br>14.31<br>6.87<br>10.95<br>6.09<br>7.71<br>5.31<br>6.91<br>7.58<br>11.04<br>OliVe<br>15.38<br>15.82<br>10.49<br>**13.51**<br>6.55<br>9.84<br>5.91<br>7.34<br>5.13<br>6.74<br>6.89<br>9.91<br>MX-FP4<br>15.39<br>15.81<br>10.72<br>13.72<br>6.62<br>10.24<br>5.82<br>7.39<br>5.11<br>6.71<br>7.04<br>10.13<br>INT4-Asym<br>15.41<br>15.74<br>10.67<br>13.65<br>6.32<br>9.69<br>5.77<br>7.31<br>**5.01**<br>6.62<br>6.84<br>9.79<br>_BitMoD_<br>**14.89**<br>**15.29**<br>**10.48**<br>13.53<br>**6.23**<br>**9.58**<br>**5.72**<br>**7.26**<br>**5.01**<br>**6.61**<br>**6.73**<br>**9.66**|1.23<br>0.68<br>0.79<br>0.62<br>**0.48**|
|3-bit<br>ANT<br>340.6<br>332.9<br>15.57<br>18.35<br>9.01<br>14.32<br>8.51<br>10.28<br>6.40<br>7.98<br>15.22<br>17.56<br>OliVe<br>76.79<br>59.63<br>14.93<br>17.76<br>32.42<br>66.02<br>9.13<br>12.04<br>8.69<br>12.43<br>26.76<br>46.39<br>MX-FP3<br>1E+3<br>771.6<br>17.89<br>20.37<br>15.41<br>21.97<br>8.86<br>11.99<br>7.19<br>9.13<br>23.82<br>31.39<br>INT3-Asym<br>139.4<br>144.9<br>13.92<br>16.79<br>8.66<br>13.33<br>7.08<br>9.29<br>5.64<br>7.35<br>13.26<br>17.80<br>_BitMoD_<br>**22.67**<br>**20.47**<br>**12.91**<br>**15.69**<br>**7.66**<br>**11.98**<br>**6.55**<br>**8.36**<br>**5.50**<br>**7.18**<br>**8.96**<br>**12.82**|57.61<br>23.14<br>152.8<br>24.34<br>**2.94**|



> ***** All quantization data types use per-group quantization. The MX data type uses a group size of 32 following the standard in [38], while other data types use a group size of 128. The perplexity of MX degrades when using a larger group size. 

One concern of the bit-serial dequantization is whether it will take more cycles than the normal dot product stage and cause potential pipeline stalling. As discussed in Section III-C, the per-group scaling factor has 8 bits, which requires 8 cycles for dequantization. On the other hand, even the lowestprecision data type FP3 in _BitMoD_ requires two cycles to process two bit-serial terms. Given the PE dot-product size of 4 and a commonly used group size of 128, the group dot-product stage takes 128 _/_ 4 _×_ 2 = 64 cycles to complete. Therefore, the proposed bit-serial dequantization will never stall the computing pipeline. Furthermore, since INT6 and the extended FP4/FP3 data types contain three and two bitserial terms, the proposed _BitMoD_ PE is able to compute 4 multiply-accumulate operations in 3 and 2 cycles, respectively. Compared to the normal FP16 multiply-accumulate hardware, _BitMoD_ achieves a throughput improvement of 1 _._ 33 _×_ and 2 _×_ for INT6 and FP4/FP3 data types, respectively. In fact, as will be evaluated in Section V-C, the _BitMoD_ PE consumes 24% less area than an FP16 PE, thus is able to provide even higher throughput under an iso-compute area constraint. 

Besides matrix multiplication between weights and activations, LLMs also contain the self-attention layer, which involves two matrix multiplication operations between three activation tensors, i.e., query, key, and value. Given that the _BitMoD_ PE only maintains one activation tensor in FP16, the other two activation tensors need to be low-precision integers. Fortunately, prior works have demonstrated that the key and value tensors are very amenable to quantization due to the softmax normalization inside self-attention, and can be safely quantized to INT8 and even INT4 with negligible accuracy loss [44], [52], [58]. Hence, _BitMoD_ can accommodate the self-attention layer with the proposed bit-serial PE by quantizing the key and value tensors to low-precision integers. 

## _C. BitMoD Accelerator_ 

Fig. 6 shows the overall architecture of the _BitMoD_ accelerator. The input and weight buffers are banked to provide adequate bandwidth for the access from PEs. The bit-serial term generator receives the weight data and decomposes them into bit-serial terms as discussed in Section IV-A. The main PE array contains 4 _×_ 4 tiles connected in a systolic manner. 

Every PE tile has 8 rows _×_ 8 columns and adopts an outputstationary dataflow. The bit-serial weight term is broadcast to the whole PE column, while the input is broadcast to the whole PE row. This allows _BitMoD_ to exploit data reuse through both weight-sharing and input-sharing. Every PE column contains a local output buffer and an accumulator, which is used to accumulate the per-group partial sum from a PE to obtain the final per-channel output activation. Since processing a weight group takes many cycles, there is enough time to drain the whole PE column using only one shared accumulator to amortize the hardware cost. 

## V. EVALUATION 

## _A. Experimental Methodology_ 

**LLM Benchmarks.** For evaluation, we choose six representative LLMs with diverse model sizes, including OPT-1.3B [57], Phi-2B [36], Yi-6B [1], Llama-2-7B, Llama-2-13B [34], and Llama-3-8B [35]. We obtain the pre-trained models from the HuggingFace repository, and implement the proposed _BitMoD_ quantization framework in PyTorch. To evaluate the effects of quantization on the resulting model performance, we consider both discriminative and generative tasks. For discriminative tasks, we evaluate three benchmarks: HellaSwag [56], WinoGrande [41], and Piqa [7] under the zero-shot setting using LM-Evaluation-Harness [21]. For generative tasks, we choose Wikitext-2 [33] and C4 [18] datasets and evaluate the perplexity following the methodology in prior quantization works [4], [20], [30], [42]. 

**Quantization Data Types.** We compare the model accuracy of _BitMoD_ with four baseline quantization data types: 

- ANT [26], which adaptively uses different data types to quantize different tensors at the per-channel granularity. 

- OliVe [25], that introduces an outlier-victim pair encoding mechanism, which sacrifices the normal value (i.e., victim) adjacent to the outlier to accommodate the important outlier value. 

- Microscaling (MX) [40], which groups 32 low-precision FP weights with an extra 8-bit shared exponent. 

- Per-group asymmetric integer quantization, which is commonly adopted in prior software quantization methods. 

1089 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:30:15 UTC from IEEE Xplore.  Restrictions apply. 

TABLE VII. Accuracy (higher is better) of discriminative tasks using different data types under per-group weight quantization. 

|**Precision**<br>**Datatype**<br>**OPT-1.3B**<br>**Phi-2B**<br>**Yi-6B**<br>**Llama-2-7B**<br>**Llama-2-13B**<br>**Llama-3-8B**<br>Hella Wino<br>Piqa<br>Hella Wino<br>Piqa<br>Hella Wino<br>Piqa<br>Hella Wino<br>Piqa<br>Hella Wino<br>Piqa<br>Hella Wino<br>Piqa|**Mean**∆**Acc**|
|---|---|
|16-bit<br>FP16<br>53.72 59.43 72.41<br>73.74 75.77 79.22<br>74.96 70.72 78.78<br>75.98 69.06 79.11<br>79.39 72.38<br>80.5<br>79.18 72.85 80.74|0|
|4-bit<br>INT4-Asym<br>52.31 **59.35** 71.05<br>72.29 75.14<br>78.4<br>73.91 **70.51** 77.64<br>75.29 **68.74** 78.22<br>**78.76 72.45**<br>80.2<br>78.07 **73.24** 79.76<br>_BitMoD_<br>**53.03** 59.12 **71.49**<br>**72.51 77.58 79.48**<br>**73.98** 70.09 **78.35**<br>**75.43** 68.19 **78.45**<br>78.41 72.14 **80.42**<br>**78.49** 73.09 **79.98**|-0.71<br>**-0.42**|
|3-bit<br>INT3-Asym<br>38.98 55.01 64.25<br>67.75 71.74 77.48<br>**71.3**<br>67.32 **76.71**<br>71.87 **66.46** 76.66<br>76.58 69.61 78.94<br>68.56 66.61 75.03<br>_BitMoD_<br>**49.16 58.09 68.88**<br>**70.16 75.22 78.18**<br>70.72 **67.72** 76.28<br>**72.68** 66.22 **77.53**<br>**76.79 72.37 79.22**<br>**73.56 70.32 77.91**|-4.84<br>**-2.61**|



To ensure a fair comparison with _BitMoD_ , we only apply the data types of ANT, OliVe, and MX to LLM weight quantization while maintaining activations in FP16. Despite that the original ANT and OliVe frameworks only support per-channel quantization due to the absence of dedicated dequantization hardware, we have extended their algorithms for per-group quantization. This allows to compare the model accuracy purely based on the employed quantization data types. While we mainly focus on weight quantization in our evaluation, Section V-E demonstrates that _BitMoD_ can be combined with SOTA activation quantization scheme, which offers the potential to reduce activation memory as well. 

In addition to co-design works, we show that _BitMoD_ can be seamlessly integrated into existing software-only quantization optimizations to further reduce the memory footprint or achieve better model performance. We combine _BitMoD_ with three software quantization methods: 

- SmoothQuant [52], which targets both weight and activation quantization in INT8. It addresses the quantization difficulty of large activation magnitude by partially migrating it to weights. 

- AWQ [30], which employs activation-aware weight quantization to protect the salient weight channels corresponding to larger activation magnitudes. 

- OmniQuant [42], which modulates the outlier weight values by optimizing the clipping threshold through blockwise fine-tuning. 

To integrate _BitMoD_ with these software-only methods, we replace their original weight quantizers that use integer data types with the extended FP4 and FP3 data types of _BitMoD_ . 

**Accelerator Baselines.** To evaluate the hardware performance and energy efficiency, we compare _BitMoD_ with a baseline accelerator that supports FP16 models and uses an FP16 multiply-accumulate PE instead of the proposed bit-serial PE. We also compare _BitMoD_ with ANT and OliVe, which design 

TABLE VIII. Wikitext-2 and C4 perplexity ( _↓_ ) when quantizing Llama weights using different data types. We use per-group weight quantization with a group size of 128. 

|**Precision**|**Datatype**|**Llama-2-7B**<br>Wiki<br>C4|**Llama-2-7B**<br>Wiki<br>C4|**Llama-2-13B**<br>Wiki<br>C4|**Llama-2-13B**<br>Wiki<br>C4|**Llama-3-8B**<br>Wiki<br>C4|**Llama-3-8B**<br>Wiki<br>C4|
|---|---|---|---|---|---|---|---|
||FP4|5.77|7.32|5.05|6.66|6.86|9.85|
|4-bit|FP4-ER<br>FP4-EA|5.74<br>5.81|7.28<br>7.30|5.03<br>5.08|6.63<br>6.65|6.76<br>6.83|9.71<br>9.79|
||_BitMoD_|**5.72**|**7.26**|**5.01**|**6.61**|**6.73**|**9.66**|
||FP3|7.51|10.28|5.90|7.58|15.22|19.87|
|3-bit|FP3-ER<br>FP3-EA|7.18<br>6.61|9.71<br>8.45|5.66<br>5.54|7.33<br>7.23|13.43<br>9.06|17.56<br>12.97|
||_BitMoD_|**6.55**|**8.36**|**5.50**|**7.18**|**8.96**|**12.82**|



custom decoders to support multiple data types in a unified systolic array. We evaluate the performance in LLMs with a batch size of 1 and an input sequence length of 256, catering for edge use cases as in prior work [30]. 

**Hardware Implementation.** We implement the accelerator of _BitMoD_ at RTL-level using SystemVerilog and verify the functionality of each component via RTL simulation. We use Synopsys Design Compiler to synthesize _BitMoD_ in TSMC 28nm technology to report the area and power. For endto-end performance evaluation, we implement a cycle-level simulator, where the accelerator timing and energy parameters are set based on the RTL synthesis results. The DRAM power is calculated based on the DDR4 model from DRAMSim3 [29]. All accelerators are evaluated under an iso-compute area constraint, and equipped with 512 KB activation buffer and 512 KB weight buffer, which are modelled with CACTI [6]. 

## _B. Accuracy Comparison of Different Data Types_ 

**Generative Tasks.** Table VI details the perplexity of applying different PTQ data types at 4-bit and 3-bit weight precision. For 4-bit and 3-bit weight quantization, _BitMoD_ achieves _<_ 0 _._ 5 and _<_ 3 perplexity loss on average compared to the FP16 baseline models, respectively. Although ANT, OliVe, and MX are able to maintain acceptable perplexity at 4-bit precision, they experience significant degradation in perplexity when quantizing weights to 3 bits. OliVe, which is designed to handle outlier values under per-channel quantization, finds its advantages diminished since the impact of outliers can be significantly mitigated through per-group quantization as described in Section II-C. MX employs the basic FP4 and FP3 without exploring the potential of their redundant zero, leading to worse perplexity than INT-Asym that fully utilizes the available quantization levels while supporting asymmetry. On the contrary, _BitMoD_ consistently outperforms asymmetric integer quantization at per-group granularity, and the benefits are more pronounced at 3-bit precision. This demonstrates that the combination of asymmetry and floating-point data types in _BitMoD_ can significantly reduce the quantization error. 

**Discriminative Tasks.** Table VII compares the model accuracy of discriminative tasks when employing _BitMoD_ and the 

TABLE IX. Wikitext-2 and C4 perplexity ( _↓_ ) when using different special values for FP3. 

|**Special**|**OPT-1.3B**|**OPT-1.3B**|**Phi-2B**|**Phi-2B**|**Llama-2-7B **|**Llama-2-7B **|**Llama-3-8B**|**Llama-3-8B**|
|---|---|---|---|---|---|---|---|---|
|**Values**|Wiki|C4|Wiki|C4|Wiki|C4|Wiki|C4|
|_{±_5_, ±_6_}_|23.39|**20.12**|13.02|15.84|6.61|8.48|9.09|13.81|
|_{±_3_, ±_5_}_|35.54|37.65|13.41|16.29|6.68|8.73|10.32|14.48|
|_{±_3_, ±_6_}_|**22.67**|20.47|**12.91 **|**15.69**|**6.55**|**8.36**|**8.96**|**12.82**|



1090 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:30:15 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [470 x 108] intentionally omitted <==**

Fig. 7: Speedup of different accelerators (higher is better). 

**==> picture [473 x 126] intentionally omitted <==**

Fig. 8: Energy consumption breakdown of different accelerators ( _↓_ ). “LL” and “LY” stand for ‘lossless” and ‘lossy”, respectively. 

baseline asymmetric integer quantization at per-group granularity. _BitMoD_ achieves better or comparable accuracy than asymmetric integer quantization. At 4-bit precision, _BitMoD_ has _<_ 0 _._ 5% accuracy loss on average compared to the baseline FP16 models. Moreover, on average, _BitMoD_ achieves a big improvement of 2 _._ 2% in model accuracy compared to the asymmetric integer quantization that is widely used in SOTA software quantization methods. 

_**BitMoD**_ **Data Type Ablation.** As discussed in Section III-A, _BitMoD_ introduces new data types by adding extra resolution and asymmetry to FP3 and FP4. We analyze the effects of these different data types on the perplexity of three studied Llama models. As shown in Table VIII, the _BitMoD_ data types with both extra resolution and asymmetry achieve the best perplexity. Compared to the basic FP4 data type, FP4-ER shows a greater improvement in perplexity than FP4-EA. This is because at 4-bit, the basic FP4 still has enough quantization levels to quantize a weight group with asymmetric distribution, and adding extra resolution can better reduce the quantization error. On the contrary, for 3-bit precision, FP3-EA achieves much better perplexity than FP3-ER. Given fewer quantization levels at 3-bit, the extra asymmetry introduced by FP3-EA has a larger impact when accounting for weight groups with asymmetric distribution. 

_**BitMoD**_ **Special Value Ablation.** The _BitMoD_ PE can flexibly support different special values. We evaluate two other potential combinations of special values for FP3: _{±_ 3 _, ±_ 5 _}_ and _{±_ 5 _, ±_ 6 _}_ . Table IX shows the resulting Wikitext and C4 perplexity. The adopted special values in _BitMoD_ , i.e., _{±_ 3 _, ±_ 6 _}_ , achieve the lowest perplexity on average. The 

special value combination _{±_ 5 _, ±_ 6 _}_ only introduces extra asymmetry to the basic FP3. However, many weight groups can exhibit symmetric distribution, which prefers a symmetric data type with the same absolute maximum and minimum values. On the other hand, the special values _±_ 5 have a higher quantization error than _±_ 6 as described in Section III-A, leading to worse perplexity when combined with _±_ 3. 

## _C. Accelerator Performance_ 

For accelerator evaluation, we consider two configurations for the _BitMoD_ accelerator based on the resulting model accuracy: (1) _Lossless_ , where the weight precision is INT6 given its near-zero accuracy loss under per-group quantization. We compare this configuration with the baseline FP16 accelerator. (2) _Lossy_ , where the weight can be quantized to 4-bit for discriminative tasks and 3-bit for generative tasks while maintaining good model performance. We compare this configuration with ANT and OliVe. 

**Tile Area and Power.** Table X shows the PE tile area and power breakdown of _BitMoD_ and the baseline FP16 accelerator at 1 GHz frequency. The unified bit-serial representation allows _BitMoD_ to support different weight data types with low hardware cost. As a result, the _BitMoD_ PE is 24% smaller 

TABLE X. Area and power consumption per tile of the baseline FP16 accelerator vs. _BitMoD_ at 1 GHz frequency. 

|umber<br>f PEs|Area (_µ_m2)<br>PE Array Encoder<br>Total<br>PE|Power (mW)<br>Array Encoder Total|
|---|---|---|
|6_×_8<br>8_×_8|95,498<br>–<br>95,498<br>3<br>97,090<br>2,419<br>99,509<br>|6.96<br>–<br>36.96<br>37.5<br>1.86<br>39.36|



1091 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:30:15 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [234 x 107] intentionally omitted <==**

**----- Start of picture text -----**<br>
INT8 6-bit 5-bit 4-bit 3-bit ANT OliVe BitMoD<br>17 7.5<br>15 7.0<br>13 6.5<br>11 6.0<br>9 5.5<br>0.2 0.4 0.6 0.8 1.0 0.2 0.4 0.6 0.8 1.0<br>Phi-2B EDP Llama-2-7B EDP<br>better better<br>Perplexity Perplexity<br>**----- End of picture text -----**<br>


Fig. 9: Wikitext-2 perplexity-EDP pareto plot for Phi-2B and Llama-2-7B. 

than the baseline PE, which allows to fit more _BitMoD_ PEs within the same compute area. Furthermore, the bit-serial term encoder has a tiny hardware overhead and only accounts for 2 _._ 5% of the PE array area. 

**Performance.** Fig. 7 presents the hardware performance normalized to the baseline FP16 accelerator for discriminative and generative tasks. _BitMoD_ achieves the best performance under both lossless and lossy quantization. The performance gain of _BitMoD_ mainly comes from its careful algorithm-hardware co-design of different quantization data types. Since discriminative tasks are compute-bound and mainly involve matrixmatrix multiplications, the higher throughput of _BitMoD_ PE leads to better performance than the baseline PE. In contrast, OliVe requires more complicated PEs with significant hardware overhead to accommodate the outliers, which have a much wider numerical range (e.g., _{_ 24 _, ...,_ 192 _}_ at 4-bit precision). In comparison, the _BitMoD_ data type has a small value range and can be efficiently processed with the proposed unified bit-serial representation. Regarding memorybound generative tasks, _BitMoD_ can quantize LLM weights to very low precision such as 3-bit, which offers significant memory saving while maintaining good perplexity. On the contrary, ANT and OliVe do not natively support per-group quantization due to the lack of dedicated dequantization hardware, and cannot maintain acceptable model quality under perchannel quantization using 3-bit weight precision. Therefore, they must adopt a higher weight precision to compensate for the significant degradation in perplexity. Overall, the lossless _BitMoD_ achieves 1 _._ 99 _×_ and 2 _._ 41 _×_ speedup for discriminative and generative tasks, respectively compared to the baseline FP16 architecture. The lossy _BitMoD_ achieves 1 _._ 72 _×_ / 1 _._ 56 _×_ and 1 _._ 66 _×_ / 1 _._ 39 _×_ speedup for discriminative and generative tasks, respectively compared to ANT / OliVe. 

**Energy Consumption.** Fig. 8 presents the normalized energy breakdown of different accelerators, where the on-chip compute energy includes both buffer and core energy. The energy saving of _BitMoD_ mainly comes from the reduced weight memory footprint and efficient bit-serial PE. Both ANT and OliVe require a higher weight precision than _BitMoD_ to maintain acceptable model quality, leading to higher DRAM energy consumption. In addition, the baseline architecture uses FP16 weights, which is an overkill for LLMs since the simple INT6 data type can achieve comparable accuracy under per-group 

**==> picture [220 x 85] intentionally omitted <==**

Fig. 10: Normalized area and power of _BitMoD_ and different bit-parallel PEs. 

weight quantization. Overall, the lossless _BitMoD_ achieves 2 _._ 31 _×_ better energy efficiency over the baseline architecture across different tasks. The lossy _BitMoD_ has 1 _._ 48 _×_ and 1 _._ 31 _×_ better energy efficiency than ANT and OliVe, respectively. 

**Accuracy-Efficiency Trade-offs.** The proposed _BitMoD_ can offer good trade-offs between model accuracy and hardware efficiency. To demonstrate this, we analyze the relationship between energy-delay product (EDP) and model perplexity of Phi-2B and Llama-2-7B on Wikitext-2. We compare _BitMoD_ with ANT and OliVe under different LLM weight precision. Fig. 9 shows the resulting perplexity-EDP relationship for the studied two LLMs and three accelerators. Note that while the 5-bit precision is not presented explicitly, _BitMoD_ can be easily extended to perform bit-serial INT5 computation using its Booth encoder. Similarly, the custom data types introduced by ANT and OliVe can be extended to 5-bit precision based on their data type definition. As indicated in Fig. 9, the lower left region indicates a _better_ trade-off between perplexity and EDP. Although ANT and OliVe propose different algorithmhardware co-design approaches for LLM acceleration, they only leverage the per-channel quantization granularity that fails to preserve the model quality at very low precision, and they lack a unified architecture to efficiently support different data types and precision. In contrast, _BitMoD_ exploits new data types tailored for per-group quantization and adopts an efficient bit-serial computing paradigm to support various data types. Hence, _BitMoD_ can always sit on the Pareto frontier. 

_D. Comparison to Mixed-Precision Bit-Parallel Architecture_ 

FIGNA [27] proposes a family of bit-parallel PEs for arithmetic between low-precision integer weight and floating-point activation. However, every FIGNA PE is designed separately and only supports one weight precision, which fails to offer a trade-off between model accuracy and hardware efficiency. We explore the possibility of FIGNA to support mixed-precision integer weights. We consider a baseline FIGNA-like PE that performs multiply-accumulate between FP16 activation and INT8 weight. We extend the baseline PE to support either one FP16-INT8 operation or two FP16-INT4 operations, which multiply the same FP16 activation with two INT4 weights. Fig. 10 compares the normalized area and power of different PEs. Although the FP-INT8 PE has the smallest area, adding support for mixed weight precision incurs significant hardware overhead and leads to even higher area and power consumption than the conventional FP-FP PE. This is because a bit-parallel PE computing two FP16-INT4 operations will 

1092 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:30:15 UTC from IEEE Xplore.  Restrictions apply. 

TABLE XI. Wikitext-2 and C4 perplexity ( _↓_ ) of different quantization strategies. We use per-group weight quantization with a group size of 128. For every table column, we highlight the best two perplexity results in bold. 

|TABLE XI. Wikitext-2 and C4 perplexity (_↓_) of different quantization<br>strategies. We use per-group weight quantization with a group size<br>of 128. For every table column, we highlight the best two perplexity<br>results in bold.|TABLE XI. Wikitext-2 and C4 perplexity (_↓_) of different quantization<br>strategies. We use per-group weight quantization with a group size<br>of 128. For every table column, we highlight the best two perplexity<br>results in bold.|
|---|---|
|**Bits**<br>**Method**<br>**Llama-2-7B Llama-2-13B Llama-3-8B Mean**||
|Wiki<br>C4<br>Wiki<br>C4<br>Wiki<br>C4|∆**PPL**|
|16-bit<br>FP16<br>5.47<br>6.97<br>4.88<br>6.47<br>6.13<br>8.88|0|
|4-bit<br>QuaRot<br>5.60<br>7.48<br>5.00<br>6.88<br>6.54 10.18<br>GPTQ<br>5.63<br>7.13<br>4.99<br>6.56<br>6.53<br>9.38<br>AWQ<br>5.60<br>7.12<br>4.97<br>6.56<br>6.54<br>9.39<br>OmniQ<br>5.59<br>7.12<br>4.96<br>6.56<br>6.57<br>9.50<br>_BitMoD_+ AWQ<br>**5.59**<br>**7.09**<br>**4.96**<br>**6.55**<br>**6.50**<br>**9.33**<br>_BitMoD_+ OmniQ **5.57**<br>**7.07**<br>**4.95**<br>**6.55**<br>**6.45**<br>**9.30**|0.48<br>0.24<br>0.23<br>0.25<br>**0.20**<br>**0.18**|
|3-bit<br>QuaRot<br>6.09<br>8.44<br>5.37<br>7.52<br>**7.64** 12.49<br>GPTQ<br>6.29<br>7.89<br>5.42<br>7.00<br>9.58 11.66<br>AWQ<br>6.24<br>7.81<br>5.32<br>6.95<br>8.22 11.56<br>OmniQ<br>**6.05**<br>7.76<br>5.28<br>6.99<br>8.33 12.04<br>_BitMoD_+ AWQ<br>6.07<br>**7.64**<br>**5.27**<br>**6.88**<br>7.81 **11.07**<br>_BitMoD_+ OmniQ **5.89**<br>**7.59**<br>**5.21**<br>**6.85**<br>**7.57 11.05**|1.88<br>1.51<br>1.22<br>1.28<br>**0.98**<br>**0.89**|



produce two separate outputs, doubling the cost of a floatingpoint accumulator and output register. In contrast, the bit-serial _BitMoD_ PE trades-off between weight precision and latency, which requires only one accumulator and output register for any weight precision. Consequently, _BitMoD_ offers the highest flexibility to support variable weight precision with better efficiency compared to the decomposable bit-parallel PE. 

## _E. Combining BitMoD with Other Quantization Schemes_ 

_BitMoD_ can be seamlessly integrated with existing softwareonly quantization methods by replacing their original weight quantizers that use integer data types with the extended FP4 and FP3 data types of _BitMoD_ . We demonstrate such feasibility on AWQ [30], OmniQuant [42], and SmoothQuant [52]. 

**Orthogonal to Quantization Optimization.** The original AWQ and OmniQuant adopt INT-Asym weight quantization with several algorithmic optimizations such as weight clipping and scaling factor search, leading to SOTA model performance under 4-bit and 3-bit weight precision. We evaluate the model performance when applying AWQ and OmniQuant optimizations on top of the _BitMoD_ data type. We also compare with SOTA software-only LLM quantization methods, including GPTQ [20] and QuaRot [4] under weight-only quantization. Table XI shows the Wikitext and C4 perplexity of the studied Llama models using different quantization strategies. Combining _BitMoD_ with AWQ and OmniQuant significantly outperforms other approaches. For example, applying the _BitMoD_ data type reduces the average perplexity loss of OmniQuant by 28% and 31% at 4-bit and 3-bit precision, respectively. Overall, _BitMoD_ combined with AWQ and OmniQuant achieves an average perplexity loss of _<_ 1 for both 4-bit and 3-bit weight precision, pushing the limit of LLM weight quantization to a new state-of-the-art. It’s important to note that using AWQ and OmniQuant does not inhibit the functionality of the _BitMoD_ accelerator—their optimization merely adjusts the per-group scaling factor, which is supported by the bit-serial dequantization unit of _BitMoD_ . 

TABLE XII. Wikitext-2 perplexity ( _↓_ ) when activation maintains in FP16 or is quantized to INT8 with SmoothQuant (SQ8). For weights, we use per-group quantization with a group size of 128. 

|**Weight**|**Weight**|**Llama-2-7B**|**Llama-2-7B**|**Llama-2-13B**|**Llama-2-13B**|**Llama-3-8B**|**Llama-3-8B**|
|---|---|---|---|---|---|---|---|
|**Precision**|**Datatype**|FP16|SQ8|FP16|SQ8|FP16|SQ8|
|8-bit|INT8|5.47|5.52|4.95|4.93|6.13|6.26|
|4-bit|INT4-Asym<br>_BitMoD_|5.77<br>5.72|5.83<br>5.76|5.01<br>5.01|5.09<br>5.07|6.84<br>6.73|7.05<br>6.87|
|3-bit|INT3-Asym<br>_BitMoD_|7.08<br>6.55|7.58<br>6.85|5.64<br>5.5|5.99<br>5.82|13.26<br>9.09|25.78<br>10.57|



**Orthogonal to Activation Quantization.** SmoothQuant can quantize LLM activation to INT8 with low accuracy loss. We conduct weight PTQ using _BitMoD_ and INT-Asym data types on the pre-calibrated Llama models from SmoothQuant that use INT8 activation. Table XII shows the WikiText-2 perplexity under FP16 and INT8 activation using different quantized weight precision and data types. The perplexity improvement of _BitMoD_ over INT-Asym remains after quantizing activation to INT8 using SmoothQuant, and the improvement is particularly pronounced at lower precision (i.e., 3-bit). For instance, on Llama-3-8B, _BitMoD_ improves the perplexity by 15 _._ 21 compared to INT3-Asym after applying SmoothQuant. Notably, on Llama-2-7B, the perplexity of _BitMoD_ weight with INT8 activation is even better than INT-Asym weight with FP16 activation, which demonstrates the potential of _BitMoD_ for further reducing the LLM memory footprint under a target model quality. 

## VI. RELATED WORK 

**DNN Accelerators.** There is an abundance of prior work on DNN accelerators [3], [5], [12], [23]–[28], [39], [40], [43], [45], [48], [50], [54], [55], [59]. These accelerators propose specialized processing elements and data flow to match the computational characteristics and memory access pattern of DNNs. Some accelerators exploit value sparsity to accelerate small-scale DNNs with the help of retraining [23], [24], [48], [50], [59]. Other works target low-precision DNN acceleration based on model quantization [25]–[27], [39], [40], [54], [55]. Among them, [25], [26], [40] introduce custom data types to better fit the value distribution of DNNs. Another line of works relies on bit-serial computing to scale the performance with lower operand precision, and leverages bit-level sparsity to skip ineffectual bit operations [3], [5], [12], [28], [43], [45]. The proposed _BitMoD_ combines the benefits of quantization and bit-serial computing to efficiently trade-off between weight precision and hardware efficiency. 

**LLM Quantization.** Numerous algorithmic studies have proposed quantization solutions to reduce the memory footprint of LLMs [4], [11], [15], [19], [20], [30], [31], [42], [49], [52]. Most of these works rely on asymmetric integer quantization for LLM weights while applying other techniques to optimize the quantization parameters. The proposed _BitMoD_ data type is orthogonal to many of these works and can be synergistically combined with different quantization optimizations. 

1093 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:30:15 UTC from IEEE Xplore.  Restrictions apply. 

## VII. CONCLUSION 

In this paper, we introduce _BitMoD_ , an algorithm-hardware co-design scheme for efficient LLM acceleration. On the algorithm side, _BitMoD_ designs new data types that are tailored for per-group LLM weight quantization at very low precision. By intelligently repurposing the redundant zero value to an additional number, _BitMoD_ extends the resolution or range of 3-bit-and 4-bit floating-point data types. Moreover, the _BitMoD_ quantization framework can be seamlessly integrated with existing software-only quantization methods to further improve the model performance. On the hardware side, _BitMoD_ proposes a unified bit-serial representation for diverse low-precision data types and an efficient bit-serial PE to process quantized weight and FP16 activation. Our evaluation demonstrates that _BitMoD_ significantly outperforms existing LLM quantization methods, pushing the limit of LLM weight quantization to a new state-of-the-art. Compared to prior accelerators ANT / OliVe, _BitMoD_ achieves 1 _._ 69 _×_ / 1 _._ 48 _×_ speedup and 1 _._ 48 _×_ / 1 _._ 31 _×_ better energy efficiency, while being able to support diverse weight precision to offer a good trade-off between model accuracy and hardware efficiency. 

## ACKNOWLEDGMENT 

This project is supported in part by Intel Corporation, the National Science Foundation under Grant No. 2339084, and the Engineering and Physical Sciences Research Council (EPSRC) under Grant No. EP/S030069/1. We would like to thank Jordan Dotzel, Hongxiang Fan, Stylianos I. Venieris, Alexandros Kouris, Mahesh Iyer, Grace Zgheib, Sergey Gribok, and the anonymous reviewers for their constructive feedback. 

## REFERENCES 

- [1] 01-AI, “01-ai/yi-6b.” [Online]. Available: https://huggingface.co/01ai/Yi-6B 

- [2] Abdelfattah Lab, “BitMoD Artifacts,” Nov. 2024. [Online]. Available: https://doi.org/10.5281/zenodo.14252531 

- [3] J. Albericio, A. Delmas, P. Judd, S. Sharify, G. O’Leary, R. Genov, and A. Moshovos, “Bit-pragmatic deep neural network computing,” _IEEE/ACM 50th Annual International Symposium on Microarchitecture (MICRO)_ , 2017. 

- [4] S. Ashkboos, A. Mohtashami, M. L. Croci, B. Li, M. Jaggi, D. Alistarh, T. Hoefler, and J. Hensman, “QuaRot: Outlier-free 4-bit inference in rotated llms,” _arXiv preprint arXiv:2404.00456_ , 2024. 

- [5] O. M. Awad, M. Mahmoud, I. E. Vivancos, A. H. Zadeh, C. Bannon, A. Jayarajan, G. Pekhimenko, and A. Moshovos, “FPRaker: A processing element for accelerating neural network training,” _IEEE/ACM 54th Annual International Symposium on Microarchitecture (MICRO)_ , 2020. 

- [6] R. Balasubramonian, A. B. Kahng, N. Muralimanohar, A. Shafiee, and V. Srinivas, “CACTI 7: New tools for interconnect exploration in innovative off-chip memories,” _ACM Trans. Archit. Code Optim._ , vol. 14, no. 2, June 2017. 

- [7] Y. Bisk, R. Zellers, R. L. Bras, J. Gao, and Y. Choi, “PIQA: Reasoning about physical commonsense in natural language,” _arXiv preprint arXiv:1911.11641_ , 2019. 

- [8] A. D. Booth, “A signed binary multiplication technique,” _Quarterly Journal of Mechanics and Applied Mathematics_ , vol. 4, pp. 236–240, 1951. 

- [9] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. HerbertVoss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei, “Language Models are Few-Shot Learners,” in _Advances in Neural Information Processing Systems_ , 2020, pp. 1877–1901. 

- [10] J. Chee, Y. Cai, V. Kuleshov, and C. D. Sa, “QuIP: 2-bit quantization of large language models with guarantees,” _arXiv preprint arXiv:2307.13304_ , 2023. 

- [11] M. Chen, W. Shao, P. Xu, J. Wang, P. Gao, K.-C. Zhang, Y. Qiao, and P. Luo, “EfficientQAT: Efficient quantization-aware training for large language models,” _arXiv preprint arXiv:2407.11062_ , 2024. 

- [12] Y. Chen, J. Meng, J.-S. Seo, and M. S. Abdelfattah, “BBS: Bi-directional bit-level sparsity for deep learning acceleration,” _57th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , 2024. 

- [13] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, “Think you have solved question answering? try ARC, the ai2 reasoning challenge,” _arXiv preprint arXiv:1803.05457_ , 2018. 

- [14] S. Dai, R. Venkatesan, H. Ren, B. Zimmer, W. J. Dally, and B. Khailany, “VS-Quant: Per-vector scaled quantization for accurate low-precision neural network inference,” in _Proceedings of Machine Learning and Systems (MLSys)_ , 2021. 

- [15] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, “LLM.int8(): 8-bit matrix multiplication for transformers at scale,” _arXiv preprint arXiv:2208.07339_ , 2022. 

- [16] T. Dettmers, M. Lewis, S. Shleifer, and L. Zettlemoyer, “8-bit optimizers via block-wise quantization,” _arXiv preprint arXiv:2110.02861_ , 2022. 

- [17] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient finetuning of quantized llms,” _arXiv:2305.14314_ , 2023. 

- [18] J. Dodge, A. Marasovic, G. Ilharco, D. Groeneveld, M. Mitchell, and M. Gardner, “Documenting large webtext corpora: A case study on the colossal clean crawled corpus,” in _Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2021. 

- [19] J. Dotzel, Y. Chen, B. Kotb, S. Prasad, G. Wu, S. Li, M. S. Abdelfattah, and Z. Zhang, “Learning from students: Applying t-distributions to explore accurate and efficient formats for llms,” _arXiv preprint arXiv:2405.03103_ , 2024. 

- [20] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, “GPTQ: Accurate post-training compression for generative pretrained transformers,” _arXiv preprint arXiv:2210.17323_ , 2022. 

- [21] L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac’h, H. Li, K. McDonell, N. Muennighoff, C. Ociepa, J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron, L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou, “A framework for few-shot language model evaluation,” 12 2023. [Online]. Available: https://zenodo.org/records/10256836 

- [22] A. Gholami, Z. Yao, S. Kim, C. Hooper, M. W. Mahoney, and K. Keutzer, “Ai and memory wall,” _IEEE Micro_ , 2024. 

- [23] A. Gondimalla, N. Chesnut, M. Thottethodi, and T. N. Vijaykumar, “SparTen: A sparse tensor accelerator for convolutional neural networks,” _Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , 2019. 

- [24] A. Gondimalla, M. Thottethodi, and T. N. Vijaykumar, “Eureka: Efficient tensor cores for one-sided unstructured sparsity in dnn inference,” _56th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , 2023. 

- [25] C. Guo, J. Tang, W. Hu, J. Leng, C. Zhang, F. Yang, Y.-B. Liu, M. Guo, and Y. Zhu, “OliVe: Accelerating large language models via hardware-friendly outlier-victim pair quantization,” _ACM/IEEE 50th Annual International Symposium on Computer Architecture (ISCA)_ , 2023. 

- [26] C. Guo, C. Zhang, J. Leng, Z. Liu, F. Yang, Y.-B. Liu, M. Guo, and Y. Zhu, “ANT: Exploiting adaptive numerical data type for low-bit deep neural network quantization,” _IEEE/ACM 55th Annual International Symposium on Microarchitecture (MICRO)_ , 2022. 

- [27] J. Jang, Y. Kim, J. Lee, and J.-J. Kim, “FIGNA: Integer unit-based accelerator design for fp-int gemm preserving numerical accuracy,” _IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ , 2024. 

- [28] P. Judd, J. Albericio, and A. Moshovos, “Stripes: Bit-serial deep neural network computing,” _49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , 2016. 

- [29] S.-J. Li, Z. Yang, D. Reddy, A. Srivastava, and B. Jacob, “DRAMsim3: A cycle-accurate, thermal-capable dram simulator,” _IEEE Computer Architecture Letters_ , vol. 19, pp. 106–109, 2020. 

- [30] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han, “AWQ: Activation-aware weight quantization for llm compression and acceleration,” in _Proceedings of Machine Learning and Systems (MLSys)_ , 2024. 

- [31] Z. Liu, B. O˘guz, C. Zhao, E. Chang, P. Stock, Y. Mehdad, Y. Shi, R. Krishnamoorthi, and V. Chandra, “LLM-QAT: Data-free quan- 

1094 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:30:15 UTC from IEEE Xplore.  Restrictions apply. 

tization aware training for large language models,” _arXiv preprint arXiv:2305.17888_ , 2023. 

- [32] J. Meng, Y. Liao, A. Anupreetham, A. Hasssan, S. Yu, H.-s. Suh, X. Hu, and J.-s. Seo, “Torch2Chip: An end-to-end customizable deep neural network compression and deployment toolkit for prototype hardware accelerator design,” in _Proceedings of Machine Learning and Systems (MLSys)_ , 2024. 

- [33] S. Merity, C. Xiong, J. Bradbury, and R. Socher, “Pointer sentinel mixture models,” _arXiv preprint arXiv:1609.07843_ , 2016. 

- [34] Meta, “Meta llama.” [Online]. Available: https://github.com/metallama/llama 

- [35] Meta, “Meta llama 3.” [Online]. Available: https://github.com/metallama/llama3 

- [36] Microsoft, “microsoft/phi-2.” [Online]. Available: https://huggingface. co/microsoft/phi-2 

- [37] NVIDIA, “Jetson TX2 Module.” [Online]. Available: https://developer. nvidia.com/embedded/jetson-tx2 

- [38] Open Compute Project, “OCP Microscaling Formats (MX) Specification.” [Online]. Available: https://www.opencompute.org/ documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf 

- [39] E. Park, D. Kim, and S. Yoo, “Energy-efficient neural network accelerator based on outlier-aware low-precision computation,” _ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA)_ , 2018. 

- [40] B. D. Rouhani, R. Zhao, V. Elango, R. Shafipour, M. Hall, M. Mesmakhosroshahi, A. More, L. Melnick, M. Golub, G. Varatkar, L. Shao, G. Kolhe, D. Melts, J. Klar, R. L’Heureux, M. Perry, D. Burger, E. S. Chung, Z. Deng, S. Naghshineh, J. Park, and M. Naumov, “With shared microexponents, a little shifting goes a long way,” _ACM/IEEE 50th Annual International Symposium on Computer Architecture (ISCA)_ , 2023. 

   - [52] G. Xiao, J. Lin, M. Seznec, J. Demouth, and S. Han, “SmoothQuant: Accurate and efficient post-training quantization for large language models,” _arXiv preprint arXiv:2211.10438_ , 2022. 

   - [53] Z. Yao, R. Y. Aminabadi, M. Zhang, X. Wu, C. Li, and Y. He, “ZeroQuant: Efficient and affordable post-training quantization for largescale transformers,” _arXiv preprint arXiv:2206.01861_ , 2022. 

   - [54] A. H. Zadeh, I. Edo, O. M. Awad, and A. Moshovos, “GOBO: Quantizing attention-based nlp models for low latency and energy efficient inference,” _IEEE/ACM 53rd Annual International Symposium on Microarchitecture (MICRO)_ , 2020. 

   - [55] A. H. Zadeh, M. Mahmoud, A. Abdelhadi, and A. Moshovos, “Mokey: enabling narrow fixed-point inference for out-of-the-box floating-point transformer models,” _ACM/IEEE 49th Annual International Symposium on Computer Architecture (ISCA)_ , 2022. 

   - [56] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi, “HellaSwag: Can a machine really finish your sentence?” in _Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2019. 

   - [57] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. T. Diab, X. Li, X. V. Lin, T. Mihaylov, M. Ott, S. Shleifer, K. Shuster, D. Simig, P. S. Koura, A. Sridhar, T. Wang, and L. Zettlemoyer, “Opt: Open pre-trained transformer language models,” _arXiv preprint arXiv:2205.01068_ , 2022. 

   - [58] Y. Zhao, C.-Y. Lin, K. Zhu, Z. Ye, L. Chen, S. Zheng, L. Ceze, A. Krishnamurthy, T. Chen, and B. Kasikci, “Atom: Low-bit quantization for efficient and accurate llm serving,” in _Proceedings of Machine Learning and Systems (MLSys)_ , 2024. 

   - [59] X. Zhou, Z. Du, Q. Guo, S. Liu, C. Liu, C. Wang, X. Zhou, L. Li, T. Chen, and Y. Chen, “Cambricon-S: Addressing irregularity in sparse neural networks through a cooperative software/hardware approach,” _51st Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , 2018. 

- [41] K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi, “WinoGrande: An adversarial winograd schema challenge at scale,” _arXiv preprint arXiv:1907.10641_ , 2019. 

- [42] W. Shao, M. Chen, Z. Zhang, P. Xu, L. Zhao, Z. Li, K. Zhang, P. Gao, Y. J. Qiao, and P. Luo, “OmniQuant: Omnidirectionally calibrated quantization for large language models,” _arXiv preprint arXiv:2308.13137_ , 2024. 

- [43] S. Sharify, A. D. Lascorz, M. Mahmoud, M. Nikolic, K. Siu, D. M. Stuart, Z. Poulos, and A. Moshovos, “Laconic deep learning inference acceleration,” _ACM/IEEE 46th Annual International Symposium on Computer Architecture (ISCA)_ , 2019. 

- [44] Y. Sheng, L. Zheng, B. Yuan, Z. Li, M. Ryabinin, D. Y. Fu, Z. Xie, B. Chen, C. W. Barrett, J. Gonzalez, P. Liang, C. R´e, I. Stoica, and C. Zhang, “FlexGen: High-throughput generative inference of large language models with a single gpu,” in _International Conference on Machine Learning (ICML)_ , 2023. 

- [45] M. Shi, V. Jain, A. Joseph, M. Meijer, and M. Verhelst, “BitWave: Exploiting column-based bit-level sparsity for deep learning acceleration,” _Proceedings of the 30th IEEE International Symposium on HighPerformance Computer Architecture (HPCA)_ , 2024. 

- [46] R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Ng, and C. Potts, “Recursive deep models for semantic compositionality over a sentiment treebank,” in _Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2013. 

- [47] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozi`ere, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample, “Llama: Open and efficient foundation language models,” _arXiv preprint arXiv:2302.13971_ , 2023. 

- [48] Y. Wang, C. Zhang, Z. Xie, C. Guo, Y. Liu, and J. Leng, “Dual-side sparse tensor core,” _ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)_ , 2021. 

- [49] X. Wu, H. Xia, S. Youn, Z. Zheng, S. Chen, A. Bakhtiari, M. Wyatt, R. Y. Aminabadi, Y. He, O. Ruwase, L. Song, and Z. Yao, “ZeroQuant(4+2): Redefining llms quantization with a new fp6-centric strategy for diverse generative tasks,” _arXiv preprint arXiv:2312.08583_ , 2023. 

- [50] Y. N. Wu, P.-A. Tsai, S. Muralidharan, A. Parashar, V. Sze, and J. S. Emer, “HighLight: Efficient and flexible dnn acceleration with hierarchical structured sparsity,” _56th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , 2023. 

- [51] H. Xia, Z. Zheng, X. Wu, S. Chen, Z. Yao, S. Youn, A. Bakhtiari, M. Wyatt, D. Zhuang, Z. Zhou, O. Ruwase, Y. He, and S. L. Song, “Quant-llm: Accelerating the serving of large language models via fp6centric algorithm-system co-design on modern gpus,” in _USENIX Annual Technical Conference (ATC)_ , 2024. 

1095 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:30:15 UTC from IEEE Xplore.  Restrictions apply. 

## APPENDIX 

## _A. Abstract_ 

This artifact appendix describes how to access and evaluate the BitMoD artifact [2] to reproduce the experiments as performed in Section V. 

## _B. Artifact Check-List (Meta-Information)_ 

- **Program:** Python, Shell script 

- **Runtime Environment:** Ubuntu 20.04 

- **Hardware:** NVIDIA GPU with _≥_ 40 GB of VRAM (e.g., A6000). 

- **Model (LLM):** OPT-1.3B, Phi-2B, Yi-6B, Llama-2-7B, Llama-2-13B, Llama-3-8B. 

- **Dataset:** Wikitext, C4. 

- **Experiments:** LLM perplexity evaluation in Section V-B and Section V-E. Accelerator performance evaluation in Section V-C. 

- **Disk space required (approximately):** 512 GB. 

- **How much time is needed to complete experiments (approximately)?:** 50 hours. 

- **Publicly available?:** Yes. 

- **Archived:** https://doi.org/10.5281/zenodo.14252531 

## _C. Description_ 

_1) How to Access:_ We maintain a publicly-available repository [2] where we have open-sourced all of our artifacts. 

_2) Hardware Dependencies:_ One NVIDIA GPU with at least 40 GB of VRAM (e.g., A6000), in addition to a normal desktop computer with at least 512 GB of free disk space. 

_3) Software Dependencies:_ All experiments require Conda for managing virtual Python environments. The quantization experiments require CUDA. Other requirements are automatically installed by scripts in the following sections. When running experiments, please use a tmux session to make sure long-running jobs are not killed unexpectedly. 

## _D. Installation_ 

For artifact evaluation, begin by downloading the top-level repository from Zenodo: 

$ wget −O BitMoD.zip https://zenodo.org/records/14252531/ files/BitMoD−HPCA−25.zip $ unzip BitMoD.zip 

The artifact is inside the unzipped **BitMoD-HPCA-25** repository, which contains five sub-folders, each targeting one set of experiments: 

- 1) **bitmod-quant** , which runs the baseline weight-only quantization with different data types. This can reproduce the results in Table VI and Table VIII. 

- 2) **bitmod-sim** , contains a custom simulator that calculates the latency and energy of the _BitMoD_ accelerator. This can reproduce the results in Fig. 7 and Fig. 8. 

- 3) **AWQ-BitMoD** , which runs AWQ [30] with integer and _BitMoD_ data types. This can reproduce the AWQ results in Table XI. 

- 4) **OmniQuant-BitMoD** , which runs OmniQuant [42] with integer and _BitMoD_ data types. This can reproduce the OmniQuant results in Table XI. 

- 5) **SmoothQuant-BitMoD** , which runs SmoothQuant [52] with integer and _BitMoD_ data types for weight quantization. This can reproduce the results in Table XII. 

Please go to every sub-folder and refer to the corresponding ‘README.md’ for detailed setup instructions. Note that AWQ, OmniQuant, and SmoothQuant will require different conda environments. Hence, please change back to the base environment after installing a conda environment before creating the next. 

For example, the AWQ environment can be created by running: 

- $ **cd** AWQ−BitMoD 

- $ conda create −n awq−bitmod python=3.10 −y $ conda activate awq−bitmod 

- # Follow ‘README.md’ inside AWQ−BitMoD folder to set 

- up other dependencies. 

- $ conda deactivate # change back to the base environment 

The OmniQuant and SmoothQuant environments can be created in a similar way by repeating the above step and following their ‘README.md’ to set up other dependencies. 

## _E. Experiment workflow_ 

Once the environment is set up, we will conduct five sets of experiments, each corresponding to one of the five sub-folders within the **BitMoD-HPCA-25** repository. 

_1) BitMoD Weight-only Quantization:_ Run the basic LLM weight-only quantization experiments to reproduce the results in Table VI and Table VIII. 

- $ **cd** bitmod quant 

$ conda activate awq−bitmod 

In ‘run_exp.sh’, modify the ‘export’ command by specifying the HuggingFace home directory, ‘HF_HOME’, on your computer. By default, this can be set to your home directory. 

- $ **export** HF HOME=”your/HF HOME/directory” 

Then run the following: 

- $ bash run exp.sh 

The perplexity result will be saved in the folder called ‘results_quant’. 

- _2) BitMoD Hardware Simulation:_ Before running the sim- 

- ulator, go to ‘bitmod_sim’ of the repository: 

   - $ **cd** bitmod sim 

   - $ conda activate awq−bitmod 

1096 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:30:15 UTC from IEEE Xplore.  Restrictions apply. 

In ‘run_shape_profile.sh’, modify the ‘export’ command by specifying the HuggingFace home directory, ‘HF_HOME’, on your computer: 

## $ **export** HF HOME=”your/HF HOME/directory” 

Then generate the model shape information that can be passed to the accelerator simulator: 

$ bash run shape profile.sh 

## $ **export** HF HOME=”your/HF HOME/directory” 

In every shell script, you need to change the parameter of _--model_ flag to the LLM path in your computer. By default, this can be set to the LLM directory from the official HuggingFace website. For example, inside the script ‘llama-2-13b-int.sh’, the Llama-2-7B model path can be specified with: 

## −−model meta−llama/Llama−2−7b−hf 

Next, run different simulators for the baseline FP16 accelerator, ANT, OliVe, and _BitMoD_ : 

$ python test baseline.py −−is generation $ python test ant.py −−is generation $ python test olive.py −−is generation $ python test bitmod.py −−is generation −−is lossless 

The flag _--is generation_ is optional. When enabled / disabled, it will evaluate the hardware performance of generative / discriminative tasks. The flag _--is lossless_ is optional for _BitMoD_ . When enabled / disabled, it will evaluate the hardware performance of lossless / lossy _BitMoD_ quantization. 

Finally, to generate Fig. 7 and Fig. 8 of the paper, go to ‘bitmod_sim/plot’ directory and run the Jupyter notebooks inside. Note that the cycle and energy numbers are the same as those output by the simulators. 

## _3) AWQ:_ Go to the ‘AWQ-BitMoD’ directory: 

$ **cd** AWQ−BitMoD $ conda activate awq−bitmod 

In ‘run_awq.sh’ and ‘run_eval_ppl.sh’, modify the first ‘export’ command by specifying the HuggingFace home directory, ‘HF_HOME’, on your computer: 

$ **export** HF HOME=”your/HF HOME/directory” 

Then, run the following two commands separately: 

$ bash run awq.sh # will take several hours $ bash run eval ppl.sh 

The perplexity results will be saved in the folder called ‘results’. You can compare these with the AWQ results in Table XI. 

_4) OmniQuant:_ Go to the ‘OmniQuant-BitMoD’ directory: 

$ **cd** OmniQuant−BitMoD $ conda activate omniquant−bitmod 

The comprehensive scripts to reproduce the Table XI OmniQuant results are available in the ‘scripts’ directory. Before running any command in the scripts, execute the following ‘export’ command and specify the HuggingFace home directory, ‘HF_HOME’, on your computer: 

After changing the _--model_ flag to the correct model path, copy and execute every script’s python command under the ‘OmniQuant-BitMoD’ directory. You may check the perplexity results at the end of the log file specified by the _--output dir_ flag in every command, and compare those with the OmniQuant results in Table XI. 

- _5) SmoothQuant:_ Go to the ‘SmoothQuant-BitMoD’ 

- directory: 

$ **cd** SmoothQuant−BitMoD $ conda activate smoothquant−bitmod 

In ‘run_experiments.sh’, modify the ‘export’ command by specifying the HuggingFace home directory, ‘HF_HOME’, on your computer: 

$ **export** HF HOME=”your/HF HOME/directory” 

Then, run the following command: 

$ bash run experiments.sh 

The perplexity results will be saved in the folder called ‘results_mod’. You can compare these results with the SmoothQuant results in Table XII. 

## _F. Evaluation and expected results_ 

There are five result folders after running the above experiments: 

- 1) bitmod-quant/results_quant, contains the perplexity results in Table VI and Table VIII. 

- 2) bitmod-sim/plot, contains two Jupyter notebooks to reproduce Fig. 7 and Fig. 8, respectively. 

- 3) AWQ-BitMoD/results, contains the AWQ results in Table XI. 

- 4) OmniQuant-BitMoD/log, contains the OmniQuant results in Table XI. 

- 5) SmoothQuant-BitMoD/results_mod, contains the SmoothQuant results in Table XII. 

## _G. Methodology_ 

Submission, reviewing and badging methodology: 

- https://www.acm.org/publications/policies/artifactreview-and-badging-current 

- https://cTuning.org/ae 

1097 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:30:15 UTC from IEEE Xplore.  Restrictions apply. 

