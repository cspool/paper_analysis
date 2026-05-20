2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO) 

# SOFA: A Compute-Memory Optimized Sparsity Accelerator via Cross-Stage Coordinated Tiling 

Huizheng Wang _[∗]_ , Jiahao Fang _[∗]_ , Xinru Tang _[∗]_ , Zhiheng Yue _[∗]_ , Jinxi Li _[∗]_ , Yubin Qin _[∗]_ , Sihan Guan _[∗]_ , Qinze Yang _[∗]_ , Yang Wang _[∗]_ , Chao Li _[†]_ , Yang Hu _[∗‡]_[�] , Shouyi Yin _[∗‡]_ 

> _∗_ School of Integrated Circuits, Tsinghua University, Beijing, China, 100084 

> _†_ School of Computer Science and Engineering, Shanghai Jiao Tong University, Shanghai, China, 200240 

> _‡_ Shanghai Artificial Intelligence Laboratory, Shanghai, China, 200433 

> � Corresponding author, hu yang@tsinghua.edu.cn 

_**Abstract**_ **—Benefiting from the self-attention mechanism, Transformer models have attained impressive contextual comprehension capabilities for lengthy texts. The requirements of highthroughput inference arise as the large language models (LLMs) become increasingly prevalent, which calls for large-scale token parallel processing (LTPP). However, existing dynamic sparse accelerators struggle to effectively handle LTPP, as they solely focus on separate stage optimization, and with most efforts confined to computational enhancements. By re-examining the endto-end flow of dynamic sparse acceleration, we pinpoint an everoverlooked opportunity that the LTPP can exploit the intrinsic coordination among stages to avoid excessive memory access and redundant computation. Motivated by our observation, we present SOFA, a cross-stage compute-memory efficient algorithmhardware co-design, which is tailored to tackle the challenges posed by LTPP of Transformer inference effectively. We first propose a novel leading zero computing paradigm, which predicts attention sparsity by using log-based add-only operations to avoid the significant overhead of prediction. Then, a distributed sorting and a sorted updating FlashAttention mechanism are proposed with cross-stage coordinated tiling principle, which enables finegrained and lightweight coordination among stages, helping optimize memory access and latency. Further, we propose a SOFA accelerator to support these optimizations efficiently. Extensive experiments on 20 benchmarks show that SOFA achieves** 9 _._ 5 _×_ **speed up and** 71 _._ 5 _×_ **higher energy efficiency than Nvidia A100 GPU. Compared to eight SOTA accelerators, SOFA achieves an average** 15 _._ 8 _×_ **energy efficiency,** 10 _._ 3 _×_ **area efficiency and** 9 _._ 3 _×_ **speed up, respectively.** 

_**Index Terms**_ **—Transformer, attention, sparsity accelerator, cross-stage tiling, top-k, FlashAttention, software-hardware codesign.** 

## I. INTRODUCTION 

Remarkable success has been witnessed recently in the development of Transformer architecture [1], for both natural language processing (NLP) [2]–[10] and computer vision (CV) tasks [11]–[19]. The impressive capabilities of Transformers greatly stems from their _self-attention_ module, which excels at extracting global context information [20]. Typically, selfattention modules take three matrices as their inputs: namely, **Q** (query), **K** (key) and **V** (value). First, an attention matrix **A** _∈_ R _[S][×][S]_ is obtained by multiplying **Q** and **K** , where _S_ is sequence length. Next, **A** goes through the softmax function for normalization, then is multiplied by **V** for the final output. 

Large language models (LLMs) have driven the transformer architecture to unprecedented levels of complexity 

**==> picture [247 x 91] intentionally omitted <==**

**----- Start of picture text -----**<br>
Memory and Computation Overheads Breakdown<br>���� �<br>����� ��� ����� ��� ����������� ��� ����� ���<br>��� ���<br>��� ��<br>��� ��<br>��� ��<br>�� ��� ��� ��� ���� �� �� ��� ��� ����<br>�������� �����<br>Memory Footprint Computation Ratio<br>**----- End of picture text -----**<br>


Fig. 1. Transformer memory and computation breakdown for long sequence. 

and capability, particularly in handling extended sequence lengths [21]. This evolution places heightened demands on inference capabilities and throughput [22], critically impacting the performance of key transformer components: the attention module, feed-forward network (FFN) module, and the querykey-value (QKV) computations. 

Traditionally, in Transformers designed for smaller sequence lengths( _≤_ **2k** ), the FFN module typically presented the main bottleneck due to its dense computational requirements [23], [24]. However, with recent advancements in processing long text, where sequence lengths can exceed 128,000 characters [25]–[27], the performance bottleneck is shifting from the FFN to the attention module. Our detailed profiling indicates that as sequence lengths surpass 32,000 characters, the attention module becomes the dominant factor affecting inference time, as shown in Fig.1. This shift is primarily because the complexity of the attention mechanism scales quadratically with sequence length, making it increasingly challenging to manage as sequences extend. 

_Dynamic sparsity (DS) acceleration_ [23], [28]–[34] have emerged as a promising solution to mitigate the latency issue of self-attention. The key idea is to predict vital Q-K pairs at runtime and calculate attention based on these vital pairs to reduce the inference latency. Typically, it consists of three stages. A _pre-compute stage_ firstly estimates the matrix **A** (denoted as **A**[ˆ] ). Then, a _top-k stage_ picks the vital Q-K pairs. In the subsequent _formal computing stage_ , self-attention is calculated only based on the vital pairs. 

The need for _**high parallelism of dynamic sparsity token processing in the context of LLM inference**_ is increasing, 

979-8-3503-5057-9/24/$31.00 ©2024 IEEE 1247 DOI 10.1109/MICRO61859.2024.00093 Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:19:49 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [510 x 153] intentionally omitted <==**

**----- Start of picture text -----**<br>
Stage 1: Pre-compute Stage 2: Top-k Sort  Stage 3: Formal Computation Traditional Dynamic<br>Sparsity<br>SRAM capacity exceeded T bit Q 4  Pre-Atten (4 bit) K (4 bit) Store to  Computation  DRAM Inefficient  row-wiseRead by  Whole-row-processing  Top-k sort Style (4 bit) Top-k Mask bit Q 16  K (16 bit)(16 bit) A SRAM capacity exceeded Store to DRAM � row-wiseRead by  single-stage latencyserialization + long Rigorous stage  Softm.(A) (16 bit) x Long Processing  bit16 V Latency = bit16 O Pre-compute Separate Optimization for each stageOpportunity Top-k Missed<br>Challenge 1: Heavy Pre-compute Cost Chall. 2: High Latency and Memory Access  Challenge 3: Unaffordable  end-to-end Latency cost Formal-compute (Cross-Stage-Tiling)<br>Differential leading zero summation(DLZS) Sphere-search-aided distributed sorting(SADS)  Sorted-updating  Reuse-aware schedule scheme (RASS) Proposed SOFA<br>Converter << resources and improve Reduce computational efficiency s/4 s/4 s s/4 s/4 dataflow ShortensOptimized tiling latency FlashAttention(SU-FA)and latencycomplexityReduce  QueryQuery Enhances data reuse Pre-compute Cross-stage Optimization Top-k<br>Max-value ensuring circuit avoids mistakes Reorder KV executing order<br>128x32  Optimized data  Flexible-input  PE line 0 PE line 0 to reduce memory access<br>Shift-Adder Array flow enhances data reuse supported SADS engine  PE line 1 PE line 1 Formal-compute<br>PE line 127 PE line 127 FSM End-to-End OptimizationComprehensive<br>DRAM<br>DRAM<br>Challenges<br>SW<br>Designs<br>Output<br>HW Elimin. Config. Index BM-2 BM-4 Reuse Awa.  Scheduler MAX Ensure Tiles Synch.<br>���<br>**----- End of picture text -----**<br>


Fig. 2. Dynamic sparsity challenges for LTPP and SOFA’s software and hardware co-design. 

especially during the prefill stage. In this stage, entire contexts are processed simultaneously, favoring high token parallelism to enhance efficiency. This scenario is especially meaningful as modern LLM inference often employs separate deployments for the prefill and decode stages [35], [36]. Moreover, the advent of speculative inference [37] can transform decode operations into prefill tasks, further emphasizing the need for efficient large-scale token processing parallelism (LTPP). 

However, supporting dynamic sparsity with large-scale token parallel processing would present prohibitive overheads, as shown in Fig. 2. This is because, firstly, current dynamic sparsity acceleration solutions lack efficient prediction schemes to reduce computation complexity. Though calculating selfattention based on vital Q-K pairs can be beneficial in reducing compute and memory consumption, the newly introduced _precompute_ and _top-k_ stages consume non-trivial computational and memory resources when large amounts of tokens are processed, which can even offset the benefits brought by sparsity acceleration methods in some cases. Our characterization depicts that even with 4-bit during the _prediction stage_ and 16bit during the _formal stage_ , the power overhead of prediction is already 1 _._ 4 _×_ that of formal computing when top- _k_ equals 20%. Unfortunately, the overhead in prediction will further rise sharply with increased parallelism. 

Secondly, the processing stages in current dynamic sparsity acceleration are not designed to be partitionable, and miss the opportunity to support fine-grained pipelining, which would enable more efficient processing. The _top-k_ sorting must be based on the readiness of a whole row of the Pre-Atten ( **A**[ˆ] ) matrix. In LTPP scenarios, the increased delay in processing each stage accumulates continuously, ultimately resulting in a significant increase in end-to-end latency. This “wholerow-processing” style also increases the amount of intermediate data, resulting in a substantial rise in DRAM access requirements. Fig. 3 shows the memory access time (MAT) of two SOTA Transformer accelerators when scaled to process multiple tokens. The increase in parallelism leads to a sharp rise in off-chip memory access and surging MAT. On average, the MAT ratio rises to 72%, overshadowing computation time 

**==> picture [240 x 82] intentionally omitted <==**

**----- Start of picture text -----**<br>
Sparsity Accelerator  for Parallel Processing with 2MB SRAM<br>����������� �����������<br>% ���������������� ���������� ������������� ��������������<br>���<br>��<br>��<br>�� ��������<br>�������<br>1 512 1 512 1 256 1 256 1 128 1 128 1 8<br>Energon FACT Energon FACT Energon FACT Energon FACT<br>���<br>��� ��� ��� ���<br>���<br>54%<br>Latency breakdown<br>**----- End of picture text -----**<br>


Fig. 3. MAT for SOTA dynamic sparsity accelerators (FACT [23], Energon [34]) with diverse parallelisms. 

and becoming the primary bottleneck. 

Thirdly, current dynamic sparsity acceleration solutions do not exploit cross-stage coordination, missing the opportunity to reduce the computation complexity of later stages by leveraging guidance extracted from former stages. Although FlashAttention2 (FA-2) already provides a tiling scheme for softmax to reduce memory access overhead, _**the decreased memory access comes with surging computations**_ . This occurs because repeated exponentiation and comparison operations are necessary to refresh the MAX among tiles, ensuring the correctness of the global MAX value. We observe an opportunity to guide FA-2 computation with top- _k_ information. These limitations highlight the need for more advanced strategies to manage dynamic sparsity with LTPP effectively. 

_**Our Insights:**_ Motivated by the challenges, we observe an opportunity that breaks down the computation, memory, and latency overheads in each stage by adopting a cross-stage coordinated tiling strategy, thus a stage is decomposed into fine-grained sub-stages. Therefore the process in the following stages doesn’t have to wait for the finish of processes in the last stage. The coordination among stages becomes more swift and excessive DRAM memory access could be saved. Notably, it is non-trivial to achieve this goal as we need to figure out effective methods to partition top- _k_ module and efficiently forward the information to formal stages. 

We propose an algorithm-hardware co-design for attention optimizations, named SOFA. It features three key designs that correlate to three challenges, as depicted in Fig. 2. First, the 

1248 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:19:49 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [249 x 108] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input Embedding Layer QKV Generation MHA FFN<br>100%<br>75%<br>Q Proj. K Proj. V Proj. 50%<br>25%<br>Q x K [T] Q x K [T] Q x K [T] 0<br>Softmax Softmax Softmax ViT-B BERT-B (b) GPT2-L Bloom-3B<br>Score x V Score x V Score x V Bloom-3B GPT-2<br>50<br>40<br>Three Linear Layers  30 The improvement in OI enhance the performance ceiling<br>20<br>10<br>Transformer Block  (a) N 0 1 2 Token Parallelism 4 8 16 32 64 128(c)<br>Block 1<br>Operational Intensity<br>**----- End of picture text -----**<br>


Fig. 4. Basic components of a Transformer model and operation intensity. 

computation overhead in _pre-compute_ stage is alleviated via a multiplier-free _differential leading zero summation (DLZS)_ paradigm, which helps reduce the sparsity prediction overhead of each tile. Second, we propose a _sphere-search-aided distributed sorting (SADS)_ , which distributes a long segment into sub-segments to execute individual tiled sorting, while effectively reducing total comparisons. Third, we propose a _sorted-updating FlashAttention (SU-FA)_ . It skillfully decouples the _softmax_ row-dependence to enable the formal computing stage tiling, while leveraging cross-stage sorting information to reduce computation. In summary, DLZS and SADS together serve as a low-complexity prediction (LP) mechanism to reduce prediction overhead. SADS collaborates with SU-FA, employing fine-grained tiling for sparse acceleration, to optimize memory access and processing latency. 

We propose a dedicated accelerator to support the proposed mechanism effectively. Compared to naive implementation, which only has a limited 19 _._ 6 _×_ energy saving over Nvidia A100 GPU, SOFA accelerator improves its performance with four novel algorithm-hardware co-designs. 

Evaluated on 20 benchmarks, SOFA achieves an average energy efficiency of 7183 GOPS/W, which is 71 _._ 5 _×_ and average 15 _._ 8 _×_ higher than Nvidia A100 GPU and eight SOTA accelerators, respectively. Overall, SOFA’s computational efficiency is 9 _._ 5 _×_ higher than that of the GPU A100 and 11 _._ 1 _×_ higher than the TPU, respectively. We also conduct comprehensive ablation on GPU to quantify the performance benefits brought by our software mechanism and various hardware components. Evaluations on GPU/TPU show that SOFA’s software optimization achieves a 3 _._ 16 _× /_ 2 _._ 8 _×_ speedup, while hardware acceleration delivers a 3 _._ 03 _× /_ 3 _._ 9 _×_ speedup. 

## II. BACKGROUND AND MOTIVATION 

## _A. Preliminaries for Transformer_ 

Fig. 4(a) shows a typical Transformer model: an input sequence containing _S_ tokens is transformed into an embedding matrix **X** _∈_ R _[S][×][H]_ , projected to **Q** , **K** and **V** spaces, split into _A_ chunks R _[S][×][H/A]_ , and processed by multi-head attention (MHA) to generate an attention matrix. The attention matrix, after softmax and multiplication with **V** , resulting in a matrix **O** _∈_ R _[S][×]_[(] _[H/A]_[)] . Outputs from all heads are concatenated, projected by **W** _O ∈_ R _[S][×][H]_ , and passed through the FFN with two fully connected layers to generate final outputs. 

**==> picture [239 x 177] intentionally omitted <==**

**----- Start of picture text -----**<br>
FlashAttention2:<br>Divide  Q  into Tr blocks, K , V  into Tc blocks<br>For  i = 1 to Tr do<br>1   Load  Qi � R [Bc×d]  from DRAM to on-chip SRAM<br>2   F or  j = 1 to Tc do<br>3     Load  Kj � Vj � R [Br×d]  from DRAM to SRAM<br>4 Si(j)  =  QiKjT<br>5     mi(j) = max(mi(j-1),rowmax( QiKjT )) (b)<br>6 Pi(j)  = exp( Si(j) -mi(j)) � RBr×Bc<br>7     li(j) =exp(mi(j-1)-mi(j))li(j-1)+rwosum( Pi(j))<br>8     Oi(j)=diag[exp(mi(j-1)-mi(j))]Oi(j-1)+ Pi(j)Vj<br>9 end<br>10 Oi  = diag(liTc)-1 OiTc<br>11   Write  Oi  to DRAM<br>end Reduced memory access comes<br>with increased computations<br>Output O (a)<br>(c)<br>**----- End of picture text -----**<br>


Fig. 5. Process of FlashAttention-2 and its computation overhead. 

**Computation Properties Analysis.** We analyze the operation intensity (OI) [38] for the three parts of a Transformer layer. As shown in Fig. 4(b), MHA exhibits notably lower OI, averaging 15% of the FFN. This means MHA requires more data movement for the same computation FLOPs, due to element-wise operations. Fig. 4(c) further illustrates the relationship between the OI of MHA and the token processing parallelism. We can figure increasing parallelism effectively boosts OI, thus theoretically reducing the demand for data movement under equivalent computational power and PE utilization. This gain is attributed to increased data reuse. 

## _B. FlashAttention (FA)_ 

To reduce data movement of attention, Tri Dao _et. al_ proposed FlashAttention (FA) [39] and the improved version FA-2 [40], both of which successfully minimized memory access but greatly increased computational cost. Fig. 5(a) outlines the procedure of FA-2 and Fig. 5(b) compares its exponential operations and comparison complexity with vanilla implementation regarding _S_ . Here we assume the number of tiles _Tc_ = _S/_ 16, i.e., tiling size _Bc_ =16. We employ the arithmetic complexity model [41] to normalize the complexity for different operations. As _S_ increases, FA-2 exhibits a notable increase in exponential and comparison operations compared to the vanilla scheme. When _S_ =2048, it demands 9 _×_ 10[6] more exponential calculations and 3 _×_ 10[5] more comparisons than the vanilla implementation. Fig. 5(c) compares the increased computational load after summing all calculations. The computational complexity of FA-2 soars with the growth of _S_ , and the increased magnitude correlates with _Tc_ . The larger _Tc_ leads to a faster increase, due to the repeated calculations among _Tc_ blocks, as shown in lines 5-8 of Fig. 5(a). 

## _C. Sparsity in Attention_ 

Typically, as shown in Fig. 4 (a), the results (a.k.a scores) of **Q** _×_ **K** _[T]_ are then processed by a _softmax_ operator. Due to the _softmax_ ’s approximation to the _argmax_ operator, most smaller score values become extremely close to zero after passing the 

1249 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:19:49 UTC from IEEE Xplore.  Restrictions apply. 

_softmax_ . Therefore, they usually impose a negligible impact on the final results and can be reasonably removed. The key difference between the attention sparsity and the sparsity in DNN/Transformer models lies in that attention sparsity is entirely driven by the input data and requires dynamic evaluation at runtime, whereas model sparsity is based on static weight sparsity, which can be optimized through quantization or structured pruning. 

To accelerate _self-attention_ , emerging dynamic sparsity accelerations [23], [28]–[31], [33], [34] offer a promising solution. Their key idea is to predict key Q-K pairs at runtime and calculate attention for selected pairs. Typically, their workflow proceeds as Fig. 6. First, a low-precision computational paradigm is employed to predict the attention ( _Pre-compute stage_ ); Next, vital Q-K pairs are filtered out from each row to generate a mask ( _Top-k sorting stage_ ). Finally, based on the mask, the scheduler initiates the _Formal Computing Stage_ , typically with higher precision. 

## _D. Analysis for Large-scale Token Parallel Processing (LTPP)_ 

Despite its promising adaptability, dynamic sparsity incurs additional overhead during inference, due to the _Pre-compute_ and _Top-k stages_ . As a result, previous works [23], [28], [29], [33], [34] were constrained to processing queries with low parallelism, to minimize the memory and computation overhead. However, as modern LLMs demand significantly longer context than before (GPT4 32k [42], LongLLaMa 256k [43]), the rapid processing of these extended context becomes increasingly crucial [44]. This highlights the necessity for accelerators with LTPP capabilities. However, the current dynamic sparsity attention workflow poses three challenges for LTPP. Illustrated in Fig. 2: 

1) Supposing processing _T_ tokens in parallel, the precompute and sorting complexity rises to _O_ ( _TSH_ ) and _O_ ( _TSSk_ ), respectively. Taking Llama-13B ( _T_ =512, _k_ =0 _._ 25) as an example, the required numbers of comparisons and multiplication would be over 10[11] and 10[8] , respectively. In this case, prediction requires performing over 2[11] MACs and 2[10] comparisons, accounting for more than 57% of the total execution latency. Such prohibitive overhead will negate the improvements brought by sparsity. 

2) As _top-k_ sorting and _softmax_ is applied row-wise, matrices **Pre-Atten** and **A** must be stored to DRAM first and then loaded by row blocks, thus leading to massive DRAM access. Such extensive memory access would lead to inefficient inference. In 45 nm CMOS technology, the energy cost of a DRAM access (5 to 20 pJ/bit) is two orders of magnitude higher than that of internal cache access (0 _._ 1 pJ/bit) [45], while its bandwidth (DDR4 25 _._ 6GB/s) is also orders-of-magnitude lower than the SRAM (19TB/s) [39]. A coarse scheme is to enlarge the on-chip SRAM capacity but this would lead to area inefficiency. Taking ( _T_ =512, _S_ =2048) for instance, it directly necessitates 5MB SRAM, leading to 5 _._ 47 mm[2] footprint under TSMC 28nm technology, which is 7 _._ 4 _×_ , 8 _._ 9 _×_ of the overall area of SpAtten [33] and ELSA [29], respectively. 

TABLE I 

SUMMARY FOR SOTA TRANSFORMER ACCELERATORS. 

|SUMMAR|Y FORSOTA TRANSFORMERACCELERATORS.|Y FORSOTA TRANSFORMERACCELERATORS.|Y FORSOTA TRANSFORMERACCELERATORS.|Y FORSOTA TRANSFORMERACCELERATORS.|Y FORSOTA TRANSFORMERACCELERATORS.|
|---|---|---|---|---|---|
|**Accelerator**|**Optimization**|||||
||Compute||Memory||Cross<br>Stage|
||QKV|Attention|QKV|Attention||
|**A**3 [28]<br>|_×_|�<br>|_×_|_×_|_×_|
|**ELSA** [29]|_×_|�|_×_|_×_|_×_|
|**Sanger** [30]<br>|_×_|�<br>|_×_|_×_|_×_|
|**DOTA** [31]|_×_|�|_×_|_×_|_×_|
|**Energon** [34]<br>|_×_|�<br>|_×_|Low|_×_|
|**DTATrans** [32]|_×_|�|_×_|_×_|_×_|
|**SpAtten** [33]<br>|�<br>|�<br>|_×_|Low|_×_|
|**FACT** [23]|�|�|_×_|_×_|_×_|
|**SOFA**|�|�|�|�|�|



3) FlashAttention2 (FA-2) employs a tiling scheme for the _softmax_ operation to keep the working set of data in the faster on-chip memory, thus successfully reducing off-chip memory access overhead. However, the benefits come with soaring computation costs, making it unsuitable for dynamic sparsity scenarios in LTPP. As an example, when the tile size is _Bc_ = 4 for a sequence length _S_ =1024, FA-2 must frequently compute and compare values across these tiles to ensure correct global results. This leads to a computational load approximately 1 _._ 5 _×_ higher than that of a regular implementation without tiling. 

_**We argue that the main bottleneck in extending existing dynamic sparsity methodology towards LTPP lies in information decoupling among stages, thus missing the cross-stage-tiling opportunity.**_ Table I offers an overview of the effectiveness of existing approaches in optimizing Transformer components. Works [28]–[31] focus on reducing pre-computation overhead, such as ELAS [29] using Binary Hash, A[3] [28] employing Greedy search and DOTA [31] using low-rank transformation. However, these methods still cannot address the row dependency of key operators, like _topk_ and _softmax_ , thus still resulting in significant memory access overhead under LTPP. Further, SpAtten [33] and DTATrans [32] involve sorting the cumulative distribution probabilities of tokens, introducing substantial sorting complexity and latency in LTPP scenarios. While SpAtten [33] and Energon [34] realize challenges with extensive memory access, their sparsity strategies fail to handle the severe memory access overhead with the LTPP scenario. In summary, existing works are all limited on individual-stage optimization, thereby overlooking opportunities for cross-stage joint optimizations, making them inadequate for supporting LTPP. This motivates us to propose a cross-stage compute-memory efficient accelerator design, targeting the LTPP scenario. 

## III. ALGORITHM OPTIMIZATIONS OF SOFA 

Fig. 6 (a) presents an overview of the SOFA algorithm optimizations. First, at the _pre-compute_ stage, we propose DLZS, a log-domain computing paradigm to predict **A[ˆ]** . Then, exploiting _DCE_ , we introduce SADS, to partition a long sequence into several sub-segments for independent tiled sorting. Next, leveraging the sorting information, a memory-compute efficient attention-computing mechanism (SU-FA) is designed. The SADS and SU-FA enable SOFA to execute a cross-stage 

1250 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:19:49 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [251 x 182] intentionally omitted <==**

**----- Start of picture text -----**<br>
S S(1-k) Sparsity<br>(a) K [T] D D<br>D Sub-segment Sort D D<br>Bc<br>Q Q<br>(Pre Atten Matrix) (Pre Atten Matrix) (Atten Matrix)<br>Bc Bc Formal  Bc<br>Pre-compute stageDLZS Top-k stageSADS computing stage Output (O) D<br>� Sec.III A � � Sec.III B � SU-FlashAttention T � Sec.III C �<br>(b) Standard workflow  Memory Access Computation<br>Pre- 4bit Q x K [T]<br>compute Store Pre-Atten<br>Top-k  Load Pre-Atten SOFA frees DRAM load and store<br>stage Top-k Sort<br>16 bit Q x KT<br>Formal Store Atten<br>Computestage Load AttenScore x VSoftmax SOFA fuses (QxK(Score x V) into one stage [T] ) + softmax +<br>SOFA workflow  Reduced pipeline filling time<br>Pre-compute Stage 1.  Cross-stage fusion  enables fine-grained tiled pipeline<br>Top-k Sort Stage 2.  Tiled  execution avoids off-chip access, with smaller SRAM<br>Formal Comp Stage Redcued latency<br>T T S(1-k)<br>**----- End of picture text -----**<br>


Fig. 6. (a) High-level diagram of the SOFA algorithm optimizations. (b) Tilebased pipelined dataflow (SOFA) vs. standard dataflow. 

**==> picture [243 x 45] intentionally omitted <==**

Its workflow is depicted in Fig. 7(b). As the weights are pre-known and fixed during inference, we pre-convert the **W** _k_ into LZ format and store it. Then, in the _Key prediction phase_ (1.1), no LZE is required, as the weight **W** _k_ has been converted into LZ format. In the subsequent _Attention prediction phase_ (1.2), to mitigate error accumulation, we convert **Q** into the log domain instead of **K**[ˆ] , then perform shifting and sum operations. Compared to the vanilla leading zero strategy (Fig. 7(c)), the proposed DLZS exhibits three Pros: a) Lower converter overhead; b) Higher accuracy; c) Less memory access. 

## _B. Sphere-search Aided Distributed Sorting (SADS)_ 

tiling pipeline dataflow. Compared to the vanilla workflow in Fig. 6(b), the tiling execution makes SOFA require minimal SRAM for storing intermediate results without extra memory access, while the fine-grained pipelined dataflow can reduce inference latency. 

## _A. Cross-Phase DLZS Sparsity Prediction_ 

Traditional _dynamic sparsity_ entails predicting significant Q-K pairs, then utilizing these important Ks and Vs to execute computations. However, blindly generating unnecessary KV leads to wastage in computation and memory access. To this end, SOFA employs an _on-demand_ computation strategy for KV. As shown in Fig. 7(a), _On-demand_ means: only the required Ks and Vs are generated ( **K** _i_ = **x** _i_ **W** _k_ , **V** = **x** _i_ **W** _v_ ), while trivial ones are not computed from the beginning. However, this requires the _pre-compute stage_ first to estimate the **K**[ˆ] , then utilize it with **Q** to predict **A**[ˆ] . Unfortunately, even utilizing low-precision matrix multiplication (e.g. halfprecision with MSBs only) results in considerable power consumption. Therefore, a power and memory-efficient prediction is imperative. 

We propose a log-domain multiplication-free strategy, named differential leading zero summation (DLZS). _Differential_ means: For multiplication, it only transforms one operand into the logarithmic domain using the leading zero encoder (LZE), to obtain its leading zero (LZ). Then, based on the LZ, it substitutes the costly multiplication with low-power shift operations on the other operand. Specifically, an INTtype number _x_ can be mathematically expressed as Eq. (1a), where _W_ stands for the bit-width, _M_ represents the mantissa lying [0 _,_ 1], and _LZ_ denotes the leading-zero count of _x_ . Accordingly, the corresponding multiplication is derived as Eq. (1b) and approximated as Eq. (1c). Since the bit width _W_ is fixed for certain operands, we can directly operate _LZy_ on _x_ , to estimate the magnitude for the product of two numbers. Therefore, incorporating shifting and the sign bit, the results of multiplication can be predicted. 

To effectively identify vital tokens in the _top-k stage_ , previous works [33], [34] have tried to explore low-cost sorting algorithms and design corresponding hardware to improve throughput. However, they all fail to consider the data distribution property, thus only achieving limited efficiency and missing opportunities for cross-stage optimization. 

As _softmax_ approximates the _argmax_ operation, its results primarily depend on dominant tokens when multiple tokens with prominent amplitudes appear, as denoted in Type-I of Fig. 8(a). Alternatively, there are two potential scenarios for element distribution: a uniform distribution, exemplified by Type-II, and a concentration of slightly larger elements in a specific region, depicted as Type-III. To ascertain their practical distributions in Transformer inference, we conducted a token analysis for BERT/L [3], ViT/B [12], GPT-2 [7] and Llama7B [46] with 4096 rows. The statistical results in Fig. 8(b) reveal that the Type-II distribution predominates across all four models, accounting for over 76% on average. Type-I occurrence is more frequent in ViT, GPT-2 and Llama, with an average rate of 25%, which may be attributed to image local similarity and the self-autoregressive token generation, respectively. By contrast, the occurrence probability of TypeIII is notably low in all models, even approaching nearly 0 in GPT-2 and Llama7B. This is primarily attributed to the extended context, which diminishes the likelihood of a concentration for higher magnitude tokens in a specific region. 

Combined, Type-I and Type-II collectively constitute over 95% of the total distribution. Hence, these two types can effectively represent the overall data distribution characteristics of the attention. We observe that the larger values within each region of these two data types can aptly represent the overall larger values. We term this characteristic as the ‘ _Distributed Cluster Effect (DCE)_ ’. _Distributed_ implies that a long segment can be divided into several shorter sub-segments, while _Cluster_ indicates that each sub-segment contains its primary information. Therefore, sorting based on well-segmented partitions is expected to have a negligible impact on holistic performance. 

1251 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:19:49 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [510 x 116] intentionally omitted <==**

**----- Start of picture text -----**<br>
Traditional Sparsity Prediction (a) Proposed Cross Phase DLZS-based Sparsity Prediction (b) Standard INT Multiplications (c)<br>00011000 (24) 00000110 (6) 10010000 (144)<br>Query �  QKV Linear ProjectionKey �� Value �� On-demand KV Linear Projection K-Gen Attention Prediction xxxx002010s0 x x x x Token 1.1 DLZS Key prediction 012111s1 ����  x x x x1n0n2nsn Shift sum wwww1000n020 w w w w W 110121n1 K ���� wwww1d0d2dnd wwwww DRAM Storage 0001020304 0000010001001010111000011000010000010000 Cons1: Convert two multipliers to one hot � 0001 Vanilla Leading Zero Scheme DRAM0000 Store 0000010000000100 Converter01000000 (+64)Cons2: Evident errorTokens80 +144<br>memory access (8bit)<br>�� (Low bit-width)  Att. Calculation Att. Prediction (High bit-width)ComputeOmitWeak connectionStrong connection QQKK Sparse KQuery �  Att. CalculationQ Spar. �� Key (High bit-width) Sparse VSpar. VOutput KK 1.2 DLZS Attention prediction K1000s0 K K K1101s1 ���   K  K  K1d0d A sdA A00 10 s0 AAAShift sum0111s1 ��� AA A QQQ0dsd 1d 0010s0 QQQ Top-K select 0111s1 Q ��� QQQsd0d1d wwwww Pre- wnd00010203nd Sign bit p 1 rocessin 111100000001010100110101011 g � Pros1: Half overhead for converters00011000 Differential Leading Zero Summation (DLZS) SRAMDRAMSRAM Less memory access (4bit) Store 2(0010)ConverterConverter00000100 Converter01100000 (+96)<<<< Outp.Outp.Pros2: Half errorTokens 480110000001000000+144<br>� � � � � � � �<br>� �<br>� � � � � � � �<br>� � � � � �<br>**----- End of picture text -----**<br>


Fig. 7. (a) Traditional sparsity prediction. (b) Cross-phase DLZS sparsity prediction. (c) Comparisons between DLZS and vanilla scheme. 

**==> picture [249 x 153] intentionally omitted <==**

**----- Start of picture text -----**<br>
Type I: dominated by a few tokens BERT- ViT- GPT2- Llama7B<br>0.75 #layer Cola ImageNet WikiText2 Winograde<br>0.50<br>0.25 Region 1 Region 2 1<br>Type II: dominated by several tokens,<br>which are evenly distributed 2<br>0.75<br>0.50<br>Region 1 Region 2<br>0.25<br>3<br>Type III: dominated by several tokens,<br>which are concentrated in one region<br>0.75<br>0.50 Region 1 Region 2 4<br>0.25<br>(a) (b) Type I Type II Type III<br>After softmax<br>After softmax<br>After softmax<br>**----- End of picture text -----**<br>


Fig. 8. (a) Three types of attention data distribution. (b) Corresponding proportions in diverse Transformer models. 

**==> picture [249 x 169] intentionally omitted <==**

**----- Start of picture text -----**<br>
Sphere-aided Distributed Sorting (a)<br>A Row of Attention (S)<br>S/4 S/4 S/4 S/4<br>Exclusive entries Exclusive entries Exclusive entries Exclusive emtries<br>r r r r<br>Search radius Each sub-segment searches corresponding top (k/n) in radius<br>FC FC FC FC<br>SubFC SubFC SubFC SubFC<br>Scenario 1: There is a Type-I occurrence   Scenario 2: Predominantly Type-II<br>Region 1 Region 2 Region 3 Region 4 Region 1 Region 2 Region 3 Region 4<br>Can filter out the dominant values Values at the edges are small and close<br>(c)<br>(b)<br>We can reasonably ignore their orders<br>**----- End of picture text -----**<br>


Fig. 9. (a) Low-complexity SADS sorting algorithm. (b) Scenario 1: Type-I occurs. (c) Scenario 2: Type-II dominates. 

To this end, we propose the SADS sorting, which exploits the _DCE_ to reduce complexity in a tiled manner. Initially, as shown in Fig. 9 (a), one row of the attention matrix is divided into _n_ sub-segments (assuming _n_ =4). Next, each sub-segment pick up the top-( _k/n_ ), i.e., top-( _k/_ 4) values, from its own data. Due to the area constraints of hardware implementation, sorting may necessitate multiple iterations. In each iteration, the Max value from the previous iteration 

serves as a benchmark. A feasible range ( _FR_ ) is obtained by subtracting the spherical radius ( _R_ ) from the benchmark. Then, the sorting is exclusively performed on the entries within the _FR_ , rather than all entries. Following this, for each sorted set, the largest _k/_ 4 elements are collected into **FC** set, which represents the indices of vital KVs. This set is used to guide the subsequent _Formal Computing Stage_ . 

Figs. 9 (b)-(c) exemplify why SADS can maintain accuracy with reduced complexity. For Scenario 1, where Type-I distribution occurs, SADS is certain to capture the dominant values, irrespective of which sub-segment they fall into. For Scenario 2, where the majority of the distribution is Type-II, SADS can effectively select all relatively larger values that dominate in the complete row. Given that the values falling on the edges of the top- _k_ are typically smaller, we can reasonably relax the sorting requirements for them. According to our experimental results, with _s_ =1024 and _n_ =4, the average error of the top 25% is only around 3% (Different K indices are considered as errors). Furthermore, the specific number of sub-segments (e.g. tiling size) of each layer is obtained by the DSE in Section III-D. 

## _C. Sorted-Updating FlashAttention (SU-FA)_ 

The attention is the primary bottleneck in scaling to LTPP scenario, as its memory complexity increases quadratically with the sequence length. To tackle this issue, we propose an attention acceleration mechanism called SU-FA, which is both computationally and memory efficient, by leveraging specific sorting information generated from the _top-k stage_ . It also enables cross-stage tiling for the formal-compute stage. Traditionally, addressing overflow in hardware _softmax_ implementation requires identifying the Max value in each row [47]. This leads to continual comparison operations in classical FA [39], [40] to refresh the Max value across diverse blocks, which however, results in skyrocketing computational cost as revealed in Fig. 5. 

The indices of the top- _k_ values provided by _top-k stage_ allow us to get the potential index of the Max value. A direct but coarse approach is to calculate the Max value based on the potential index and then send it into the FA for computation. However, there are two critical problems: 1) The index of the Max value is not guaranteed to be accurate due to the approximation properties of DLSZ, which could result in 

1252 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:19:49 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [249 x 173] intentionally omitted <==**

**----- Start of picture text -----**<br>
(1) (2)<br>Additional multiplication (a)<br>Proposed compute-memory efficient SU-FA<br>Initial:  Divide  K , V  into Tc blocks, each with Bc × H/A<br>Parallel for i  = 1 to T do<br>1       Load  Q i �R [1×H/A]  from DRAM to on-chip SRAM<br>for  j = 1 to Tc do<br>//Scheduler ensure sij[1] is the Max in Block<br>2<br>3<br>4<br>5<br>6<br>(b)<br>7 //Different Tiles synchronization<br>**----- End of picture text -----**<br>


Fig. 10. (a) Formulas for diverse updating orders. (b) Procedure of SU-FA. 

potential overflow; 2) Separately calculating the Max value introduces additional computational and power overhead. To this end, we propose a novel sorted-updating FA. Instead of computing for the Max separately, SU-FA executes either ascend or descend updates during the computation process. Descend updating means first computing Fig. 5(a) line 5 from the index of Max, followed by the index of the 2nd large value, until the _k_ -th value. Ascend updating proceeds in the opposite order. Although at first glance, both of these approaches can effectively eliminate the max comparison (Fig. 5 (a) line5), we found that the benefits vary significantly with different updating orders. Specifically, when executing ascend updating, the line 5-7 can be rewritten as Eq. (1) in Fig. 10(a), where we denote **S** _[j] i_[as] _[x]_[(] _i[j]_[)] for clarity. Though _m_[(] _i[j]_[)] equals to _x_[(] _i[j]_[)] constantly, it is noteworthy the calculating for _li_[(] _[j]_[)] still acquires one exponentiation (Exp), one multiplication (Mul) plus an addition (Add). 

By contrast, if descending order is employed, as Eq. (2) in Fig. 10(a), the updating for _li_[(] _[j]_[)] merely requires one Exp and one Add. While such benefits may seem minor, the performance gain is substantial when large-scale parallel process long sequences. The procedure for the descending SU-FA is summarized in Fig. 10(b). Compared to the traditional FA and ascending SU-FA, the descending SU-FA on average reduces 25% and 11% complexity, respectively. In subsequent discussions, SU-FA defaults to adopt the descending order. Please note the inaccuracy of the predicted Max is co-optimized by the architecture in Section IV-D. 

## _D. Design Space Exploration (DSE)_ 

In the SOFA algorithm mechanism, the tiling size, i.e., _Bc_ in each layer and top- _k_ form an interesting design space. For larger _Bc_ , i.e. smaller _Tc_ ( _S_ = _Bc × Tc_ ), inference accuracy tends to increase. However, sorting complexity escalates significantly, yet the computation complexity of SU-FA decreases. On the contrary, when _Bc_ decreases, it will lead to opposite effects. We provide each of the hyperparameters with plenty of options as 1) _Tci_ : 2 _−_ 32, step=2; 2) Top- _k_ : 

**Algorithm 1:** DSE for SOFA Tiling Size. 

**==> picture [232 x 94] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 Input: Evaluation function L and exploration space Θ;<br>2 Initial: Max Iter T , sample Dt =  {Ri, L ( Ri ) , i  = 1 , ..., n} ,<br>Best target function result J =  ∞ ;<br>3 while t < T and result does not converge do<br>4 Rt ← arg maxΘ  α (Θ , Dt ), Jnew ←L ( Rt );<br>5 Dt +1 ←{Dt,  ( Rt, Jnew ) } ;<br>6 GP new ← Update( GP, Dt +1);<br>7 if Jnew < J then<br>8 J ← Update( Jnew );<br>**----- End of picture text -----**<br>


**==> picture [249 x 138] intentionally omitted <==**

**----- Start of picture text -----**<br>
Token SRAM (192KB) Weight SRAM (96KB) Temp SRAM (28KB)<br>Cross Stage DLZS Prediction PE Array (128x4)<br><< << 3 6 PE line 0<br>128x32 Shift- PE line 1<br>Adder Array<br><< << PE line 127<br>2 7<br>Q, X ,Wk,Wv Fetcher SU-FlashAttention (128x2x2)<br>Q-K [T]  PE line 0 S-V PE line 0<br>1 4 5 (MASK)<br>Q-K [T]  PE line 1 S-V PE line 1<br>Tiled & Out-of-Order<br>Computation Ctlr 8 Q-K [T]  PE line 127 S-V PE line 127<br>Zero Elimin. Config. LZE Iter. SADS Unit Row/Col Router<br>External DRAM<br>Reuse Awa.  Scheduler MAX Ensure Tiles Synch.<br>**----- End of picture text -----**<br>


Fig. 11. High-level block diagram for the SOFA accelerator. 

5% _−_ 50%, step=5%, to ensure that we can obtain a highquality solution. However, such space is huge and unaffordable for brute force search. Taking BERT-Base with 12 Transformer layers as an example, we need to search for the optimal choice in a 26-dimensional space consisting over 10[15] choices. Even though the inference on highly parallel GPU clusters costs less than 1 ms, it will take unbearable time consumption (over 10[8] h) using traversal-based grid search for this remarkable design space. To this end, we apply a Bayesian optimization method to execute the search process. The targeted optimization problem (modeled as a Gaussian Process (GP) in Bayesian optimization) is constructed concerning both the accuracy and the computational complexity, which is formalized as Eq. (2). 

**==> picture [236 x 31] intentionally omitted <==**

**==> picture [185 x 17] intentionally omitted <==**

where _R_ is the hyperparameter vector composed of the _k_ and _Bci_ of each layer, _Len_ is the cross-entropy loss, _Lcmp_ and _Lexp_ are the penalty terms for computation overhead, as defined in Eqs. (3) and (4). _α_ and _β_ are two coefficients to balance the accuracy and performance. The whole searching process is summarized in Alg. 1. 

## IV. ARCHITECTURE AND HARDWARE INNOVATION 

Despite substantial algorithmic acceleration, a naive implementation of SOFA faces three challenges. First, LP is crucial in predicting vital tokens. It must ensure high precision 

1253 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:19:49 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [249 x 83] intentionally omitted <==**

**----- Start of picture text -----**<br>
Configurable LZ Encoder Reusable DLZS Engine Design<br>D(8/16b) LZC 8-bit #0 LZC 8-bit #1 LZC 128×32 Shift Array<br>D(8/16b) a0 z0 a1 z1 Array  Col Scheduler<br>D(8/16b) 5b<br>D(8/16b) 4b << << << <<<br>D(8/16b) 0 1 8b <<<< <<<< <<<< <<<<<br>8 bit output 16 bit output Sign bit << << << <<<br>16b Systolic dataflow<br>K Estim. Data Path QxK [T]  Data Path<br>y = abs (x) Out Buff(4KB)<br>Zero-eliminator Row Scheduler One-Hot Accu.<br>**----- End of picture text -----**<br>


Fig. 12. Architecture for the cross-stage DLZS prediction. 

and low power consumption. Additionally, the top- _k_ engine must support variable-length inputs and high throughput within low power overhead, due to the flexible tiling execution and high parallelism of LTPP. Second, specific architecture and datapath designs are needed to support the intra-stage operatorfusion paradigm of SU-FA for enhanced efficiency. Finally, during LTPP execution, the varying requirements of K and V for each query may lead to redundant memory access, thus necessitating a memory-efficient scheduling strategy. 

## _A. Architecture Overview_ 

Fig. 11 depicts SOFA’s overall architecture, which comprises six main modules: on-chip SRAM storage, a DLZS prediction unit, an iterative SADS unit, a PE array, an SU-FA unit, and a tiled & out-of-order controller. SOFA is designed to process 128 queries in parallel. First, the indices of tokens and corresponding **W** _k_ of a tile produced by the _controller_ are sent to _data fetcher_ , which calculates the physical address and fetches data to on-chip SRAM . Then, the _DLZS predictor_ starts to estimate matrices **K**[ˆ] and **A**[ˆ] with log-based shift and summations . Next, the 128-row **A**[ˆ] is sent to _SADS unit_ , to find out the top- _k_ important Q-K pairs . Subsequently, the sorting results are sent back to the _controller_ , which generates a top- _k_ mask, then _data fetcher_ reads corresponding data according to the mask . After that, the scheduler controls the _PE array_ to generate the necessary Ks and Vs . Later, the generated KVs are sent to the _SU-FA unit_ to execute computememory efficient attention calculations . Finally, the outputs of attention are stored to off-chip DRAM . 

## _B. Reusable and Configurable DLZS Engine_ 

As discussed in Section III-A, the DLZS unit is acquired to predict the **K[ˆ]** and **A[ˆ]** , respectively. The two phases demand diverse precisions. In the former case, the inputs are 8-bit token and weights, where the weights are pre-converted into LZ format. In contrast, the latter case requires operations with 16-bit precision, as the output of the former is truncated to at most 16 bit. To this end, the LZE is designed as configurable to enable 8 & 16-bit mixed precisions. As depicted in Fig. 12 left, each LZE unit contains two 8-bit leading zero counters (LZCs) [48] connected in series. When the input is 8-bit, the two LZCs work independently. However, when the input becomes 16-bit, the two _all-zero flag a_ 0 and _a_ 1 are performed through logic AND, then the corresponding output is employed as a selected signal to pick up 16-bit outputs. The processing flow of DLZS engine is illustrated in Fig. 12 right. First, 

**==> picture [249 x 116] intentionally omitted <==**

**----- Start of picture text -----**<br>
Search radius r<br>68 r Attention(Q×K [T] ) Clipping< Threshold Updating UnitMIN Value MAX Value<br>4 Threshold 1<br>2<br>0 0<br>Blocked in subsequent iteration 0 1<br><<br>Negligible influence on final<br>results post-softmax. 0<br> After Softmax 0 1<br>1.00 <<br>0.75<br>0.50 MAX region tokensDominated by near<br>0.250 16-4 sorting<br>New data<br>Output buffer<br>Index reordering Parallel BM-2 Parallel BM-4 Index rescaling<br>0<br>**----- End of picture text -----**<br>


Fig. 13. Architecture for the flexible-input supported SADS engine. 

the operands are sent to a zero eliminator module, where calculations with zeros are removed. Next, in the **K[ˆ]** prediction phase, 8-bit tokens and 4-bit LZ-format weights are transferred to the 128 _×_ 32 systolic shift array, and **K[ˆ]** would be generated and cached in the output buffer. Then, in the **A** prediction phase, the 16-bit Qs are fed to the 16-bit mode LZC array. The generated 5-bit LZs along with the **K[ˆ]** are sent to the shift array again, to produce the final estimated **A**[ˆ] . 

## _C. High-parallel and Flexible-input Supported SADS Engine_ 

As illustrated in Fig. 6, the tile sizes in SOFA’s tiled pipeline mechanism vary across different models and tasks. In other words, the length of the sub-segment that the sorting unit needs to process is flexible. This demands a sorting module that supports flexible inputs with low power overhead and high throughput to avoid bottlenecks. To this end, we design a flexible-input sorting architecture, with the high-parallel bitonic sorting core. Fig. 13 illustrates the SADS engine, which consists of two main modules: 

_1) Sorting Module:_ The core sorting architecture uses a fully parallel 16-to-4 bitonic sort design [49]. To handle flexible-length inputs, the module receives 12 new inputs each time. combines them with the four largest values from the previous round, and outputs four new sorted values. After all elements are processed, the final results are generated. Throughout this process, we observe an opportunity to further reduce power consumption. Essentially, we only need the top- _k_ values and the top-1 and top-2 value (the top-1, top-2 values are utilized to accelerate the SU-FA), and the order among the 3rd to the _k_ -th Max value is inconsequential. Therefore, we can eliminate redundant comparators without compromising the outcome, as shown in the shaded area in Fig. 13. 

_2) Clipping Module:_ According to the proposed SADS in Section III-B, only elements in the feasible range are picked up and sorted accordingly. To this end, an adaptive clipping mechanism is implemented in this module to perform the filter function. As illustrated in Fig. 13, it first reads the data to be sorted from DLZS unit and the threshold from Threshold Updating (TU), respectively. The threshold is selected as the larger value between the _top margin_ (=Max- _r_ ) and the _low bound_ (The current Min value in the output buffer). In the beginning, both the _low bound_ and _top margin_ are set as zero and no values are eliminated. After obtaining the 

1254 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:19:49 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [249 x 106] intentionally omitted <==**

**----- Start of picture text -----**<br>
Selected K  Reuse-Aware  Selected V<br>buffer Scheduler buffer<br>Line 0Line 1 QQ 01 PEPE0010 PEPE0111 ss 01 1010 << ExpExp Partial sumPartial sum PEPE 0212 PEPE 0313 ll 10 OO 01<br>Line 2 Q 2 PE20 PE21 MAX Ensuring Circuit (Floding) s 2 10 < Exp Partial sum PE22 PE23 l 2 O 2<br>Line buffer Line  T Q T Output Stationary  PE T 0 PE T 1 s T Auxiliary Process 01 < Exp Partial sum Output Stationary  PE T 2 PE T 3 o  Updating l T O T<br>for Q Systolic Array 1 module Systolic Array 2 module<br>**----- End of picture text -----**<br>


Fig. 14. The dedicated data flow architecture for the SU-FA mechanism. 

temporal sorted results, the _low bound_ and _top margin_ are updated in TU module. After that, the clipping mechanism is active and the smaller values are blocked in the following iterations. Given the efficiency of hardware implementation, we opt to directly substitute the blocked values with zeros. This approach effectively reduces power consumption from switching activities while maintaining hardware compatibility. 

## _D. Successive Updating FlashAttention Engine_ 

While SU-FA can effectively reduce non-linear computations of traditional FA by leveraging the Max value provided by the _top-k stage_ , it still faces a critical precision issue. This is because DLZS inherently is log-domain approximate computing, thus inevitably leading to estimation errors. Hence, hardware support is required to provide runtime assurance for the Max value. However, introducing a dedicated module for dynamic comparison directly would incur huge area overheads. To achieve this, we design a folded auxiliary process (AP) module capable of simultaneously supporting both Max value assurance functionality and synchronization between tiles (line 5-6 in Fig. 10). As depicted in Fig. 14, this module operates in two configuration modes: _computation_ (0) and _max update_ (1). In mode 0, the intermediate value _s_ from the systolic array (SA) 1 is directly subtracted with the Max value cached in Reg, and then fed to the Exp unit. Otherwise, in mode 1, the _s_ is sent into a comparator, compared with the Max cached in Reg, and the Reg’s Max value is updated accordingly. Please note Mode 1 is only activated during switching between different tiles or in the first computation phase within the same tile. The tiled computation controller manages the switching between the two modes. 

**Workflow.** The SU-FA engine consists of four main parts: two SAs, an AP module, and an **O** updating module. First, the 128-row Q vectors are stored in the line buffer. Subsequently, two rows of K vectors corresponding to each Q vector are incrementally fed into SA-1, generating the corresponding **s** . Then, **s** is sent into the AP module to perform the corresponding comparison or Exp calculation (Fig. 10 line3, 5, 6), yielding intermediate partial sum results. The partial sum results are then fed into SA-2 and multiplied with the corresponding V vectors. Finally, the resulting output is sent to the **O** updating module to compute the final outputs (Fig. 10 line 7). 

**Reuse-Aware Schedule Scheme (RASS).** Due to dynamic sparsity, different queries select different Ks and Vs, with some 

**==> picture [249 x 143] intentionally omitted <==**

**----- Start of picture text -----**<br>
Naive Execution KV Out-of-order Execution Reuse-Aware Schedule Execution<br>KV5V05, VK6V16  V are only needed by Q3 2 V3 V4 V5 V6 V7 RASS Scheduler V0 V1 V2 V3 V4 V5 V6 V7<br>K0 K1 K2 K3 K4 K5 K6 K7 1000 5,6 K0 K1 K2 K3 K4 K5 K6 K7<br>Q0 0100 NA ControlFSM Q0<br>0010 0,1<br>Q1 Q1<br>1110 NA<br>Q2 1101 4,7 0,1,4,7 Q2<br>Q3 10110111 NA2,3 2,3,5,6 Q3<br>Eg: Buffer-1000 stores IDs that are  1111 NA Executed at Phase 0<br>required by Q3  Executed at Phase 1<br>Naive Execution Reuse-Aware Schedule Execution<br>Phase 0 Load{K0V0,K1V1,K2V2,K3V3,K4V4,K5V5} Load{K2V2,K3V3,K5V5,K6V6}<br>Phase 1 Load{K2K2,K3V3,K4V4,K5V5,K6V6,K7V7} Load{K0V0,K1V1,K4V4,K7V7}<br>AccessMem  24 vectors (12Ks, 12Vs) 33% reduction 16 vectors (8 Ks, 8Vs)<br>Issuing FIFO<br>Bitmask Coding of needing Qs<br>���<br>**----- End of picture text -----**<br>


Fig. 15. Comparisons between RASS strategy and vanilla execution. 

overlap. Hence, how to effectively reuse K and V between different queries is a crucial challenge, especially in large-scale parallel processing. Based on [31], we design a _reuse-aware schedule scheme (RASS)_ with KV out-of-order execution to reduce overall memory access. As shown in Fig 15, _k_ 2 and _k_ 3 are shared among three queries: _q_ 0, _q_ 1 and _q_ 2, making them the top candidates for initial scheduling. Then, RASS seeks out Ks which are exclusively used by the remaining unscheduled query _q_ 3, i.e., _k_ 5 and _k_ 6. As a result, _k_ 2, _k_ 3, _k_ 5, and _k_ 6 are packed together for execution in Phase 0. Such greedy search continues until all queries are allocated adequate Ks. As exemplified in Fig 15, compared to the default left-toright computation order, RASS reduces 33% memory access. 

We design a scheduler to implement the RASS. As shown in the middle of Fig. 15, the whole condition statement and control logic are implemented in an FSM controller. Besides, it involves a single-port read-write ID Buffer which is indexed using a bitmask of queries. For example, _k_ 5 _v_ 5 and _k_ 6 _v_ 6 are exclusively required by query _q_ 3. Consequently, the pair ‘5, 6’ is stored in buffer-1000. Then the FSM controller accesses the ID Buffer according to the RASS, and dispatches the IDs into the Issuing FIFO in an optimized execution order. 

## V. EVALUATION 

## _A. Experimental Setup_ 

We evaluate the soft performance of SOFA with several typical Transformer models and tasks by NVIDIA A100 GPU. For NLP tasks, the BERT-base and BERTlarge models [3], are selected and evaluated by eight tasks from GLUE [50] and SQuAD v1.1 [51]. The maximum sequence length for BERT-B/L is 256/256/384/512/512 for MRPC/RTE/SQUAD/STSB/QNLI, respectively. Moreover, for GPT-2 [7], Bloom-1.7B [52], Llama7B/13B [46], language modeling tasks on Wikitext-2 [53], WikiLingua [54], Wiki-raw and Winogrande [55] are evaluated. The maximum length for datasets on evaluated Bloom1.3B/Llama7B/13B is 2k/4k/4k, respectively. For CV tasks, we choose the latest PVT (with 3192 sequence length) [56] for ImageNet-1k classification [57] by fine-tuning the checkpoint of ImageNet-21k. All models are implemented with Pytorch libraries [58] and Huggingface 

1255 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:19:49 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [189 x 86] intentionally omitted <==**

**----- Start of picture text -----**<br>
Pre-deployment  Model : Bloom, Llama7B, ...<br>Preparation (offline) Dataset : Wiki, Dolloy, ...<br>Configuration List Top-k : 15%, 10%, ...<br>Preprocess : Convert Wk in<br>LZ format and store<br>Model Preparation DSE : tiling size (Bc)<br>Adjust : Top-k finetune<br>User Choose Model load SOFA Sparsity Inference<br>User Inference (online)<br>**----- End of picture text -----**<br>


**==> picture [251 x 73] intentionally omitted <==**

**----- Start of picture text -----**<br>
4bit+vanilla sorting+FA2 DLZS+vanilla sorting+FA2 DLZS+SADS+FA2 DLZS+SADS+SU-FA<br>100%<br>95%<br>90% 18%<br>85% 25%<br>80% 28%<br>75%<br>Normalized Complexity 70%<br>**----- End of picture text -----**<br>


Fig. 17. Complexity reduction for the proposed DLZS, SADS and SU-FA. 

Fig. 16. The preparation and execution flow diagram of SOFA. 

Transformer project [59]. For each task, we execute finetuning on NVIDIA A100 GPU after token pruning to recover accuracy. 

For hardware evaluation, we performed the RTL design for the SOFA accelerator and utilized Synopsys DC on TSMC 28nm CMOS technology, to estimate the logic parts’area and power consumption. The power, area, and read/write bandwidth of on-chip SRAM buffers are estimated through CACTI [60]. For modeling off-chip DRAM, we utilize Ramulator [61] to simulate the memory behaviors and employ the same method with [62]–[64] to estimate the IO power. According to the synthesized results, the latency of the critical path is less than 1 ns. Then, we assume the running frequency of SOFA is 1 GHz. We extract each stage’s actual cycles by simulating the RTL with Verilator [65], based on which a cycle-level simulator is implemented to evaluate end-to-end performance. 

For comparisons with GPU, we deploy the benchmarks on the A100 platform using the Pytorch framework. We measure execution time by inserting torch.cuda.synchronize at the start and end points, and then calculate the elapsed time. For power measurement, based on nvidia-smi, we first measure the system’s idle power, and then repeatedly run workloads and get the total power. The dynamic power is total power minus idle power. Based on the computational workload, we derive the average throughput and energy efficiency. Similarly, we run the cloud TPU [66], [67] to analyze the performance on diverse commercial hardware. 

## _B. Algorithm Performance_ 

Fig. 16 illustrates the SOFA flow-diagram, which consists of two phases: _Pre-deployment Preparation (PP)_ and _User Inference (UI)_ . During the PP phase, the server selects models and corresponding datasets, then preprocesses each model through DSE (Section III-D) and fine-tuning. The processed models are then stored for user selection. In the UI phase, users simply select their desired model, which, once loaded, enables real-time dynamic sparsity inference using SOFA. 

_1) Algorithm Settings:_ In DSE objective function (2), the coefficient _α_ adjusts the proportion of the increased sorting cost, while _β_ controls the proportion of the benefit from reduced exponential operations. Initially, we conducted numerous experiments on BERT/PVT/GPT-2/Bloom/Llama to determine the search range for each hyperparameter. Subsequently, during training, we employed grid search to find the optimal parameter for each model and applied the suc- 

cessive halving method to accelerate the process. According to our experiments, the _α_ / _β_ is set as 0.24/0.31 (BERT-B/L), 0.2/0.24 (ViT), 0.4/0.42 (GPT-2), 0.53/0.56 (Bloom-1.7B), and 0.58/0.63 (Llama-7B/13B), respectively. We then search for 200 iterations with each learning rate (1e-1, 5e-2, 1e-3) to obtain the optimal tiling setting. 

_2) Overall Performance:_ We first set an ablation experiment to evaluate the low-complexity advantages of DLZS, SADS and SU-FA by comparing them with a baseline scheme. The baseline is assumed to utilize 4-bit multiplications in _pre-compute stage_ , vanilla sorting in _top-k stage_ and traditional FA in _formal-compute stage_ . The complexity for different operations is normalized by the arithmetic complexity model [41]. For fairness, each model’s loss remains under 2%. As shown in Fig.17, DLZS reduces complexity by 18% on average compared to the baseline. The reduction mainly comes from its multiplier-free computing and half-conversion feature. Further, SADS and SU-FA contribute to an extra 10% reduction through segmented sorting and simplifying redundant non-linear computations using top- _k_ information. Overall, compared to traditional mechanisms, SOFA’s software strategy achieves 28% lower computation complexity under the same token sparsity, making SOFA accelerator effective for handling the LTPP scenario. 

To demonstrate the effectiveness of SOFA in detecting token sparsity, Fig. 18 shows the QKV and attention computation reduction, introduced by the SOFA’s sparsity prediction (LP). For practicality, we statistically analyzed the reduction in computational workload while ensuring accuracy losses remained below 0%, 1%, and 2% respectively. Different endto-end metrics are utilized for evaluation, such as F1 score for SQuAD and accuracy for RTE. On average, SOFA’s sparsity prediction can reduce the attention+QKV computation by 56 _._ 8%/62 _._ 6%/67 _._ 4% with 0%/1%/2% accuracy loss, respectively. Focusing solely on the attention part, SOFA reduces computation by 81 _._ 3%/87 _._ 7%/92 _._ 6%. 

**Discussion on accuracy:** In _top-k_ pruning, there is a hyperparameter _k_ . Lowering _k_ eliminates more QK-pairs, which in turn reduces computation. However, reducing _k_ too aggressively could lead to the exclusion of some relatively important QK-pairs, thus hurting the model’s accuracy. Moreover, different datasets exhibit varying features of sparsity due to their distinct data types and tasks. Consequently, in the pre-deployment preparation, the value of _k_ can be modified to optimize the algorithm’s exploiting of sparsity to minimize computation while maintaining accuracy. For example, 

1256 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:19:49 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [512 x 260] intentionally omitted <==**

**----- Start of picture text -----**<br>
0% Loss 1% Loss 2% Loss<br>80%<br>60%<br>40%<br>20%<br>Fig. 18. Computation reduction by LP with diverse loss tolerance. [X, Y] respectively denote the computation reduction for the Atten part and QKV+Atten.<br>11<br>we observed that datasets like SST2 and STS-B, used for GPU(0% loss) GPU(1% loss) GPU(2% loss) SOFA(0% loss) SOFA(1% loss) SOFA(2% loss) 9.5<br>9<br>sentiment classification or semantic analysis, typically exhibit 7<br>high sparsity because one or two keywords often indicate 5<br>sentiment. Therefore, their computation reduction is adjusted 3<br>1<br>90% while the accuracy loss is controlled within 1%. In (a) GeoMean<br>contrast, image datasets generally contain a high amount of 119 GPU (LP 2% loss) GPU(LP 2% loss+FlashAttention1) GPU(LP 2% loss+FlashAttention2) SOFA (2% loss) 9.5<br>key information and have lower data redundancy compared to 7<br>text classification datasets, resulting in lower sparsity. As a 53<br>result, their computation reduction is adjusted to 73% with a 1<br>1% accuracy loss. (b)<br>[0.3775, 0.6044]<br>[0.2021, 0.4598] [0.1271, 0.3868] [0.0863, 0.3466] [0.1679, 0.4253] [0.1062, 0.3666] [0.0510, 0.3096] [0.0882, 0.3586] [0.055, 0.3138] [0.0318, 0.2880] [0.1292, 0.3968] [0.0995, 0.3599] [0.0735, 0.3334] [0.2826, 0.5312] [0.1563, 0.4141] [0.0550, 0.3138] [0.2921, 0.5381] [0.1763, 0.4341] [0.0672, 0.3268] [0.1974,0.4354] [0.1271, 0.3668] [0.0863, 0.3265] [0.1521, 0.3913] [0.0855, 0.3274] [0.0572, 0.2948] [0.1184, 0.3583] [0.0735, 0.3134] [0.0491, 0.2809] [0.1621,0.3954] [0.1131,0.3537] [0.0863, 0.3265] [0.2564, 0.4698 [0.1638, 0.4012] [0.0798, 0.3199] [0.2597, 0.4254] [0.1488, 0.3873] [0.0798, 0.3199] [0.2375, 0.3732] [0.1834, 0.4263] [0.1438, 0.2253] [0.2622, 0.5363] [0.2030, 0.4908] [0.1689, 0.3758] [0.1062, 0.3145] [0.0735, 0.2814] [0.1735, 0.3966] [0.1415, 0.3486] [0.0863, 0.2945] [0.1715, 0.3735] [0.1215, 0.3186] [0.0883, 0.2645] [0.2116, 0.4364] [0.1724, 0.3946] [0.1363, 0.3445] [0.1415, 0.3221] [0.1115, 0.3486] [0.1063, 0.2545] [0.1915, 0.4132] [0.1355, 0.3886] [0.0963, 0.3245] [0.1872, 0.4321] [0.1234, 0.3736] [0.0740, 0.3260]<br>Normalized Computation<br>Throughput Gain<br>3.01x<br>3.57x<br>5.4x<br>Throughput Gain 1.76<br>**----- End of picture text -----**<br>


Fig. 18. Computation reduction by LP with diverse loss tolerance. [X, Y] respectively denote the computation reduction for the Atten part and QKV+Atten. 

we observed that datasets like SST2 and STS-B, used for sentiment classification or semantic analysis, typically exhibit high sparsity because one or two keywords often indicate sentiment. Therefore, their computation reduction is adjusted to 90% while the accuracy loss is controlled within 1%. In contrast, image datasets generally contain a high amount of key information and have lower data redundancy compared to text classification datasets, resulting in lower sparsity. As a result, their computation reduction is adjusted to 73% with a 1% accuracy loss. 

## _C. Architecture Evaluation_ 

_Throughput Improvement:_ Fig. 19 (a) compares the throughput of SOFA with A100 GPU on all benchmarks versus diverse accuracy loss. As can be seen, LP enables 1 _._ 08-1 _._ 78 _×_ of speed up on GPU with its sparsity detection. Unfortunately, the GPU cannot leverage the LP results as it cannot handle high sparsity or fine-grained on-demand KV calculations. Nor can it run the cross-stage DLZS-based prediction efficiently. By contrast, the SOFA exhibits an average 85 _._ 2% PE utilization due to its stage-fused fine-grained tiled dataflow, which pipelines crossstage DLZS prediction, SADS sorting, and SU-FA, leading to almost triple sparsity utilization than GPU. Further, the SU-FA engine is tailored to support sparsity attention acceleration with reduced computational complexity. Overall, SOFA achieves 6 _._ 1 _×_ , 7 _._ 2 _×_ and 9 _._ 5 _×_ inference speed up with 0% _/_ 1% _/_ 2% accuracy degradation. Fig.19 (b) further compares the SOFA with LP+traditional FA and LP+FA2 on A100. On average, FA on GPU brings around 1 _._ 5 _×_ gain, leading to a total 2 _._ 7 _×_ speed up combined with LP. By adjusting the loop order to avoid some factor scaling nonlinear computations, FA2 achieved a further 1 _._ 19 _×_ throughput improvement. However, due to the difficulty of fine-grained cross-stage data movement on GPUs and the challenges of optimizing FA1/2 to support fine-grained scheduling and sparse computation, it is difficult to achieve higher improvements. By contrast, SOFA (soft+archi) achieves 9 _._ 5 _×_ gain, which is 3 _._ 01 _×_ greater than vanilla LP+FA2 on GPU. Fig. 20 (a) shows the memory access reduction effectiveness of SOFA. Compared with the baseline with vanilla dynamic sparsity, SOFA with RASS can reduce average 23% memory access. With SU-FA and tiled dataflow, the reduction rises further 79%. 

Fig. 21 (a) illustrates the breakdown of throughput improvement achieved by GPU A100 and TPU with the hardwaresoftware mechanism of SOFA. The baseline is executing a 

Fig. 19. Throughput gain of SOFA over (a) LP (b) LP+FA-1/2 on A100 GPU. 

dense Transformer model on GPU/TPU. With SOFA software optimization, GPU and TPU achieve improvements of 3 _._ 16 _×_ and 2 _._ 8 _×_ , respectively. However, both of them cannot fully leverage all the benefits of SOFA software. GPU performs better than TPU due to its better ability to handle some of the fine-grained computations and scheduling in SOFA software. Adding SOFA’s engines incrementally, we observed significant performance gains. The GPU with the DLZS engine achieves a 1 _._ 65 _×_ speedup due to the systolic data flow improving data reuse, which the GPU’s vector engine cannot support. The TPU with the DLZS engine shows an even higher improvement of 1 _._ 82 _×_ because its limited control instructions are inefficient at handling DLZS’s logical branching. Similarly, the SADS engine, with its customized data paths, achieves a 1 _._ 28 _×_ improvement on the GPU and 1 _._ 52 _×_ on the TPU by quickly and efficiently executing redundant computations. Further, the SU-FA engine improves performance by 1 _._ 26 _×_ on the GPU and 1 _._ 1 _×_ on the TPU due to its max-assured circuits that avoid inefficient recomputation and data movement caused by log-domain calculation errors. The SU-FA engine employs a systolic array design. Since the GPU’s support for systolic arrays is inferior to that of the TPU, it achieves a greater speedup than TPU. Lastly, the RASS unit achieves improvements of 1 _._ 14 _×_ on the GPU and 1 _._ 3 _×_ on the TPU owing to its customized control unit, which enables more efficient scheduling and data arrangement. 

_Area, Power and Energy:_ Table III shows the power and area breakdown of SOFA accelerator. It has a total area of 5 _._ 69 mm[2] , and LP accounts for merely 18% and 15% of area and power. This benefits from the multiplier and converter-free 

1257 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:19:49 UTC from IEEE Xplore.  Restrictions apply. 

TABLE II 

SUMMARY AND COMPARISON WITH SOTA WORKS. 

|||**Software Performance**|**Software Performance**|**Software Performance**|||||||**Hardware **|**Hardware **|**Performance**|**Performance**|**Performance**|||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**Accelerators**||Sparsityx|Accu<br>Loss|Saved<br>Compy|Tech<br>[nm]|Freq<br>[Hz]|Area<br>[mm2]|Power <br>Core||[W]<br>IO||Throup.<br>[GOPS]||Energy Eff.z [GOPS/W]<br>Core<br>Device_†_||Area Eff.z<br>[GOPS/mm2]|Latency<br>[ms]|
|**A**3 [28]||Unstr|5_._3%|40%|40|1G|2.08|0.205||0.617||221||1863|**300**|**217**|**622**|
|**ELSA** [29]||Unstr|2%|73%|40|1G|1.26|0.969||0.525||1090||1944|**1004**|**1765**|**252**|
|**Sanger** [30]||Str|0%|76%|55|500M|16.9|2.76||-||2285||2342|-|**522**|**241**|
|**DOTA** [31]||Str|0_._8%|80%|22|1G|4.44|3.02||-||4905||817|-|**683**|**448**|
|**Energon** [34]||Unstr|0_._9%|77%|45|1G|4.2|0.32||2.4||1153||7007|**450**|**709**|**477**|
|**DTATrans** [32]||Unstr|0_._74%|74%|40|1G|1.49|0.734||-||1304||3071|-|**1786**|**652**|
|**SpAtten** [33]||Str|0_._9%|67%|40|1G|1.55|0.325||0.617||360||1915|**447**|**474**|**382**|
|**FACT** [23]||Unstr|0%|79%|28|500M|6.03|0.337||-||928||2754|-|**154**|**296**|
|**SOFA**||Unstr|0%|82%|28|1G|5.69|0.95||2.45||24423||25708|**7183**|**4292**|**45**|



> x Unstructured or Structured sparsity. y Comp saving = Reduced attention computaion _−_ Prediction computation. _†_ Device = IO + Core. 

> z Scaled to 28nm and 1.0V CMOS with _f ∝_ 1 _/s_ 2 and power (core) _∝_ (1 _/s_ )(1 _._ 0 _/V dd_ )2, where _s_ =Tech/28nm [64], [68]. 

TABLE III 

AREA AND POWER BREAKDOWN FOR SOFA (CORE PART) AT 1GHZ. 

|AREA ANDPOWERB|REAKDOWN FORSOFA|(COREPART) AT1GHZ|
|---|---|---|
|**Modules**|**Parameters**<br>**Ar**|**ea[mm**2**]**<br>**Power[mW]**|
|DLZS prediction|128_×_32 shift PEs<br>128 LZEs|0_._351<br>29_._05|
|Iterative SADS|128 16-4 sort cores<br>128 clipping units|0_._679<br>112_._79|
|KV generation<br>SU-FA module|128_×_4 16bit PEs<br>128_×_4 16bit PEs<br>128 EXP units<br>128 DIV units|0_._875<br>146_._21<br>3_._012<br>485_._12|
|Memory|192KB Token SRAM<br>96KB Weight SRAM<br>28KB TempSRAM|0_._497<br>170_._23|
|Scheduler & Others|-|0_._280<br>6_._45|
|Off-Chip DRAM|HBM2, 16_×_ HBM|channels @ 2GHz|
|**Total**|TSMC 28nm: Area=5_._69|mm2, Power=949_._85mW|



TABLE IV 

||POWER|BREAKDOWN OF|SOFA.||
|---|---|---|---|---|
||**Core Part**|**Memory Interface**|**DRAM**|**Overall**|
|**Power**|0_._95W|0_._53W|1_._92W|3_._40W|



> x The DRAM and Interface power are estimated with 59 _._ 8GB/s. 

**==> picture [251 x 240] intentionally omitted <==**

**----- Start of picture text -----**<br>
Vanilla dynamic sparsity (LP) SOFA (LP+RASS) SOFA (LP+RASS+SU-FA+Tiled Pipeline Dataflow)<br>100%<br>80%<br>60%<br>40%<br>20%<br>(a) GeoMean<br>SOFA(0% loss) SOFA(1% loss) SOFA(2% loss) 71.5<br>70<br>50<br>30<br>10<br>(b)<br>Fig. 20. (a) Memory access reduction of SOFA. (b) Efficiency gain of SOFA<br>over Nvidia A100 GPU.<br>������������������������������ ���������������������������������������������<br>Dense Model on<br>GPU/TPU GPU TPU GPU<br>GPU/TPU+S ��������� ����<br>GPU/TPU+SD ���������� �����<br>GPU/TPU+SDA ����� ����� ����<br>GPU/TPU+SDAU ��������� �����<br>GPU/TPU+SDAUR ����� ���� �����<br>� � � � � � � � � � �� �� �� � �� �� �� �� �� �� �� ��<br>(S: Software, D: DLZS engine, A: SADS engine, U: SU-FA engine, R: RASS Unit)<br>23%<br>79%<br>Memory Access<br>Efficiency Gain<br>**----- End of picture text -----**<br>


Fig. 20. (a) Memory access reduction of SOFA. (b) Efficiency gain of SOFA over Nvidia A100 GPU. 

design in DLZS engine and the low-overhead design of SADS engine. Fig. 20 (b) illustrates the overall energy-efficiency gain of SOFA compared to the A100 GPU. On average, SOFA achieves 49 _._ 8 _×_ , 57 _._ 6 _×_ , and 71 _._ 5 _×_ greater energy efficiency in comparison to the A100 GPU with 0%, 1% and 2% accuracy loss, respectively. In Fig. 21 (b), we also show the efficiency gain breakdown. DLZS and SADS engines bring 2 _._ 48 _×_ and 2 _._ 1 _×_ efficiency gain, respectively. Further, SU-FA and RASS units together bring about 3 _._ 27 _×_ gain. In Table IV, we list the power overhead consumed by the memory interface [69] and external DRAM. 

## _D. Comparison with Existing Acclerators_ 

FACT, Sanger, Energon, SpAtten, ELSA and DOTA are SOTA Transformer dynamic sparsity accelerators. However, their designs focus on computational optimization, overlooking that memory access becomes the de facto bottleneck after computational optimization. The head pruning technique in SpAtten can partly alleviate memory access issues, but its efficiency is limited as it fundamentally depends on the 

Fig. 21. (a) Throughput gain of GPU/TPU with SOFA’s mechanism (b) Energy efficiency gain of GPU with SOFA’s mechanism. 

characteristics of the task. On the other hand, although Energon considers a certain computation-to-memory access ratio in its architecture design, it still suffers from inefficiency due to the variability of computational tasks and models. In summary, previous efforts still lack simultaneously optimizing both computation and memory access. When imbalance arises between computation and memory access due to sparsity, it hampers further enhancement of hardware efficiency. SOFA employs a holistic FlashAttention-like scheme to divide all work stages of dynamic sparsity into fine-grained tile manner, and leverages the sort information for cross-stage collaborative optimization. Table II compares the features of software and hardware performance across existing SOTA accelerators. Benefiting from the low complexity of LP mechanism, SOFA achieves the greatest reduction (82%) in computation at the same accuracy loss of 0%. We list their hardware parameters and present a normalized comparison [64], [68] of energy 

1258 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:19:49 UTC from IEEE Xplore.  Restrictions apply. 

and area efficiency. Compared to these SOTA accelerators, SOFA achieves a device (core + I/O) energy efficiency of 7183 GOPS/W, representing an average improvement of 7 _._ 2 _×_ to 24 _×_ . This improvement stems from the fine-grained data flow achieved through collaborative cross-stage optimization, which effectively reduces off-chip memory accesses. Additionally, SOFA achieves 4292 GOPS/mm[2] area efficiency, which is 2 _._ 4 _×_ to 27 _._ 9 _×_ better than the SOTA accelerators. The gain in area efficiency primarily arises from the algorithm-hardware co-optimization for low complexity. 

In Table II, we also quantitatively compare the latency of the SOTA accelerators by evaluating them to execute an attention part (137GOPs) of Llma7B. For fairness, all accelerators are scaled to 128 multipliers clocked at 1GHz. For example, the effective throughput of FACT is 928 GOPS in 500MHz with 512 multipliers. Then its execution latency is 2*137/928=296ms. Compared to the 0% loss accelerators FACT and Sanger, SOFA achieves 6 _._ 6 _×_ and 5 _,_ 4 _×_ latency reduction, respectively. Moreover, SOFA achieves 8 _._ 5 _×_ and 10 _._ 6 _×_ latency decrease over SpAtten and Energon, respectively. Such reduction in SOFA latency is mainly attributed to the fine-grained tiling execution across stages, as illustrated in Fig.6. 

## VI. RELATED WORKS AND DISCUSSION 

**Efficient Transformer Accelerator.** Numerous studies [23], [28]–[34], [70]–[75] have been proposed to improve the energy efficiency and speed of Transformer inference. However, most of these works focus on attention computation reduction, including static sparsity [75]–[78], dynamic sparsity [23], [28]–[34] and hybrid sparsity [79]. However, when computation is optimized, the memory access would dominate the overall power and time, especially for LTPP scenarios, which these works ignore. By contrast, SOFA optimizes both compute and memory access, thus greatly outperforming previous works. Further, all dynamic sparsity efforts focus on individually optimizing each stage for higher efficiency. Unlike these works, SOFA exhibits a cross-stage holistic optimization. This provides SOFA with an ever-overlooked opportunity for cross-stage tiling, executing a fine-grained tiled dataflow that accelerates inference while reducing off-chip memory access. 

**Neural network accelerator with sparsity.** There are very many ASIC or FPGA accelerators [70], [72], [80]–[98] that leverage sparsity to optimize the performance of neural network inference. There also exist general sparse tensor algebra accelerators [99]–[104] proposed in recent years, which can be used to process sparse FC layers. Recently, works [105]– [107] utilize hierarchical sparsity to construct a comprehensive design space and provide accurate performance metrics, which enable the automatic and optimal design of sparse DNN accelerators. However, most of the works focus on exploiting pre-trained static sparse weights. By contrast, SOFA leverages LP to predict on-the-fly dynamic sparsity. Especially, such sparsity comes from the _argmax approximation_ property of softmax, thus needing to be detected actively. This makes the traditional near-zero-based sparsity methods inapplicable. Through recently some works are config for activation sparsity 

[108] and both weight and activation sparsity [106], [109], [110], they are all based on the near-zero sparsity, thus failing to the top- _k_ sparsity scene, which is SOFA targets. 

**Fused operator tiling accelerators.** Many works [111]– [118] leverage layer-fusion strategy to optimize the DNN inference performance. Specifically, DNNBuilder [117] and DeFiNES [118] use a depth-first-like layer fusion in CNNs to enhance data reuse via cross-layer tiling, enabled by the weak operator dependencies in CNNs. However, _dynamic sparsity_ of Transformers face bottlenecks due to row dependency in the top- _k_ /softmax operator, restricting dynamic sparsity for long sequences. SOFA addresses this by employing the DCE data distribution property, unlocking the possibility of depthfirst-like execution in Transformer _dynamic sparsity_ for the first time. DeepBurning [119] partitions NN graphs at the inter-operator granularity and executes them in a pipeline fashion. In contrast, SOFA achieves finer-grained execution by dividing within the operator, leading to more efficient SRAM utilization. FLAT [120] fuses the two matmul operators and softmax in attention to reduce off-chip memory access but fails to resolve softmax row dependency. Traditional FlashAttention [39], [40] successfully unlocks the row dependency of softmax but at the expense of surging computation costs. In this aspect, SOFA leverages SU-FA to successfully solve the row dependency in softmax, allowing for finer-grained tiling and reducing SU-FA complexity using top- _k_ sorting information. 

## VII. CONCLUSION 

We propose SOFA, a cross-stage compute-memory efficient algorithm-hardware co-design to accelerate dynamic sparsity Transformer inference for LTPP. We introduce a novel logdomain DLZS computing paradigm to estimate Q-K pairs with add-only operation, requiring less converters. To prevent memory access from becoming a bottleneck after sparsity computation optimization, we propose SADS and SU-FA to enable cross-stage tiling for the end-to-end workflow. Leveraging this tiling strategy, SOFA executes a fine-grained pipeline dataflow across diverse stages, effectively mitigating memory access and latency issues. Efficient architecture is designed to support and accelerate the above mechanisms with a memoryefficient reuse-aware schedule. SOFA achieves 71 _._ 5 _×_ energy saving than Nvidia A100 GPU, and average 15 _._ 8 _×_ higher energy efficiency than eight SOTA accelerators, respectively. 

## ACKNOWLEDGMENT 

We would like to thank Mingcong Song, for his valuable suggestions. This work was supported in part by the National Science and Technology Major Project under Grant 2022ZD0115201; the NSFC under Grant 62125403, and Grant 92164301; Beijing S&T Project Z221100007722023; in part by the project funding for the 2022 Special Project on Industrial Foundation Reconstruction and High Quality Development of Manufacturing Industry CEIEC-2022-ZM02-0245; in part by the Beijing National Research Center for Information Science and Technology; and in part by the Beijing Advanced Innovation Center for Integrated Circuits. 

1259 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:19:49 UTC from IEEE Xplore.  Restrictions apply. 

## REFERENCES 

- [1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. attentionis all you need. _Advances in neural information processing systems_ , 30, 2017. 

- [2] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877– 1901, 2020. 

- [3] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional Transformers for language understanding. _arXiv preprint arXiv:1810.04805_ , 2018. 

- [4] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite BERT for self-supervised learning of language representations. _arXiv preprint arXiv:1909.11942_ , 2019. 

- [5] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized BERT pretraining approach. _arXiv preprint arXiv:1907.11692_ , 2019. 

- [6] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018. 

- [7] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9, 2019. 

- [8] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text Transformer. _The Journal of Machine Learning Research_ , 21(1):5485– 5551, 2020. 

- [9] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. _arXiv preprint arXiv:1910.01108_ , 2019. 

- [10] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-LM: Training multibillion parameter language models using model parallelism. _arXiv preprint arXiv:1909.08053_ , 2019. 

- [11] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with Transformers. In _Proceedings of the European conference on computer vision_ , pages 213–229, 2020. 

- [12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv preprint arXiv:2010.11929_ , 2020. 

- [13] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In _Proceedings of the International Conference on Machine Learning_ , pages 12888–12900, 2022. 

- [14] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, Furu Wei, and Baining Guo. Swin Transformer V2: Scaling up capacity and resolution. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 12009–12019, 2022. 

- [15] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin Transformer: Hierarchical vision Transformer using shifted windows. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 10012–10022, 2021. 

- [16] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In _Proceedings of the International Conference on Machine Learning_ , pages 8748–8763, 2021. 

- [17] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10684–10695, 2022. 

- [18] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision Transformers. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 12104–12113, 2022. 

- [19] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable DETR: Deformable Transformers for end-to-end object detection. _arXiv preprint arXiv:2010.04159_ , 2020. 

- [20] Zhanghao Wu, Zhijian Liu, Ji Lin, Yujun Lin, and Song Han. Lite Transformer with long-short range attention. In _Proceedings of the International Conference on Learning Representations_ , 2019. 

- [21] Claire Cardie Faisal Ladhak, Esin Durmus and Kathleen McKeown. WikiLingua: A new benchmark dataset for multilingual abstractive summarization. In _Findings of EMNLP, 2020_ , 2020. 

- [22] Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and Jianfeng Gao. Large language models: A survey. _arXiv preprint arXiv:2402.06196_ , 2024. 

- [23] Yubin Qin, Yang Wang, Dazheng Deng, Zhiren Zhao, Xiaolong Yang, Leibo Liu, Shaojun Wei, Yang Hu, and Shouyi Yin. FACT: FFNattention co-optimized Transformer architecture with eager correlation prediction. In _Proceedings of the 50th Annual International Symposium on Computer Architecture_ , pages 1–14, 2023. 

- [24] Sehoon Kim, Coleman Hooper, Thanakul Wattanawong, Minwoo Kang, Ruohan Yan, Hasan Genc, Grace Dinh, Qijing Huang, Kurt Keutzer, Michael W Mahoney, Yakun Sophia Shao, and Amir Gholami. Full stack optimization of Transformer inference: A survey. _arXiv preprint arXiv:2302.14017_ , 2023. 

- [25] Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue, and Wenhu Chen. Longcontext LLMs struggle with long in-context learning. _arXiv preprint arXiv:2404.02060_ , 2024. 

- [26] Bingyang Wu, Shengyu Liu, Yinmin Zhong, Peng Sun, Xuanzhe Liu, and Xin Jin. LoongServe: Efficiently serving long-context large language models with elastic sequence parallelism. _arXiv preprint arXiv:2404.09526_ , 2024. 

- [27] Jiaheng Liu, Zhiqi Bai, Yuanxing Zhang, Chenchen Zhang, Yu Zhang, Ge Zhang, Jiakai Wang, Haoran Que, Yukang Chen, Wenbo Su, Tiezheng Ge, Jie Fu, Chen Wenhu, and Bo Zheng. E[2] -LLM: Efficient and extreme length extension of large language models. _arXiv preprint arXiv:2401.06951_ , 2024. 

- [28] Tae Jun Ham, Sung Jun Jung, Seonghak Kim, Young H Oh, Yeonhong Park, Yoonho Song, Jung-Hun Park, Sanghee Lee, Kyoung Park, Jae W Lee, and Deog-Kyoon Jeong. A[3] : Accelerating attention mechanisms in neural networks with approximation. In _Proceedings of the IEEE International Symposium on High Performance Computer Architecture (HPCA)_ , pages 328–341, 2020. 

- [29] Tae Jun Ham, Yejin Lee, Seong Hoon Seo, Soosung Kim, Hyunji Choi, Sung Jun Jung, and Jae W Lee. ELSA: Hardware-software co-design for efficient, lightweight self-attention mechanism in neural networks. In _Prceedings of the 48th ACM/IEEE Annual International Symposium on Computer Architecture (ISCA)_ , pages 692–705, 2021. 

- [30] Liqiang Lu, Yicheng Jin, Hangrui Bi, Zizhang Luo, Peng Li, Tao Wang, and Yun Liang. Sanger: A co-design framework for enabling sparse attention using reconfigurable architecture. In _Proceedings of the 54th Annual IEEE/ACM International Symposium on Microarchitecture_ , pages 977–991, 2021. 

- [31] Zheng Qu, Liu Liu, Fengbin Tu, Zhaodong Chen, Yufei Ding, and Yuan Xie. DOTA: Detect and omit weak attentions for scalable Transformer acceleration. In _Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems_ , pages 14–26, 2022. 

- [32] Tao Yang, Fei Ma, Xiaoling Li, Fangxin Liu, Yilong Zhao, Zhezhi He, and Li Jiang. DTATrans: Leveraging dynamic token-based quantization with accuracy compensation mechanism for efficient Transformer architecture. _IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems_ , 42(2):509–520, 2022. 

- [33] Hanrui Wang, Zhekai Zhang, and Song Han. SpAtten: Efficient sparse attention architecture with cascade token and head pruning. In _Proceedings of the IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ , pages 97–110, 2021. 

- [34] Zhe Zhou, Junlin Liu, Zhenyu Gu, and Guangyu Sun. Energon: Toward efficient acceleration of Transformers using dynamic sparse attention. 

1260 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:19:49 UTC from IEEE Xplore.  Restrictions apply. 

   - _IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems_ , 42(1):136–149, 2022. 

- [35] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving. _arXiv preprint arXiv:2401.09670_ , 2024. 

- [36] Pratyush Patel, Esha Choukse, Chaojie Zhang,[´] I˜nigo Goiri, Aashaka Shah, Saeed Maleki, and Ricardo Bianchini. Splitwise: Efficient generative LLM inference using phase splitting. _arXiv preprint arXiv:2311.18677_ , 2023. 

- [37] Weilin Zhao, Yuxiang Huang, Xu Han, Chaojun Xiao, Zhiyuan Liu, and Maosong Sun. Ouroboros: Speculative decoding with large model enhanced drafting. _arXiv preprint arXiv:2402.13720_ , 2024. 

- [38] Samuel Williams, Andrew Waterman, and David Patterson. Roofline: An insightful visual performance model for multicore architectures. _Communications of the ACM_ , 52(4):65–76, 2009. 

- [39] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. FlashAttention: Fast and memory-efficient exact attention with IOawareness. _Advances in Neural Information Processing Systems_ , 35:16344–16359, 2022. 

- [40] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. _arXiv preprint arXiv:2307.08691_ , 2023. 

- [41] Richard P Brent and Paul Zimmermann. _Modern computer arithmetic_ , volume 18. Cambridge University Press, 2010. 

- [42] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- [43] Szymon Tworkowski, Konrad Staniszewski, Mikołaj Pacek, Yuhuai Wu, Henryk Michalewski, and Piotr Miło´s. Focused Transformer: Contrastive training for context scaling. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [44] Christoforos Kachris. A survey on hardware accelerators for large language models. _arXiv preprint arXiv:2401.09890_ , 2024. 

- [45] Mark Horowitz. Computing’s energy problem (and what we can do about it). In _Proceedings of the IEEE international solid-state circuits conference digest of technical papers (ISSCC)_ , pages 10–14, 2014. 

- [46] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. LLaMa: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 

- [47] Pierre Blanchard, Desmond J Higham, and Nicholas J Higham. Accurately computing the log-sum-exp and softmax functions. _IMA Journal of Numerical Analysis_ , 41(4):2311–2330, 2021. 

- [48] Nebojˇsa Z Milenkovi´c, Vladimir V Stankovi´c, and Miljana Lj Mili´c. Modular design of fast leading zeros counting circuit. _Journal of Electrical Engineering_ , 66(6):329–333, 2015. 

- [49] Shih-Hsiang Lin, Pei-Yin Chen, and Yu-Ning Lin. Hardware design of low-power high-throughput sorting unit. _IEEE Transactions on Computers_ , 66(8):1383–1395, 2017. 

- [50] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In _Proceedings of the International Conference on Learning Representations_ , 2018. 

- [51] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions for machine comprehension of text. In _Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing_ , pages 2383–2392, 2016. 

- [52] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagn´e, Alexandra Sasha Luccioni, Franc¸ois Yvon, Matthias Gall´e, et al. Bloom: A 176B-parameter openaccess multilingual language model. _arXiv preprint arXiv:2211.05100_ , 2022. 

- [53] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In _Proceedings of the International Conference on Learning Representations_ , 2016. 

- [54] Faisal Ladhak, Esin Durmus, Claire Cardie, and Kathleen McKeown. WikiLingua: A new benchmark dataset for cross-lingual abstractive summarization. _arXiv preprint arXiv:2010.03093_ , 2020. 

- [55] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial Winograd schema challenge at scale. _Communications of the ACM_ , 64(9):99–106, 2021. 

- [56] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision Transformer: A versatile backbone for dense prediction without convolutions. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 568–578, 2021. 

- [57] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 248–255, 2009. 

- [58] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in PyTorch. 2017. 

- [59] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. Transformers: State-of-the-art natural language processing. In _Proceedings of the Conference on Empirical Methods in Natural Language Processing: System Demonstrations_ , pages 38–45, 2020. 

- [60] Naveen Muralimanohar, Rajeev Balasubramonian, and Norman P Jouppi. CACTI 6.0: A tool to model large caches. _HP laboratories_ , 27:28, 2009. 

- [61] Yoongu Kim, Weikun Yang, and Onur Mutlu. Ramulator: A fast and extensible DRAM simulator. _IEEE Computer architecture letters_ , 15(1):45–49, 2015. 

- [62] Lukas Cavigelli and Luca Benini. Origami: A 803-GOp/s/W convolutional network accelerator. _IEEE Transactions on Circuits and Systems for Video Technology_ , 27(11):2461–2475, 2016. 

- [63] Renzo Andri, Lukas Cavigelli, Davide Rossi, and Luca Benini. YodaNN: An ultra-low power convolutional neural network accelerator based on binary weights. In _Proceedings of the IEEE Computer Society Annual Symposium on VLSI (ISVLSI)_ , pages 236–241, 2016. 

- [64] Yizhi Wang, Jun Lin, and Zhongfeng Wang. An energy-efficient architecture for binary weight convolutional neural networks. _IEEE Transactions on Very Large Scale Integration (VLSI) Systems_ , 26(2):280–293, 2017. 

- [65] Wilson Snyder. Verilator and SystemPerl. In _North American SystemC Users’ Group, Design Automation Conference_ , 2004. 

- [66] Norman P Jouppi, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Boden, Al Borchers, Rick Boyle, Cantin Pierre-luc, Clifford Chao, Chris Clark, Jeremy Coriell, Mike Daley, Matt Dau, Jeffrey Dean, Ben Gelb, Tara Vazir Ghaemmaghami, Rajendra Gottipati, William Gulland, Robert Hagmann, C Richard Ho, Doug Hogberg, John Hu, Robert Hundt, Dan Hurt, Julian Ibarz, Aaron Jaffey, Alek Jaworski, Alexander Kaplan, Harshit Khaitan, Andy Koch, Naveen Kumar, Steve Lacy, James Laudon, James Law, Diemthu Le, Chris Leary, Zhuyuan Liu, Kyle Lucke, Alan Lundin, Gordon MacKean, Adriana Maggiore, Maire Mahony, Kieran Miller, Rahul Nagarajan, Ravi Narayanaswami, Ray Ni, Kathy Nix, Thomas Norrie, Mark Omernick, Narayana Penukonda, Andy Phelps, Jonathan Ross, Matt Ross, Amir Salek, Emad Samadiani, Chris Severn, Gregory Sizikov, Matthew Snelham, Jed Souter, Dan Steinberg, Andy Swing, Mercedes Tan, Gregory Thorson, Bo Tian, Horia Toma, Erick Tuttle, Vijay Vasudevan, Richard Walter, Walter Wang, Eric Wilcox, and Doe Hyun Yoon. In-datacenter performance analysis of a tensor processing unit. In _Proceedings of the 44th annual international symposium on computer architecture_ , pages 1–12, 2017. 

- [67] Google. Google cloud TPU. https://cloud.google.com/tpu. Accessed: 2024-06-25. 

- [68] Leibo Liu, Guiqiang Peng, Pan Wang, Sheng Zhou, Qiushi Wei, Shouyi Yin, and Shaojun Wei. Energy-and area-efficient recursive-conjugategradient-based MMSE detector for massive MIMO systems. _IEEE Transactions on Signal Processing_ , 68:573–588, 2020. 

- [69] Brian Leibowitz, Robert Palmer, John Poulton, Yohan Frans, Simon Li, John Wilson, Michael Bucher, Andrew M Fuller, John Eyles, Marko Aleksic, Trey Greer, and Nhat M Nguyen. A 4.3 GB/s mobile memory interface with power-efficient bandwidth scaling. _IEEE Journal of Solid-State Circuits_ , 45(4):889–898, 2010. 

- [70] Chao Fang, Aojun Zhou, and Zhongfeng Wang. An algorithm-hardware co-optimized framework for accelerating N:M sparse Transformers. _IEEE Transactions on Very Large Scale Integration (VLSI) Systems_ , 30(11):1573–1586, 2022. 

1261 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:19:49 UTC from IEEE Xplore.  Restrictions apply. 

- [71] Seongmin Hong, Seungjae Moon, Junsoo Kim, Sungjae Lee, Minsub Kim, Dongsoo Lee, and Joo-Young Kim. DFX: A low-latency multiFPGA appliance for accelerating Transformer-based text generation. In _Proceedings of the 55th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , pages 616–630, 2022. 

- [72] Zheng Li, Soroush Ghodrati, Amir Yazdanbakhsh, Hadi Esmaeilzadeh, and Mingu Kang. Accelerating attention through gradient-based learned runtime pruning. In _Proceedings of the 49th Annual International Symposium on Computer Architecture_ , pages 902–915, 2022. 

- [73] Amir Yazdanbakhsh, Ashkan Moradifirouzabadi, Zheng Li, and Mingu Kang. Sparse attention acceleration with synergistic in-memory pruning and on-chip recomputation. In _Proceedings of the 55th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , pages 744– 762, 2022. 

- [74] Ali Hadi Zadeh, Isak Edo, Omar Mohamed Awad, and Andreas Moshovos. GOBO: Quantizing attention-based NLP models for low latency and energy efficient inference. In _Proceedings of the 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , pages 811–824, 2020. 

- [75] Bingbing Li, Santosh Pandey, Haowen Fang, Yanjun Lyv, Ji Li, Jieyang Chen, Mimi Xie, Lipeng Wan, Hang Liu, and Caiwen Ding. FTRANS: Energy-efficient acceleration of Transformers using FPGA. In _Proceedings of the ACM/IEEE International Symposium on Low Power Electronics and Design_ , pages 175–180, 2020. 

- [76] Haoran You, Zhanyi Sun, Huihong Shi, Zhongzhi Yu, Yang Zhao, Yongan Zhang, Chaojian Li, Baopu Li, and Yingyan Lin. ViTCoD: Vision Transformer acceleration via dedicated algorithm and accelerator codesign. In _Proceedings of the IEEE International Symposium on HighPerformance Computer Architecture (HPCA)_ , pages 273–286, 2023. 

- [77] Guan Shen, Jieru Zhao, Quan Chen, Jingwen Leng, Chao Li, and Minyi Guo. SALO: An efficient spatial accelerator enabling hybrid sparse attentionmechanisms for long sequences. In _Proceedings of the 59th ACM/IEEE Design Automation Conference_ , pages 571–576, 2022. 

- [78] Hongxiang Fan, Thomas Chau, Stylianos I Venieris, Royson Lee, Alexandros Kouris, Wayne Luk, Nicholas D Lane, and Mohamed S Abdelfattah. Adaptable butterfly accelerator for attention-based NNs via hardware and algorithm co-design. In _Proceedings of the 55th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , pages 599–615, 2022. 

- [79] Jieru Zhao, Pai Zeng, Guan Shen, Quan Chen, and Minyi Guo. Hardware-software co-design enabling static and dynamic sparse attentionmechanisms. _IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems_ , 2024. 

- [80] Reza Hojabr, Ali Sedaghati, Amirali Sharifian, Ahmad Khonsari, and Arrvindh Shriraman. SPAGHETTI: Streaming accelerators for highly sparse GEMM on FPGAs. In _Proceedings of the IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ , pages 84–96, 2021. 

- [81] Ashish Gondimalla, Noah Chesnut, Mithuna Thottethodi, and TN Vijaykumar. SparTen: A sparse tensor accelerator for convolutional neural networks. In _Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture_ , pages 151–165, 2019. 

- [82] Bahar Asgari, Ramyad Hadidi, Tushar Krishna, Hyesoon Kim, and Sudhakar Yalamanchili. ALRESCHA: A lightweight reconfigurable sparse-computation accelerator. In _Proceedings of the IEEE International Symposium on High Performance Computer Architecture (HPCA)_ , pages 249–260, 2020. 

- [83] Huizheng Wang, Zaichen Zhang, Xiaohu You, and Chuan Zhang. Low-complexity winograd convolution architecture based on stochastic computing. In _Proceedings of the 23rd IEEE International Conference on Digital Signal Processing (DSP)_ , pages 1–5, 2018. 

- [84] Chunhua Deng, Yang Sui, Siyu Liao, Xuehai Qian, and Bo Yuan. GoSPA: An energy-efficient high-performance globally optimized sparse convolutional neural network accelerator. In _Proceedings of the ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)_ , pages 1110–1123, 2021. 

- [85] Eric Qin, Ananda Samajdar, Hyoukjun Kwon, Vineet Nadella, Sudarshan Srinivasan, Dipankar Das, Bharat Kaul, and Tushar Krishna. Sigma: A sparse and irregular GEMM accelerator with flexible interconnects for DNN training. In _Proceedings of the IEEE International Symposium on High Performance Computer Architecture (HPCA)_ , pages 58–70, 2020. 

- [86] Huizheng Wang, Weihong Xu, Zaichen Zhang, Xiaohu You, and Chuan Zhang. An efficient stochastic convolution architecture based on fast 

   - FIR algorithm. _IEEE Transactions on Circuits and Systems II: Express Briefs_ , 69(3):984–988, 2021. 

- [87] Sumanth Gudaparthi, Sarabjeet Singh, Surya Narayanan, Rajeev Balasubramonian, and Visvesh Sathe. CANDLES: Channel-aware novel dataflow-microarchitecture co-design for low energy sparse neural network acceleration. In _Proceedings of the IEEE International Symposium on high-performance computer architecture (HPCA)_ , pages 876–891, 2022. 

- [88] Edward Hanson, Shiyu Li, Hai’Helen’ Li, and Yiran Chen. Cascading structured pruning: Enabling high data reuse for sparse DNN accelerators. In _Proceedings of the 49th Annual International Symposium on Computer Architecture_ , pages 522–535, 2022. 

- [89] Jonathan S Lew, Yunpeng Liu, Wenyi Gong, Negar Goli, R David Evans, and Tor M Aamodt. Anticipating and eliminating redundant computations in accelerated sparse training. In _Proceedings of the 49th Annual International Symposium on Computer Architecture_ , pages 536–551, 2022. 

- [90] Gang Li, Weixiang Xu, Zhuoran Song, Naifeng Jing, Jian Cheng, and Xiaoyao Liang. Ristretto: An atomized processing architecture for sparsity-condensed stream flow in CNN. In _Proceedings of the 55th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , pages 1434–1450, 2022. 

- [91] Shiyu Li, Edward Hanson, Xuehai Qian, Hai” Helen” Li, and Yiran Chen. ESCALATE: Boosting the efficiency of sparse CNN accelerator with kernel decomposition. In _Proceedings of the 54th Annual IEEE/ACM International Symposium on Microarchitecture_ , pages 992– 1004, 2021. 

- [92] Zhi-Gang Liu, Paul N Whatmough, Yuhao Zhu, and Matthew Mattina. S2TA: Exploiting structured sparsity for energy-efficient mobile CNN acceleration. In _Proceedings of the IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ , pages 573–586, 2022. 

- [93] Julian Pavon, Ivan Vargas Valdivieso, Adrian Barredo, Joan Marimon, Miquel Moreto, Francesc Moll, Osman Unsal, Mateo Valero, and Adrian Cristal. VIA: A smart scratchpad for vector units with application to sparse matrix computations. In _Proceedings of the IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ , pages 921–934, 2021. 

- [94] Alexander Rucker, Matthew Vilim, Tian Zhao, Yaqi Zhang, Raghu Prabhakar, and Kunle Olukotun. Capstan: A vector RDA for sparsity. In _Proceedings of the 54th Annual IEEE/ACM International Symposium on Microarchitecture_ , pages 1022–1035, 2021. 

- [95] Fazle Sadi, Joe Sweeney, Tze Meng Low, James C Hoe, Larry Pileggi, and Franz Franchetti. Efficient SPMV operation for large and highly sparse matrices using scalable multi-way merge parallelization. In _Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture_ , pages 347–358, 2019. 

- [96] Sumit Walia, Bachu Varun Tej, Arpita Kabra, Joydeep Devnath, and Joycee Mekie. Fast and low-power quantized fixed posit high-accuracy DNN implementation. _IEEE Transactions on Very Large Scale Integration (VLSI) Systems_ , 30(1):108–111, 2021. 

- [97] Dingqing Yang, Amin Ghasemazar, Xiaowei Ren, Maximilian Golub, Guy Lemieux, and Mieszko Lis. Procrustes: A dataflow and accelerator for sparse deep neural network training. In _Proceedings of the 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , pages 711–724, 2020. 

- [98] Ashish Gondimalla, Mithuna Thottethodi, and T. N. Vijaykumar. Eureka: Efficient tensor cores for one-sided unstructured sparsity in DNN inference. In _Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture_ , page 324–337, 2023. 

- [99] Nitish Srivastava, Hanchen Jin, Jie Liu, David Albonesi, and Zhiru Zhang. MatRaptor: A sparse-sparse matrix multiplication accelerator based on row-wise product. In _Proceedings of the 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , pages 766–780, 2020. 

- [100] Mostafa Mahmoud, Isak Edo, Ali Hadi Zadeh, Omar Mohamed Awad, Gennady Pekhimenko, Jorge Albericio, and Andreas Moshovos. TensorDash: Exploiting sparsity to accelerate deep neural network training. In _Proceedings of the 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , pages 781–795, 2020. 

- [101] Youngeun Kwon, Yunjae Lee, and Minsoo Rhu. TensorDIMM: A practical near-memory processing architecture for embeddings and tensor operations in deep learning. In _Proceedings of the 52nd Annual_ 

1262 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:19:49 UTC from IEEE Xplore.  Restrictions apply. 

_IEEE/ACM International Symposium on Microarchitecture_ , pages 740– 753, 2019. 

- [102] Kartik Hegde, Hadi Asghari-Moghaddam, Michael Pellauer, Neal Crago, Aamer Jaleel, Edgar Solomonik, Joel Emer, and Christopher W Fletcher. ExTensor: An accelerator for sparse tensor algebra. In _Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture_ , pages 319–333, 2019. 

- [103] Konstantinos Kanellopoulos, Nandita Vijaykumar, Christina Giannoula, Roknoddin Azizi, Skanda Koppula, Nika Mansouri Ghiasi, Taha Shahroodi, Juan Gomez Luna, and Onur Mutlu. SMASH: Codesigning software compression and hardware-accelerated indexing for efficient sparse matrix operations. In _Proceedings of the 52nd annual IEEE/ACM international symposium on microarchitecture_ , pages 600– 614, 2019. 

- [104] Yuedan Chen, Guoqing Xiao, Fan Wu, Zhuo Tang, and Keqin Li. tpSpMV: A two-phase large-scale sparse matrix-vector multiplication kernel for manycore architectures. _Information Sciences_ , 523:279–295, 2020. 

- [105] Yannan Nellie Wu, Po-An Tsai, Angshuman Parashar, Vivienne Sze, and Joel S Emer. Sparseloop: An analytical approach to sparse tensor accelerator modeling. In _Proceedings of the 55th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , pages 1377– 1395, 2022. 

acceleration processor with effective-weight-based convolution and error-compensation-based prediction. _IEEE Journal of Solid-State Circuits_ , 57(5):1542–1557, 2021. 

   - [117] Xiaofan Zhang, Junsong Wang, Chao Zhu, Yonghua Lin, Jinjun Xiong, Wen-mei Hwu, and Deming Chen. DNNBuilder: An automated tool for building high-performance DNN hardware accelerators for FPGAs. In _Proceedings of the IEEE/ACM International Conference on ComputerAided Design (ICCAD)_ , pages 1–8, 2018. 

   - [118] Linyan Mei, Koen Goetschalckx, Arne Symons, and Marian Verhelst. DeFinNES: Enabling fast exploration of the depth-first scheduling space for DNN accelerators through analytical modeling. In _Proceedings of the IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ , pages 570–583, 2023. 

   - [119] Xuyi Cai, Ying Wang, Xiaohan Ma, Yinhe Han, and Lei Zhang. DeepBurning-SEG: Generating DNN accelerators of segment-grained pipeline architecture. In _Proceedings of the 55th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , pages 1396– 1413, 2022. 

   - [120] Sheng-Chun Kao, Suvinay Subramanian, Gaurav Agrawal, Amir Yazdanbakhsh, and Tushar Krishna. FLAT: An optimized dataflow for mitigating attention bottlenecks. In _Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems_ , pages 295–310, 2023. 

- [106] Yannan Nellie Wu, Po-An Tsai, Saurav Muralidharan, Angshuman Parashar, Vivienne Sze, and Joel Emer. HighLight: Efficient and flexible DNN acceleration with hierarchical structured sparsity. In _Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture_ , pages 1106–1120, 2023. 

- [107] Jong Hoon Shin, Ali Shafiee, Ardavan Pedram, Hamzah Abdel-Aziz, Ling Li, and Joseph Hassoun. Griffin: Rethinking sparse optimization for deep learning architectures. In _Proceedings of the IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ , pages 861–875, 2022. 

- [108] Jun-Woo Jang, Sehwan Lee, Dongyoung Kim, Hyunsun Park, Ali Shafiee Ardestani, Yeongjae Choi, Channoh Kim, Yoojin Kim, Hyeongseok Yu, Hamzah Abdel-Aziz, Jun-Seok Park, Heonsoo Lee, Dongwoo Lee, Myeong Woo Kim, Hanwoong Jung, Heewoo Nam, Dongguen Lim, Seungwon Lee, Joon-Ho Song, Suknam Kwon, Joseph Hassoun, SukHwan Lim, and Changkyu Choi. Sparsity-aware and reconfigurable NPU architecture for Samsung flagship mobile SoC. In _Proceedings of the 48th Annual ACM/IEEE International Symposium on Computer Architecture (ISCA)_ , pages 15–28, 2021. 

- [109] Guyue Huang, Zhengyang Wang, Po-An Tsai, Chen Zhang, Yufei Ding, and Yuan Xie. RM-STC: Row-merge dataflow inspired GPU sparse tensor core for energy-efficient sparse acceleration. In _Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture_ , pages 338–352, 2023. 

- [110] Yang Wang, Chen Zhang, Zhiqiang Xie, Cong Guo, Yunxin Liu, and Jingwen Leng. Dual-side sparse tensor core. In _Proceedings of the 48th Annual ACM/IEEE International Symposium on Computer Architecture (ISCA)_ , pages 1083–1095, 2021. 

- [111] Manoj Alwani, Han Chen, Michael Ferdman, and Peter Milder. Fusedlayer CNN accelerators. In _Proceedings of the 49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , pages 1–12, 2016. 

- [112] Koen Goetschalckx and Marian Verhelst. DepFiN: A 12nm, 3.8 TOPs depth-first CNN processor for high Res. image processing. In _Proceedings of the Symposium on VLSI Circuits_ , pages 1–2, 2021. 

- [113] Dongseok Im, Donghyeon Han, Sungpill Choi, Sanghoon Kang, and Hoi-Jun Yoo. DT-CNN: An energy-efficient dilated and transposed convolutional neural network processor for region of interest based image segmentation. _IEEE Transactions on Circuits and Systems I: Regular Papers_ , 67(10):3471–3483, 2020. 

- [114] Juhyoung Lee, Dongjoo Shin, Jinsu Lee, Jinmook Lee, Sanghoon Kang, and Hoi-Jun Yoo. A full HD 60 fps CNN super resolution processor with selective caching based layer fusion for mobile devices. In _Proceedings of the Symposium on VLSI Circuits_ , pages C302–C303, 2019. 

- [115] Feng Min, Haobo Xu, Ying Wang, Yujie Wang, Jiajun Li, Xingqi Zou, Bei Li, and Yinhe Han. Dadu-Eye: A 5.3 TOPS/W, 30 fps/1080p high accuracy stereo vision accelerator. _IEEE Transactions on Circuits and Systems I: Regular Papers_ , 68(10):4207–4220, 2021. 

- [116] Huiyu Mo, Wenping Zhu, Wenjing Hu, Qiang Li, Ang Li, Shouyi Yin, Shaojun Wei, and Leibo Liu. A 12.1 TOPS/W quantized network 

1263 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 07:19:49 UTC from IEEE Xplore.  Restrictions apply. 

