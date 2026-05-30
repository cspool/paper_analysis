# VII. EVALUATION

In this section, we evaluate the effectiveness of proposed optimizations in VQ-LLM through comprehensive experiments. We first present overall speedup results for various VQ-based computation kernels over existing approaches. Then, we provide a detailed breakdown analysis of the proposed optimizations. Next, we compare our work with FP16 kernels and several element-wise quantization works to show its viability for accelerating LLMs. Finally, we performed a comprehensive end-to-end evaluation, analyzing both the overall speedup and accuracy across various GPUs.

## A. Experimental Setup

In this study, we conduct a comprehensive evaluation at both the individual kernel and end-to-end model levels. The evaluations were performed on an NVIDIA RTX 4090 24GB GPU [44]. For the end-to-end evaluation, we included a Tesla A40 GPU [39] to explore the potential of VQ-LLM with lower bandwidth.. The evaluated computation kernels

TABLE IV Break down analysis configuration

| ID       | Optimization<br>Category | Llescription                                                                          |  |  |
|----------|--------------------------|---------------------------------------------------------------------------------------|--|--|
| GC       | No                       | Naive implementation                                                                  |  |  |
| SC       | Greedy                   | Cache all entries in shared memory                                                    |  |  |
| O1<br>O2 | Hierarchical<br>Buffer   | + Shared memory level caching (medium entries) + Register level caching (hot entries) |  |  |
| O3<br>O4 | Compute<br>Engine        | + Codebook centric dataflow<br>+ Codebook centric hierarchical fusion                 |  |  |

include various VQ-augmented GeMM, GeMV and FlashDecoding [10]. The evaluated VQ configurations are listed in Tbl. II, including QuiP#-4, AQLM-3, GPTVQ-2 and CQ-2/4, the suffix number represent the equivalent bit-width. The first two kernels adopt weight quantization and the last one adopts KV cache quantization. For the kernel-level evaluation, we set the shape for these kernels following the Llama-7B and Llama-65B [55] models. These kernels run on a single GPU, while large model serving like Llama-65B typically uses multiple GPUs with Tensor Parallel (TP) strategy [35], [47], [71]. The required adjustments to our framework include final results gathering for Attention and partial results concatenation/reduction for GeMM/GeMV [38]. These are usually conducted via communication library like NCCL [45], and we identify this distributed scenario an orthogonal topic and leave it to the future work.

Tbl. IV lists various baselines and VQ-LLM optimizations used in our evaluation. For the baselines, we use **GC** and **SC** method explained in Sec. III that stores the codebook in global memory and shared memory, respectively. For the results, we report the latency reduction against **GC**. We also decompose the optimizations used in VQ-LLM into four levels (**O1-O4**), with each explained in Tbl. IV. We also compare VQ-LLM with SOTA element-wise quantization methods under the same equivalent 4-bit width, including AWQ [30] for GeMM/GeMV and QoQ for Attention [31], all integrated into qServe [31]. For FP16, we use cutlass [43] and flash-attn [8].

In practice, LLM inference involves various operators beyond GeMM/GeMV and Attention, such as RMSNorm [66], SiLU [14], and RoPE [53], etc. Therefore, it is crucial to evaluate the end-to-end speedup that accounts for all operators. For the end-to-end evaluation, we set a batch size of 16 and a sequence length of 1024, measuring the total latency for generating 256 tokens. We also assess accuracy using the arc-challenge task [5], applying the QuiP#-4 and CQ-4 algorithms for quantizing the weights and KV-Cache, respectively. To obtain the final accuracy results, we integrate these algorithms into the LMEval framework [16].

