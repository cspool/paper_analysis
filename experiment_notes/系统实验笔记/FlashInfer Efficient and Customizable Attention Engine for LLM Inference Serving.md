## FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving

- 属于Serving调度的实现是什么？实验比较什么？
  实现FlashInfer v0.2作为LLM serving框架的attention backend，集成入SGLang v0.3.4和MLC-Engine进行端到端serving优化。核心Serving调度层面的设计：(i) **统一block-sparse KV-cache格式**——将PageAttention和RadixAttention等非连续KV-cache存储统一映射为BSR矩阵，query/output用ragged tensor存储消除padding，与上层serving框架的page table / radix tree管理透明对接；(ii) **Composable formats for prefix-caching**——利用多BSR格式分解KV-cache稀疏矩阵：共享prefix对应的dense submatrix用大$B_r$存储（3 queries共享同一KV-cache于shared memory），唯一suffix用$B_r=1$存储（各query在自己的threadblock内独立访问），无需数据移动仅需计算不同submatrix的indices/index pointer arrays；(iii) **Dynamic scheduling per generation step**——CPU端plan function处理变化的sequence length信息生成load-balanced调度计划，可跨层复用（同一步内所有decode attention共享plan），plan info cached on GPU workspace buffer；(iv) **CUDAGraph-compatible persistent kernel**——attention+contraction merged into single persistent kernel with fixed grid size，workspace buffer fixed offset保证CUDAGraph pointer不变，plan function (on CPU) not captured by CUDAGraph，run function (on GPU) captured in graph。

  实验比较：(i) **Standard LLM serving**——FlashInfer+SGLang vs SGLang-Triton，Llama 3.1 8B (1xH100)和70B (4xH100)，ShareGPT + Variable synthetic workload (seqlen uniform 512-2048)，测量P99 TTFT<200ms约束下的ITL和TTFT；(ii) **Long-context inference (Streaming-LLM)**——FlashInfer fused RoPE+attention kernel vs unfused kernels，Vicuna-13B on MT-Bench，变化Streaming-LLM recent window size；(iii) **Parallel generation with prefix-caching**——FlashInfer composable formats vs single format in MLC-Engine，Llama 3.1 8B/70B on ShareGPT，request rate=16，变化parallel tokens n∈{1,2,4,8,16,32,64}，测量ITL和TTFT。

- 硬件平台是什么，配置是什么。
  - Standard serving: 1× NVIDIA H100 80GB SXM (Llama 3.1 8B)，4× NVIDIA H100 80GB SXM (Llama 3.1 70B)
  - Long-context: NVIDIA A100 (Streaming-LLM, Vicuna-13B)
  - Parallel generation: H100 80GB SXM (Llama 3.1 8B/70B on MLC-Engine)
  - CUDA 12.4 + PyTorch 2.4.0，FP16存储和计算

- 开源Serving框架是什么。修改了什么。
  - 开源框架：SGLang v0.3.4 (https://github.com/sgl-project/sglang)，MLC-Engine (https://github.com/mlc-ai/mlc-llm)
  - FlashInfer修改：不修改上层serving框架的调度逻辑（continuous batching / radix tree管理 / page table管理保持不变），而是作为drop-in attention backend替换。在SGLang中替换Triton attention backend，在MLC-Engine中增加composable formats支持。
  - Serving框架通过FlashInfer的PyTorch编程接口（Listing 1）集成：创建workspace buffer→compile阶段：对每种attention spec+task_info组合创建AttentionWrapper→JIT compile kernel→dummy plan→capture CUDAGraph→runtime阶段：选择最优CUDAGraph→每generation step更新seqlen_info→attn.plan() (CPU)→g.replay() (GPU executed via CUDAGraph)。
  - Composable formats实现：prefix-caching场景下创建多个AttentionWrapper（不同block sizes对应shared prefix和unique suffix），framework根据KV-cache配置选择最优CUDAGraph。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  已开源：https://github.com/flashinfer-ai/flashinfer (Apache-2.0)，已集成进SGLang、vLLM、MLC-Engine。

  使用例子（SGLang + FlashInfer, Llama 3.1 8B, 1xH100, ShareGPT dataset, online serving）：
  1. **输入**：ShareGPT对话请求流（variable-length prompts），SGLang API server接收请求，continuous batching调度器维护运行中请求batch。请求按到达顺序被调度，P99 TTFT target < 200ms。
  2. **SGLang RadixAttention管理**：SGLang维护radix tree结构管理KV-cache pages（prefix-caching），KV-cache以non-contiguous page table形式存储在GPU HBM中。当请求到达时，SGLang的scheduler将请求加入running batch，更新page table。
  3. **FlashInfer作为attention backend**：
     a. **Compile阶段**（init时）：对每种task_info（attention variant spec + 平均query长度范围 + block size配置）调用`AttentionWrapper(attn_spec, task_info, workspace_buffer)`→JIT编译生成CUDA kernel（核心是BSR FlashAttention模板）→dummy plan→捕获对应CUDAGraph。多种配置对应多个CUDAGraphs（不同composable formats / tile size组合）。
     b. **Runtime text generation loop**（per step）：`seqlen_info.update()`（查询当前batch中各请求的query长度和KV-cache长度）→`attn.plan(seqlen_info)`（CPU端Algorithm 1计算load-balanced CTA调度plan，跨所有layer复用）→`g.replay()`（GPU端CUDAGraph执行：persistent attention kernel处理可变长度KV-cache chunks → persistent contraction kernel将partial outputs用⊕操作合并为final attention outputs）。
     c. **Attention计算细节**（decode step）：Q ragged tensor（各请求1 token）→ BSR KV-cache（page table映射为non-zero blocks）→ per-CTA根据plan info处理分配的KV chunks → partial attention states (O_partial, LSE_partial) → ⊕ compose → final O。
  4. **Prefill处理**：FlashInfer prefill kernel使用FA2/FA3 algorithm处理prompt tokens（query长度=prompt长度），BSR格式处理页表映射。FlashInfer prefill kernel配合causal masking。
  5. **输出**：生成tokens返回客户端。Throughput = total output tokens / total time。ITL = decode step平均时间。TTFT = prefill + 第一次decode时间。
  6. **性能**（Figure 7）：FlashInfer backend vs Triton backend——Llama 3.1 8B ITL reduction 29-69%，TTFT reduction 29-69%（ShareGPT）；Llama 3.1 70B ITL reduction 29-69%，TTFT reduction 29-69%。Variable synthetic workload similarly consistent speedup across all settings。
  
  并行生成例子（MLC-Engine + FlashInfer composable formats, Llama 3.1 70B, 4xH100, ShareGPT, n=4 parallel tokens）：
  1. **输入**：ShareGPT请求，每个请求要求并行生成n=4条回复（OpenAI API "n" parameter equivalent）
  2. **Prefix-caching**：所有4条回复共享输入prompt prefix的KV-cache。MLC-Engine的prefix-caching机制识别shared prefix
  3. **Composable format分解**：
     - Sub-matrix 1 (shared prefix): BSR with $B_r=3, B_c=1$。3 queries共享同一KV-cache page，在shared memory中复用加载的K/V tile（high-bandwidth SMEM access），减少global memory traffic
     - Sub-matrix 2 (unique suffixes): BSR with $B_r=1, B_c=1$。各query独立访问各自unique KV-cache，通过global memory / L2 cache
     - 无需KV-cache data movement，仅计算两种submatrices的`kv_indptr`和`kv_indices`数组
  4. **Attention computation**：每个submatrix使用独立的AttentionWrapper（不同$B_r$对应的不同CUDAGraph）。大$B_r$的shared prefix部分tensor core GEMM复用shared memory中KV tile（Q tile (3, head_dim) × K tile (l_kv, head_dim)），显著降低global memory bandwidth需求
  5. **输出**：ITL reduction 17.42%（8B为13.73%），TTFT reduction 22.86%（8B为16.41%）at n=4 (peak speedup)。n<4时block size增大不足抵消overhead，n>32时attention不再dominate computation（短sequence ShareGPT场景），speedup plateau
  
  - **作用**：FlashInfer作为通用attention backend统一解决LLM serving中KV-cache异构存储（paged/radix tree）、workload动态性（variable-length sequences per batch）、attention变体多样性（GQA/MQA/specialized masks/fused operations）三个核心问题。将上层serving框架的page table / radix tree数据结构直接映射为BSR sparse matrix格式，实现memory layout到compute kernel的端到端优化，无需修改框架的调度和内存管理逻辑。
