## High-Throughput Non-Uniformly Quantized 3-bit LLM Inference

- baseline方法是什么？
  Baseline方法分为两类：(1) Non-uniform 3-bit quantization方法（SqueezeLLM、Any-Precision LLM、Bitsandbytes），使用K-means clustering将FP16权重映射为3-bit index+per-row centroids，在CUDA cores上完成dequantization和matrix-vector multiplication；(2) Uniform quantization方法（GPTQ），使用线性映射将权重量化为低比特整数。

  Baseline全栈执行例子（以SqueezeLLM 3-bit在A100上跑OPT-30B为例）：
  - **算法层**：K-means clustering将每行权重映射为8个FP16 centroids+3-bit index。dequantization做centroid[C[Wq]] pointer chasing lookup恢复FP16 weight，matmul在CUDA cores上执行matrix-vector multiplication（每token顺序乘权重矩阵的一列）。
  - **系统框架层**：论文未明确说明，SqueezeLLM提供Python inference API通过HuggingFace Transformers调用。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：SqueezeLLM dequantization和matmul均在CUDA cores上完成，bit packing采用padding（10个3-bit元素塞32-bit word，浪费2 bits）或spanning（32个3-bit元素跨3个32-bit word，破坏coalesced memory access）。dequantization产生大量bitwise/type-conversion指令和cache-unfriendly indirect memory access（decoding W†=C[Wq]需pointer chasing）。Tensor Cores完全未被使用，因为3-bit sub-byte operand layout与Tensor Core所需interleaved pattern不兼容。
  - **硬件架构层**：NVIDIA A100/L40 GPU。

  Baseline的核心缺陷：
  (1) **内存节省未转化为加速**：SqueezeLLM在OPT-30B上memory reduction 4.07×但latency反增3.01×（vs FP16 baseline），matmul时间占GPU总时间92%
  (2) **低效bit packing**：3-bit与32/64-bit GPU word不对齐，padding浪费内存带宽，spanning破坏coalesced access引入分支和warp divergence
  (3) **高dequantization overhead**：CUDA cores上centroid pointer chasing lookup产生大量指令，batch增大时instruction count指数增长（图3）
  (4) **Tensor Core underutilization**：dequantization和matmul均压在CUDA cores，Tensor Cores空闲等待（pipeline bubble），GPU最强算力单元未参与核心matmul

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**Quantix**，一个"离线布局转换+在线fused kernel"的GPU执行框架，在不改变non-uniform量化模型accuracy的前提下将内存节省转化为实际推理加速。

  论文方法全栈执行例子（以Quantix 3-bit在L40上跑LLaMA-65B为例）：
  - **算法层**：沿用SqueezeLLM等non-uniform quantization的K-means clustering结果（Wq和C）。Quantix的offline bit shuffling对Wq做两步变换：(a) bit dividing将每个3-bit index拆为1-bit matrix Wq,1和2-bit matrix Wq,2，使32个1-bit元素恰好填32-bit word、32个2-bit元素恰好填64-bit word，消除padding浪费和spanning跨界；(b) bit mapping按64×64 warp tile→16×16 Tensor Core tile层次，将每个thread负责的元素收集为连续linear segments W1'/W2'，确保128-bit coalesced vectorized access且匹配Tensor Core operand layout。bit shuffling是lossless w.r.t. quantized model（不改变centroids），完全保留原non-uniform量化精度。
  - **系统框架层**：Quantix fused kernel集成进HuggingFace Transformers替换SqueezeLLM默认inference backend。对uniform baselines (GPTQ/Marlin)使用AutoGPTQ library。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Quantix fused dequantization-matmul kernel将数据搬运、dequantization和MMA融合为单一CUDA kernel的执行pipeline。具体：(a) inter-tile层：shared memory double buffering (Smem0/Smem1)，cp.async 128-bit预取下一K-tile的W1'/W2'/A到shared memory，与当前tile计算重叠；(b) intra-tile层：register double buffering (Reg0/Reg1)，CUDA cores在registers中做bit concatenation [1-bit]+[2-bit]→3-bit index→shift+mask按qi=(R>>3i)&0x7并行提取→centroid lookup得到FP16 W†，Tensor Cores同时消费另一register buffer的W†执行MMA；(c) in-register dequantization使FP16 weight仅在registers中存在并直接进入Tensor Core，消除了baseline中dequantize→write memory→read memory→matmul的多级内存路径和cache miss；(d) Split-K提升小矩阵并行度；(e) 128-bit UINT4 vectorized memory access最大化memory bandwidth。
  - **硬件架构层**：NVIDIA L40 GPU（面向inference优化）和A100 GPU。L40上kernel平均speedup 4.82× over FP16 cuBLAS。GPU utilization profiling显示Quantix维持>90% cache hit rate（baseline在batch增大时降至接近0%），有效规避"memory wall"。

  对应解决Baseline缺陷：
  **(1) 内存节省未转化为加速 → 全栈co-design将内存节省转为吞吐**：fused kernel+bit shuffling使3-bit Quantix在L40在kernel-level平均4.82× over FP16 cuBLAS，端到端LLaMA-65B可单GPU运行（FP16不行），最高相对SqueezeLLM 11.46× speedup
  **(2) 低效bit packing → bit dividing将奇数bit拆为1-bit+2-bit双路**：1/2均为32/64的因子，天然对齐GPU word边界。消除padding的内存浪费和spanning的跨界访问、branching和warp divergence。对比naive padding（10个3-bit→32-bit浪费2 bits）和spanning（32个3-bit→跨3个word），bit dividing后每路均perfectly packed
  **(3) 高dequantization overhead → in-register dequantization消除pointer chasing和cache miss**：bit concatenation和centroid indexing在registers内部完成，使用shift+mask (qi=(R>>3i)&0x7) 无分支；centroids也保持在registers中，无需indirect memory access。Ablation显示in-register dequantization是最大贡献组件（移除后性能降至40%）
  **(4) Tensor Core underutilization → fused kernel让CUDA cores做dequantization准备，Tensor Cores做matmul**：hierarchical pipeline使dequantization latency被Tensor Core compute隐藏。GPU utilization分析显示Quantix在batch size增长时Tensor Core利用率持续提升，而SqueezeLLM完全不用Tensor Cores。inter-tile+intra-tile双层double buffering消除了baseline中"Tensor Cores等dequantization"的pipeline bubble
  **(5) 关键trade-off**：non-uniform quantization用更多centroids换取更好accuracy（3-bit SqueezeLLM perplexity 6.15 vs GPTQ 7.55），但centroid overhead在compute-bound大batch场景下使4-bit Quantix可能慢于4-bit Marlin（uniform quantization dequantization更简单）。A100上因更高memory bandwidth，memory-saving benefit相对L40更小。过大批量时register pressure可能导致spilling影响ALU utilization
