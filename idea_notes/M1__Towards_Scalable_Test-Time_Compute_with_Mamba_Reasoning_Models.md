## M1__Towards_Scalable_Test-Time_Compute_with_Mamba_Reasoning_Models

- baseline方法是什么？
  Baseline是DeepSeek-R1-Distill-Qwen-1.5B，一种基于标准Transformer架构的推理模型。Qwen2.5-Math-1.5B通过超过1T MATH tokens的SFT在Qwen2.5基础模型上训练，然后通过DeepSeek-R1的蒸馏流程（从大R1模型进行token-level蒸馏）获得推理能力。在推理时，模型通过生成长chain-of-thought（平均4k-5k tokens per MATH question）来解决复杂数学问题。

  Baseline全栈执行例子（DeepSeek-R1-Distill-Qwen-1.5B推理时生成一个MATH token，batch_size=512）：
  - 算法pipeline：token → embedding lookup → L层Transformer block（每层: RMSNorm → Multi-Head Attention with GQA: QKV投影 → RoPE位置编码 → causal softmax(QK^T/√d) over full KV cache → weighted sum V → output projection → residual → RMSNorm → SwiGLU FFN → residual）→ LM head → logits → next token。对于长chain-of-thought推理，每生成一个token需要对所有历史token计算QK内积（O(N)计算量），KV cache随生成长度线性增长。当生成4k-5k tokens的推理链时，batch size=512的KV cache总量约512×5000×2×n_layers×d_head字节（1.5B约28层×1536 hidden dim → 每token约86KB per layer → 28层×86KB≈2.4MB/token → 512batch×5000tokens≈6TB总KV cache，显存需求极大）。
  - 系统框架：vLLM 0.6.3推理引擎 + PagedAttention KV cache管理。VeRL框架用于rollout生成。推理时compute-bound的prefill（处理短prompt）和memory-bound的decode（生成长chain-of-thought）两阶段分离。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用vLLM内置的FlashAttention等优化kernel）。
  - 硬件架构：NVIDIA H100 GPU，论文未涉及RTL/模拟器层面。

  Baseline缺陷：
  1. **Transformer的KV cache内存爆炸**：长chain-of-thought生成（4k-32k tokens）时，per-token KV cache随生成长度线性增长。大batch（512）推理时，KV cache总内存需求远超GPU HBM（H100 80GB），成为batch size和推理吞吐量的硬瓶颈。解码过程memory-bound导致GPU计算单元利用率低。
  2. **Transformer的二次计算复杂度限制test-time scaling**：生成k个样本做self-consistency/majority voting时，每个样本的每个token需attend所有历史token（O(N²)）。长chain-of-thought（32k tokens）×大样本数（64 samples）× batch推理的总FLOPs极高，限制了test-time compute scaling的实践可行性。
  3. **跨架构推理蒸馏效果未知**：DeepSeek-R1系列仅蒸馏到Transformer架构（Qwen/Llama），能否将推理能力迁移到sub-quadratic架构（如Mamba）并保持性能是未解问题。直接尝试从R1蒸馏到Mamba效果差（38% MATH500, 3.3% AIME24），说明需要创新的训练方案。
  4. **线性RNN在推理任务上的有效性不确定**：虽hybrid RNN模型在通用LM上表现良好，但现代推理模型需要生成长chain-of-thought（包含subtask分解、多尝试、回溯），线性RNN的固定大小hidden state是否能支撑这种复杂推理模式未知。
  5. **RL训练中生成长度受限**：RL训练（GRPO）需要高效生成长序列rollout。Transformer在RL训练的rollout阶段成为瓶颈——生成时间超过actor权重更新（forward+backward）的3倍（DeepScaleR时间分析），训练效率严重受限于生成速度。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出M1——基于Mamba架构的hybrid线性RNN推理模型，通过三阶段training pipeline（蒸馏+SFT+RL）将Transformer推理能力迁移到Mamba，实现3x推理加速并将加速转化为test-time compute scaling的准确率增益。

  论文方法全栈执行例子（M1-3B推理时生成一个MATH token，batch_size=512）：
  - 算法pipeline：token → embedding → 28层hybrid block（22 Mamba层 + 6 interleaved Attention层。Mamba层: RMSNorm → input projection expand 2x → causal Conv1d kernel=4 → SiLU → selective SSM scan with state size=16, groups=192: Δ_t=softplus(Linear(x)+bias), A_bar/B_bar=discretize(A, B_t, Δ_t), h_t=A_bar⊙h_{t-1}+B_bar⊗x [O(1) per token, state h∈R^{16×192}] → C_t^T h_t → SiLU gating → output projection → residual。Attention层: 标准MHA/GQA保留。MLP: SwiGLU）→ LM head → logits → next token。每生成一个token，Mamba层仅需O(1)计算和常量内存（h_t固定16×192维），无KV cache增长。仅6/28=21%的attention层需要KV cache。Batch=512时Mamba层内存≈512×16×192×22层×4 bytes≈137MB（远小于Transformer的GB级别）。
  - 系统框架：vLLM 0.6.3推理引擎（利用PagedAttention管理attention层的KV cache）。VeRL框架用于GRPO RL训练的rollout生成（修复了CUDA graph+FSDP兼容性，5x训练加速）。训练框架：Axolotl（蒸馏/SFT阶段）。开源代码：https://github.com/jxiw/M1。
  - 编译框架：论文未明确说明。
  - kernel调度：论文修复了VeRL中Mamba+CUDA graph+PyTorch FSDP的兼容性问题，使CUDA graph启用后Mamba生成速度提升5x。Mamba的selective scan使用硬件高效并行实现（沿非时间维度并行+SRAM resident state）。
  - 硬件架构：NVIDIA H100 GPU，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（KV cache内存爆炸）→ Mamba层的固定大小hidden state替代KV cache：每层Mamba仅需维护h_t∈R^{N×N'}（SSM state 16×192≈3072维），而非per-token K/V cache。batch=512×seqlen=4096时，22层Mamba的state内存=512×3072×22×4≈137MB vs 22层attention的KV cache约512×4096×1536×22×4≈276GB（2000x+缩减）。这使得大batch下GPU内存不再是瓶颈，**解码从memory-bound转为compute-bound**，GPU利用率提升，实测3x吞吐量提升（15169 T/s vs 7263 T/s）。

  - 缺陷2（二次复杂度限制test-time scaling）→ M1 decode O(1) per token：生成k个样本的cost=k×seqlen×O(1)（M1）vs k×seqlen²（Transformer）。固定时间预算下M1可生成更多样本（或更长的chain-of-thought），使self-consistency voting等test-time scaling技术更实用。**速度增益直接转化为准确率增益**（Figure 3 right: 同等时间预算下M1 majority voting accuracy更高；Figure 4 right: M1在同等生成时间下4/5个长度点的accuracy更高）。

  - 缺陷3（跨架构推理蒸馏未知）→ 三阶段pipeline的创新策略：(a) **先做通用MATH蒸馏再做推理SFT**而非直接从R1蒸馏：先用OpenMathInstruct-2（Llama系列generated）将hybrid Mamba训练成强MATH模型（MATH500 45%→74%），再用10B reasoning tokens做推理SFT（74%→82%）。这克服了直接跨架构推理蒸馏的数据不足问题（仅10B reasoning tokens不够，需先建立math基础）；(b) **Reverse KL divergence**用于蒸馏（mode-seeking特性），比forward KL更适合将teacher的概率分布模式集中到student有限的表达能力中；(c) **GQA→full heads expansion**：用额外线性层将Transformer的GQA KV heads扩展到Mamba的full heads，补偿Mamba无KV cache造成的表达能力损失。

  - 缺陷4（线性RNN推理有效性未知）→ Table 1/2证明M1在AIME25/AIME24/MATH500/AMC23/OlympiadBench上**全面匹配**DeepSeek-R1-Distill-Qwen-1.5B（甚至OlympiadBench上M1 47.3 vs R1 43.3），且仅用<50B tokens训练（vs R1的>1T MATH tokens）。这证明了：(a) hybrid Mamba足以支持复杂数学推理；(b) 6个保留attention层（21%）足以弥补纯Mamba在长程信息路由上的不足；(c) GRPO RL对Mamba架构同样有效。

  - 缺陷5（RL训练生成瓶颈）→ M1的3x生成加速使RL训练rollout阶段大幅缩短：(a) 训练时生成长度可扩展至32k（vs Transformer受限于生成速度），更长的chain-of-thought在RL中带来更高准确率（Figure 5: max_len 4096→<10% accuracy, 24k→23%）；(b) CUDA graph+FSDP修复使Mamba生成额外5x加速（在VeRL框架内）；(c) 论文分析指出RL训练中"生成速度>3x actor更新速度"的瓶颈可被线性RNN架构缓解。

  **Stage-by-stage ablation分析**（Table 3）：
  | Stage | MATH500 | AIME24 | 增益分析 |
  |-------|---------|--------|---------|
  | Distill | 38 | 0 | 基础跨架构迁移，无推理能力 |
  | +SFT(MATH) | 45 | 0 | 通用MATH能力建立，无推理 |
  | +SFT(Reason) | 74 | 22 | **最大增益**：推理数据带来+29/+22 |
  | +RL (GRPO) | 82 | 28 | RL进一步+8/+6，巩固推理能力 |

  **为什么先蒸馏Llama而非直接蒸馏R1？**
  论文做了直接蒸馏R1的实验（Distill from DeepSeek-R1-Qwen-1.5B + SFT on 10B reasoning data）→ 仅38%/3.3%（MATH500/AIME24）。假设原因是10B reasoning tokens不足以进行有效的跨架构推理迁移。替代策略：先用OpenMathInstruct-2（Llama系列data）建立Mamba MATH基础模型，再用reasoning data做推理SFT——这种"先通用后专项"的分阶段迁移策略仅需少量reasoning tokens即可获得强推理性能。

  **Test-time compute scaling的设计哲学**：
  - 速度→准确率转换：M1的15000+ T/s吞吐量 ≈ 每秒可生成约2个完整的8K推理链。对比R1-1.5B的7200 T/s，**同等时间预算下M1可生成2x+样本或2x+生成长度**。
  - Majority voting场景：32 samples时M1仅需~16秒（32×8K/15K），R1需要~35秒。在16秒时间预算下，R1只能生成~15个样本。**M1用更多样本弥补了单样本quality的微小差距**。
  - 生成长度场景：M1用更长时间生成更长的chain-of-thought → accuracy monotonically增加。同等时间下M1的更长chain-of-thought → 更高的accuracy。

  **论文的局限性（Limitations中承认）**：
  - 3x speedup尚未利用最新的NVIDIA hybrid Mamba kernel（可进一步提升）
  - Attention层未使用vLLM的attention优化（集成后可进一步提升）
  - 未尝试从Qwen2.5-Math模型蒸馏（因该模型的cross entropy loss on OpenMathInstruct太高，需Qwen系列数据）
  - RL训练速度改进已出现（DeepSeek R1），但Mamba架构可进一步加速这一趋势
