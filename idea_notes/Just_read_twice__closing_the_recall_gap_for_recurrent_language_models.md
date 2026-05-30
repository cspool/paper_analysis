## Just_read_twice__closing_the_recall_gap_for_recurrent_language_models

- baseline方法是什么？
  Baseline是标准causal decoder-only循环语言模型（Mamba、Based、GLA、Mamba-2等），均以因果自回归方式处理输入——从左到右逐个token处理，使用固定大小的recurrent state存储历史信息。标准ICL格式为Ŷ = A(C, Q)，其中context C在questions Q之前出现。

  Baseline全栈执行例子（Based causal LM推理时生成一个token, 360M/1.3B参数）：
  - 算法pipeline：输入token → embedding → L层Based block（每层交替: gated short convolution(kernel=3) → sliding window attention(window=128) → causal linear attention(Taylor feature map, feature dim=16, 2nd-order: φ(q)^Tφ(k)=1+q^Tk+(q^Tk)²/2 → y_i=φ(q_i)·Σ_{j=1}^{i}φ(k_j)^Tv_j / φ(q_i)·Σ_{j=1}^{i}φ(k_j)) → residual → SwiGLU MLP → residual）→ LM head → logits → next token。每token: linear attention decode O(1)，sliding window attention O(W=128)，recurrent state s ∈ R^{d×d̃}固定大小。Causal cumsum使每个token仅能看到之前的信息——当context长且问题在后面时，模型必须在看到问题前就决定存储什么。
  - 系统框架：PyTorch + FlashAttention训练代码库（https://github.com/Dao-AILab/flash-attention/tree/main）+ LM-Eval Harness推理。开源模型从HuggingFace获取。
  - 编译框架：论文未明确说明。
  - kernel调度：Based Custom CUDA kernel (ThunderKittens)，warp-register分区存储KV-state实现IO-aware prefill。
  - 硬件架构：NVIDIA A100-80GB (训练) / H100 (推理benchmark)，论文未涉及RTL/模拟器层面。

  Baseline (Causal Decoder-only循环LM) 缺陷：
  1. **数据顺序依赖导致ICL脆弱**：Causal模型从左到右处理，若context（如长文档）出现在问题之前，模型必须在未看到问题时就预测应该存储哪些信息。错误的存储决策导致信息丢失，后续无法回忆。例如[D, Q]顺序下模型需记住文档中所有事实，而[Q, D]顺序下只需记住一个。
  2. **内存容量-回忆能力tradeoff**：O(1)推理内存使循环LM在理论上无法记住所有上下文信息（Arvind et al. 2023 proved Ω(N) lower bound for recall）。虽然增加recurrent state size可以提升回忆能力，但硬件效率下降。
  3. **选择机制的局限性**：现有改进（LSMT gates, decay rates, delta rules等）通过架构偏置优化"存储/丢弃"决策，但未利用数据顺序简化选择难度。
  4. **循环LM在recall-intensive ICL任务上显著落后于Transformer**：2.8B Mamba (300B tokens) 比1.3B Transformer (50B tokens) 平均低5个点。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出两种互补方法：(1) JRT-Prompt——重复context使模型看到所有数据顺序；(2) JRT-RNN——非因果Prefix Linear Attention架构+联合训练目标。

  论文方法全栈执行例子：

  **JRT-Prompt** (推理时生成一个token, 与baseline相同架构，仅prompt改变)：
  - 算法pipeline：构造prompt Ŷ = A(C, Q, C, Q)（context重复两次）→ embedding → causal decoding。第二轮出现时模型已condition on完整context（包括前面的Q），此时决定存储什么时能看到全部信息——等价于让模型学到最优存储策略。缺点：context长度翻倍。但sub-quadratic架构使得2N长度仍渐进快于Transformer的N长度。
  - 系统框架：同baseline（使用开源模型权重，无需额外训练）。通过LM-Eval Harness调用。JRT-Prompt需要将原prompt中的context和question各重复一次，对基于HuggingFace的模型interface无额外修改。
  - kernel调度：同baseline Based kernel，但prefill长度由N变为2N（即2× prefill time）。
  - 效果：16个模型×6任务平均+11.0±1.3点提升。N=32768, B=16, H100上11.9×于FA2的prefill吞吐量（因为linear attention的2N仍远快于attention的N）。

  **JRT-RNN** (推理时生成一个token, PLA层)：
  - 算法pipeline：输入token(前M=1024为encoder区域，后N-M为decoder区域) → embedding → L层block（gated convolution/sliding window同Based；PLA层: encoder区域k_e/v_e非因果sum in parallel → decoder区域q_d/k_d/v_d causal cumsum → y_i = φ(q_i)(Σ_{j=1}^{i}k_d[j]^Tv_d[j]+Σ_{j=1}^{M}k_e[j]^Tv_e[j]) / φ(q_i)(Σ_{j=1}^{i}k_d[j]+Σ_{j=1}^{M}k_e[j])）。Pre-fill: 并行计算encoder初始state s_M = Σ_{j=1}^{M}(k_e[j]^Tv_e[j]+k_d[j]^Tv_d[j])。Decoding (i>M): O(1) standard causal linear attention。训练: L = (w1·L_NTP + w2·L_MLM)/(w1+w2)，encoder区域随机mask P比例token计算MLM loss。
  - 系统框架：PyTorch + Based代码库（https://github.com/HazyResearch/based）。训练在FlashAttention代码库上进行。开源权重在HuggingFace。
  - 编译框架：论文未明确说明。
  - kernel调度：扩展Based Custom CUDA kernel (ThunderKittens): fnbased(k_e,v_e)先计算encoder KV-state→寄存器，再fnbased(q_d,k_d,v_d)从该state续算decoder→SRAM→HBM。PLA decode O(1)无额外修改。JRT-RNN CUDA prefill达19.2×于FA2 (N=32768)。
  - 硬件架构：NVIDIA A100-80GB (训练) / H100 (推理benchmark)，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（数据顺序依赖）→ 理论形式化（SD问题+通信复杂度）证明数据顺序决定memory requirement为Ω(min(|A|,|B|))；JRT-Prompt通过重复context使模型看到所有数据顺序；JRT-RNN通过非因果encoder处理prompt使模型可以同时看到全部context信息。
  - 缺陷2（memory-recall tradeoff）→ JRT-Prompt在理论上将memory下界从Ω(max(|A|,|B|))降为Ω(min(|A|,|B|)/p)（p为重复次数）；JRT-RNN的PLA decoder O(1) memory不变但encoder非因果sum让模型充分利用decoder的有限memory（先看到问题和答案再决定存什么）。
  - 缺陷3（选择机制局限）→ 不从修改gate/decay入手，而是通过改变数据呈现方式（JRT-Prompt）或架构causality（JRT-RNN）从根本上降低选择难度。Encoder-decoder分离KV投影让encoder和decoder各自优化不同的信息处理策略。
  - 缺陷4（ICL质量差距）→ JRT-RNN 360M/30B达Transformer++同参数92%的质量（avg 42.9 vs 43.4），1.3B/50B达96%（49.5 vs 51.4）。JRT-Prompt使Based+JRT超越Transformer++ with standard prompting。
  - 训练效率 → JRT-RNN decoder区域标准NTP loss（50%数据量于纯decoder模型），encoder区域MLM loss补偿。PLA decode O(1)与causal LM完全相同。
  - 理论洞察 → BaseConv等纯卷积架构即使有JRT-prompt也无法降低memory下限（Theorem G.6/G.7/G.11），说明JRT方法的效果是架构依赖的——需要linear attention类的关联记忆机制（IP kernel + input-dependent shift）。
