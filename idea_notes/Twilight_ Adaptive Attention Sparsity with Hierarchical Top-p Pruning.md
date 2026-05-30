## Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning

- baseline方法是什么？
  Baseline是现有top-k sparse attention方法（Quest、Double Sparsity (DS)、H2O等），在LLM decoding阶段使用固定budget B选择top-k个"critical tokens"参与attention计算，以节省KV cache访问带宽。Baseline全栈执行例子（以Quest在A100上decode step为例）：

  - 算法层：Quest使用per-page (16 tokens/page) max-pooling估计token重要性——对K cache做max_pool得到K_pooled → 计算q @ K_pooled^T得approximate scores → 选择top-k pages（k=B/16）作为critical tokens的索引集合I。

  - 系统框架层：Quest kernel (CUDA/Triton) 直接替换PyTorch attention实现，可集成至vLLM/SGLang等serving框架。

  - 编译框架层：论文未明确说明。

  - Kernel调度层：Quest kernel执行q与pooled K的GEMV → topk selection → sparse FlashAttention仅对|I|=B个token计算精确attention。固定B值导致：(a) 对focused attention heads（权重集中于少数token），B过大→over-selection，加载和计算了不需要的token；(b) 对diffuse attention heads（权重均匀分布），B过小→under-selection，丢失重要context信息。从图2可见Quest/DS在不同budget(256→8192)下perplexity变化剧烈，说明B是高度敏感的hyperparameter。

  - 硬件架构层：NVIDIA A100 GPU。固定budget下，无论head是否需要，均加载相同数量的KV cache token，导致memory bandwidth浪费（over-selection heads）或accuracy损失（under-selection heads）。

  Baseline核心缺陷：**固定的token budget B无法适应不同attention head、不同layer、不同query、不同prompt下attention weight分布的动态性**。根本原因：top-k关心"选多少个"（|I|=B），但attention的数学目标是"累积足够的attention weight"（ΣW[i]≥p）。在focused分布下很小的B就能满足ΣW[i]≥p（B再大就是浪费），在diffuse分布下很大的B仍不够（精度不足）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Twilight，核心洞察：**将LLM text generation中的top-p (nucleus) sampling引入sparse attention**。top-p的本质是从"固定数量"（k个token）转为"固定累积概率"（p=Σattention weights），天然适应不同attention weight分布。

  **(1) Top-p Sparse Attention**——解决"固定budget无法适应分布动态性"缺陷：
  从数学上，attention近似误差界为‖o - ô‖ ≤ (1 - Σ_{i∈I} W[i]) · ‖V‖_F（Eq.2），因此最优策略是最小化|I|使ΣW[i]≥p——这正是top-p的定义（Definition 3.3）。对focused attention（peaked distribution），top-p自动选择很少的token（如B=97 for p=0.8，图4）；对diffuse attention（flat distribution），top-p自动选择更多token。B由分布决定而非预先固定。

  **(2) Hierarchical Select-then-Prune Architecture**——解决"如何将top-p应用于任意现有算法"问题：
  不是重新设计sparse attention算法，而是作为现有算法的**optimizer**。两层架构：(a) Token Selector——将现有算法（Quest/DS等）作为黑盒，使用保守的大budget B0≈N/4预选token子集（保证高recall，避免遗漏重要token）；(b) Twilight Pruner——在子集上用INT4 SpGEMV估计attention weights，然后top-p binary search精筛到B1<<B0。这种hierarchical design使Twilight可适配任何top-k sparse attention算法。

  **(3) INT4 K Cache + Efficient Kernels**——解决"top-p的精度要求高于top-k"的系统开销：
  top-k只需要序数正确（ordinality），top-p需要数值准确性（numerical accuracy）。实验发现4-bit是sweet spot（图6：2-bit累积attention weights显著下降，4/8-bit稳定）。基于FlashInfer实现：(a) INT4 SpGEMV——cp.async + 2-stage pipeline, memory access降至1/4；(b) Top-p binary search——O(log(range/ε))次并行reduction，避免O(N log N)排序；(c) Head-wise varlen attention——flatten head dim + load balancing处理不同head的不同budget。

  全栈执行对比baseline（以Quest-Twi, LLaMA-3.1-8B-Instruct, decode step, 32k context为例）：

  - 算法层：Twilight叠加于Quest之上。Token Selector: Quest用保守budget B0=8192（1/4 sparsity）预选token → Twilight Pruner: INT4 SpGEMV估计attention weights → softmax归一化 → top-p binary search (p=0.95) 精筛到B1≈446 → Sparse Attention仅对B1 token计算精确attention。从"固定8192个token"变为"自适应~446个token"，累积概率仍≥95%。

  - 系统框架层：基于FlashInfer构建的kernel library。支持PagedAttention，可集成至vLLM/SGLang。额外INT4 K cache内存开销为1/8 FP16 KV cache（可复用base algorithm已有的INT4 K cache）。

  - 编译框架层：论文未明确说明编译框架。使用CUDA/Triton直接编写kernel。

  - Kernel调度层：三阶段pipeline——(a) Quest Token Selector (SpGEMV on FP16 K cache, ~15% time)；(b) Twilight Pruner: INT4 SpGEMV + Softmax + Top-p Binary Search (~20% time)；(c) Sparse Attention: 仅对B1 token做FlashAttention (~65% time)。GQA下每query group取各head选择token的union后flatten head dim做load balancing。Quest-Twi vs Quest: 1.4× self-attention speedup, 1.35× end-to-end decoding speedup。up to 15.8× vs FlashAttention2。

  - 硬件架构层：单张A100 GPU。与baseline相比：Pruner引入额外INT4 SpGEMV开销，但Sparse Attention的token数从8192降至~446（18×减少），总时间减少（Quest 8192 budget vs Quest-Twi auto ~446 budget）。

  设计思路核心：论文的key insight是类比LLM token sampling中的top-k vs top-p问题。在text generation中，top-p (nucleus sampling) 替代top-k解决了"不同下一个词分布需要不同k"的问题；同理，在sparse attention中top-p替代top-k解决了"不同attention weight分布需要不同budget"的问题。Twilight的创新不在于提出全新的token selection策略，而在于认识到**budget selection本身应该是算法的一部分——且应该由sum of attention weights而非固定count来决定**。Hierarchical architecture使得这个insight可以作为任何现有top-k方法的"drop-in optimizer"叠加使用，而非重新设计整个sparse attention pipeline。
