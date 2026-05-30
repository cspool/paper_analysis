## Read-ME: Refactorizing LLMs as Router-Decoupled Mixture of Experts with System Co-Design

- baseline方法是什么？
  - Baseline 有两类：(1) **Open-source dense models**（Pythia 2.8B/6.9B, Open-Llama-v2 3.4B/6.9B）和 **dense compression methods**（Sheared-Llama 2.7B, LLM-Pruner, SliceGPT, LaCo, Compresso），它们将大模型压缩为更小的 dense 模型，完全依赖 dense FFN 计算，无法利用 MoE 的稀疏激活优势；(2) **传统 layerwise MoE**（Mixtral-8×7B, OpenMoE），每层配备独立 router G^(l)，基于第 l-1 层 hidden states 动态决定第 l 层的 expert 选择，导致系统层两个根本问题：Memory Management——无法预知未来层所需 expert，只能依赖 naive prefetching（如前一层 hidden states 预测下一层 expert + LRU cache）或 on-demand loading（在关键推理路径上增加加载延迟）；Token Batching——每层每个 token 可能选择不同 expert，导致一个 batch 内需激活大量 unique expert（Mixtral-8×7B 在 batch_size=56.8 时平均激活 7.63/8 experts），batch 内 token 必须等待所有 expert 计算完成，batching 效率退化至接近无 batching。
  - 全栈执行例子（Baseline: Mixtral-8×7B layerwise MoE, 单卡 A100-80GB, Chatbot Arena workload）：
    - **算法层**：Mixtral 采用 standard layerwise top-K routing（N=8, K=2），每层 router 为 linear layer W_r ∈ R^{d_model×N}，激活函数 softmax。第 l 层输出：y = Σ_{i=1}^N I(top-K G^{(l)}(x)) · G^{(l)}(x)_i · F_i^{(l)}(x)。Router 参数 32 layers × (4096×8) ≈ 1M params。
    - **系统框架层**：HuggingFace Transformers / vLLM 等标准推理框架。Tokens 组成 batch，每层对所有 expert 的 FFN 按 selected experts scatter-gather：scatter tokens 到对应 expert → expert FFN 计算 → gather 结果按 gating weights 加权求和。Token 间存在 implicit barrier——所有 expert 计算完成后才能进入下一层。
    - **编译框架层**：论文未明确说明。使用 PyTorch JIT / torch.compile。
    - **kernel调度层**：标准 cuBLAS GEMM kernel。Expert FFN 计算为：gate_proj(x) → SiLU(gate) ⊙ up_proj(x) → down_proj(result)。每个 activated expert 执行一次 GEMM。Batch 内不同 token 分散到不同 expert 导致多个小 GEMM（非合并大 GEMM），kernel launch overhead 和低 GPU 占用率。Expert weights 保持在 GPU memory 或通过 LRU cache + speculative prefetching 部分加载。
    - **硬件架构层**：单卡 A100-80GB。Expert weights 以 FP16 存储于 GPU HBM（~94GB for 8 experts × 32 layers）。若 GPU memory 不足 → offload 未使用 expert 到 host memory（PCIe 4.0 ~25 GB/s），推理时按需加载 → 加载延迟在关键路径上。
  - Baseline 核心缺陷：(1) **逐层 Router 冗余**——相邻层 expert 选择高度相关（transition matrix 稀疏，MI 高），独立 router 浪费参数且阻止系统预调度；(2) **Expert 预取不可靠**——基于前一层 hidden states 预测下一层 expert，假设不成立时导致 cache miss penalty；(3) **Batching 低效**——batch 内 unique expert 过多（~7.63/8），token 间同步等待主导延迟；(4) **Cache 策略次优**——LRU 基于单请求 temporal locality，跨请求共享 cache 时失效。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：Read-ME 通过 **Router-Decoupled MoE Architecture + System Co-Design** 两大创新系统性解决 baseline 缺陷：(1) **架构层**：将 dense LLM refactorize 为 MoE 并采用跨层共享的 pre-gating router，消除逐层 router 冗余，使 expert 选择可在推理前一次性确定；(2) **系统层**：利用 pre-gating 先验信息实现 expert-aware batching、fine-grained prefetching 和 Belady 最优缓存。
  - 全栈执行例子（Read-ME 4.7B-17B, 单卡 A100-80GB, Chatbot Arena workload）：
    - **算法层（解决"逐层 Router 冗余"和"Dense 模型效率低"的缺陷）**：
      - **Domain-Aware Expert Construction**：利用 activation sparsity 从 Llama2-7B-chat 构建 8 个 domain expert + 1 permanent expert。对每个子域数据 D_i，选择 top-d 激活通道（d≈D/2），通过结构化 pruning 初始化 expert F_i(x) = W_2 M_i^T σ(M_i W_1 x)，选择矩阵 M_i 最大化 E[||M_i W_1 x||_1]。
      - **Pre-Gating Router（核心架构创新）**：用单一 1-layer Transformer block（18M params）替代 32 个逐层 router。Router G 以 causal attention 处理 x_{≤t}，输出跨所有层统一的 expert 选择：y_t = Σ_{i=1}^N I(top-K G(x_{≤t})) · G(x_{≤t})_i · F_i^{(l)}(x_t)。Router 与 layer index l 无关——同一 token 在所有层选择相同的 expert。
      - **对比 baseline**：Mixtral 每层独立 router → 32 个无关 routing path；Read-ME 单一 router → 专家选择跨层一致，消除路由歧义。仅 18M router params vs Mixtral ~1M（但 Read-ME router 贡献 0.4% 延迟 vs Mixtral 3.95%）。
      - **Routing Distillation Loss**：L_RD = KL(softmax(G) || softmax([||M_0 M_1^T||_F^2, ...]))，利用 dense 模型激活稀疏性指导 router 学习，加速收敛。
      - **关键结果**：仅 1.04B training tokens 达到 MMLU 38.9%（Sheared-Llama 50B tokens 仅 26.4%），平均 accuracy 55.5% 超越同规模所有 baseline。MoE 结构（4.7B activated）比同等大小 dense 模型 MMLU +11.8%。
    - **系统框架层（解决"Expert 预取不可靠"和"Batching 低效"的缺陷）**：
      - **Expert-aware Batching**：修改 DeepSpeed inference engine。Pre-gating 后，Scheduler 按 ReqQueueByExpert 收集选择同一 expert 的 tokens → 组 batch 确保 batch 内所有 token 共享同一 expert。Algorithm 1 从请求最多的 expert 开始取 tokens。
      - **Fine-grained Prefetching**：利用 pre-gating 预知所有层 expert 需求 → compute stream（第 i 层 FFN）与 loading stream（第 i+1 层 expert 传输）流水线重叠，隐藏 PCIe 加载延迟。
      - **Belady-inspired Optimal Caching**：因 pre-gating 预知所有 future expert references → 可精确计算 F(e,t)（expert e 的下次访问时间）→ 实施 Belady 最优驱逐策略 (evict argmax F(e,t))。Cache 跨所有并发请求共享。
      - **对比 baseline**：Baseline 基于前一层 hidden states "猜测"下一层 expert + LRU cache → 跨请求时 LRU 对 temporal locality 假设失效（Table 4: LRU hit ratio 66.95% vs Belady 77.21% at capacity=4）。Baseline batching 平均 5.08-5.21 unique experts/batch → Read-ME 3.51 experts/batch。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准 cuBLAS GEMM kernel。与 baseline 相同使用 SwiGLU FFN 计算（gate_proj + up_proj → SiLU(gate)⊙up → down_proj）。但因 expert-aware batching 减少 batch 内 unique experts → 增大有效 batch size → 更高 GPU 占用率。Prefetching 将 expert 加载与计算重叠 → 加载延迟不进入关键路径。
    - **硬件架构层**：单卡 A100-80GB。与 baseline 相同的物理硬件。Read-ME 通过 pre-gating + Belady cache + prefetch 使同硬件下的端到端平均延迟降低 5.0-6.1%，p95 延迟降低 9.5-10.0%。Prefetching 模式在 cache capacity 受限时比 On-demand Loading 延迟低最多 30%。Router 计算开销仅 0.4% vs 传统 MoE router 3.95%。
  - **方法核心 insight**：将路由决策从"每层交互式"变为"推理前一次性"，使得系统可以在推理开始前完全掌握 token-to-expert 的全路径映射，从而将 expert 调度从"反应式"（reactive）升级为"前瞻式"（lookahead），同时通过算法-系统 co-design 将 MoE 的稀疏性优势从算法层贯穿至系统层。
