## ReXMoE: Reusing Experts with Minimal Overhead in Mixture-of-Experts

- baseline方法是什么？
  - Baseline 是标准 TopK Routing MoE，采用 layer-local routing 机制——每层的 router 仅能从本层的 N 个 expert pool 中选择激活的 experts。核心计算公式：h' = Σ_{i=1}^{N} g_i · E_i(h)，其中 g_i 通过 Softmax(W_gate · h) 后 TopK 选择确定。该架构的根本限制是 expert 维度（每个 expert 的 hidden dimension）受 per-layer 参数预算约束——要在固定总参数量下平衡 expert 数量（粒度）和 expert 容量（hidden dim）。增加 expert 数量（fine-grained MoE）可丰富 expert 组合的灵活性，但减少每个 expert 的 hidden dim 会降低其表达容量；反之保持 expert dim 而增加 expert 数量则膨胀总参数量。这种 trade-off 是 MoE 架构设计的核心矛盾。以 DeepSeek-MoE、Qwen3、Kimi-K2 为代表的 fine-grained MoE 趋势（128-384 experts）选择了"更多更小的 expert"路径，但牺牲了单个 expert 的容量。
  - 全栈执行例子（Baseline: vanilla MoE-2.3BA0.3B，TopK routing，4 nodes × 32 Hopper GPUs）：
    - **算法层**：L 层 MoE Transformer，每层有 N=64 个 experts，每层独立 router W_gate^l ∈ R^{N×d} 执行 Softmax + TopK 选择。第 l 层 router 仅从 E^l = {E_1^l, ..., E_N^l} 中选择——无法访问其他层的 experts。每个 expert 为独立 FFN，参数固定于其所在层。当采用 fine-grained 设计（更多 experts/层）时，受总参数量约束，每个 expert 的 intermediate_size 必须减小，降低单个 expert 的表达能力。
    - **系统框架层**：Megatron-LM 分布式训练框架。Expert Parallelism (EP)=8 将 experts 分布到 8 个 GPU。每层 MoE forward 执行 All-to-All dispatch（token 按 router 结果发送到对应 expert 所在 GPU）→ local expert FFN computation → All-to-All combine（收集 expert 输出）。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明。Megatron-LM 的 MoE kernel 包括 gating（Linear + Softmax + TopK）、All-to-All 通信、per-expert FFN（SwiGLU: W_down @ SiLU(W_gate @ x) ⊙ W_up @ x）。
    - **硬件架构层**：4 nodes × 32 Hopper GPUs。每层 64 experts × ~32 层 = 2048 个 FFN 模块。Expert Reuse 为 0（每层 experts 独立不共享）。
  - Baseline 核心缺陷根因：**Layer-local routing 将 expert 组合的灵活性（expert 数量）与单个 expert 的表达容量（hidden dim）绑定在 per-layer 预算上**——无法在不牺牲 expert 容量或不增加总参数的前提下扩大路由空间。Fine-grained MoE 增加了 expert 数量但降低了每个 expert 的 capacity；DeepSeek-MoE 的 shared expert 缓解了部分问题但仍是 layer-local 方案。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文提出 REXMOE，通过**允许 router 跨相邻层复用 experts**来解耦 expert 维度与 per-layer 参数预算。核心创新：(1) 跨层 Expert Reuse——将 r 个相邻层的 expert pool 合并为扩展候选池 U = ∪_{i∈G} E^i，使每层 router 可访问 r×N 个 experts 但无需增加任何 expert 参数；(2) Progressive Scaling Routing (PSR)——训练期间从 N 逐步扩展至 rN 候选 experts，遵循 curriculum learning 避免训练初期的负载崩溃。
  - 全栈执行例子（REXMOE-R4, MoE-2.3BA0.3B, 4 nodes × 32 Hopper GPUs）：
    - **算法层（解决"layer-local routing 绑定 expert 容量与粒度"的缺陷）**：
      - Cross-layer Expert Reuse：r=4 层连续层分为 group G，每层 router 可访问 4×64=256 个 candidates——但物理上仅存在 64 个 experts（因其来自相邻 4 层，每层 64 个共享）。不需要创建新 experts，不增加任何 expert FFN 参数。仅 router W_gate ∈ R^{rN×d}（256×512）相比 baseline（64×512）增加 4× router 参数，占模型总参数 <1%。
      - PSR 训练策略：t=0~10k steps 仅路由到 N=64 local experts（退化为 baseline）→ t=10k~30k 线性扩展至 256 candidates → t>30k 全量 256 candidates。
      - 结果：Avg Acc 从 49.15% 提升至 50.23%（+1.08 pts），WikiText PPL 从 21.19 降至 20.73。仅 Expert Reuse 贡献 +0.13% acc（边际），PSR 贡献 +1.05%（关键）。
      - 对比 baseline：baseline 中每层独立 64 experts 的路由多样性受限于 layer-local 64 选 TopK；REXMOE 通过跨层复用将每层的路由空间扩大为 256 选 TopK（4× 更多组合），不增加任何 expert FFN 参数。这打破了"更多组合必伴随更小 expert 或更多总参数"的 trade-off——expert 的 hidden dim 保持不变（intermediate=744），但路由组合多样性从 (64 choose TopK) 扩展到 (256 choose TopK)。
    - **系统框架层**：
      - 修改 Megatron-LM 的 MoE Block 和 TopK Router 实现：在每层 MoE forward 时，从相邻 r 层收集 expert 参数引用（不复制）组成扩展候选池。PSR 的 masking 在 gating score 计算后、TopK 选择前执行。
      - 论文未明确说明跨层 expert 参数的 EP 分布变化。推测相邻层的 experts 参数需要通过跨 GPU 通信获取引用，但论文声称 overhead 可忽略。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明。PSR 的 expert masking 是简单的 gating score 置零 + TopK 选择，不涉及新 kernel。
    - **硬件架构层**：同一 4×32 Hopper GPU 集群。核心变化：baseline 每层 router 仅激活本地层的 N 个 experts → REXMOE 每层 router 可激活来自相邻 r 层的 experts，但物理参数不增加，因此 GPU memory 和 compute 中 expert FFN 部分不变。唯一 overhead 来自 router 的 r× 权重增加和跨层 expert 参数引用——论文声称可忽略。
    - **定性分析收益**：Layer-wise expert activation ratio 可视化（Figure 5）显示 REX-SE-R2 相比 Base-MoE-SE 展现出更强的 task-specific specialization——同一 expert 在不同任务（SciQ/LogiQA/WinoGrande）上呈现明显差异化的激活模式，暗示扩展候选池使模型的 expert ensemble 效应对不同任务自适应。
  - 解决 Baseline 缺陷的方式总结：
    1. **针对"layer-local routing 绑定 expert 容量与粒度"**：REXMOE 通过跨层 expert reuse 将路由空间从 N 扩展到 rN，不增加任何 expert FFN 参数——仅增加 <1% 的 router 参数。这直接打破了"更多路由组合 = 更小 expert 或更多参数"的 trade-off。
    2. **针对"训练稳定性（直接跨层 routing 导致负载崩溃）"**：PSR 策略通过 curriculum learning 从 local-only routing 逐步扩展到 full cross-layer routing，使模型在训练早期建立稳定 routing 后逐步适应更丰富的 expert 组合。消融实验（Table 4）证明 PSR 是 critical component：仅 Expert Reuse 仅 +0.13% avg acc，加入 PSR 后 +1.05%。
    3. **针对"大 r 值的负载不均衡崩溃"**：通过 LBV 和 under-utilized experts ratio 的 ablation（Figure 3），论文发现 r=2~4 是最优平衡点——更大的 r（16/32）导致 expert 负载严重不均衡和大量 expert 几乎不被激活的崩溃现象。这为实际部署提供了 r 值选择的指导（r 不宜过大）。
    4. **通用性和兼容性**：REXMOE 不改变 MoE 的核心计算语义（仍是 gating + TopK + FFN），可与 shared experts、fine-grained MoE 等其他设计正交组合。推理部署仅需 vLLM 等框架的 minimal adaptation。
