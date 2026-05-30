## AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation for Efficient Neural Machine Translation

- baseline方法是什么？
  **Baseline 为手动设计的 homogeneous MoE（以 Switch Transformer 为代表）**。在传统 MoE 中，专家采用均匀设计：所有层中 expert 数量相同（如每层 4 个或每隔一层 4 个），所有 expert 的 FFN 尺寸相同（如 intermediate size = 2048 或 3072），encoder 和 decoder 的层数也固定（均为 6 层）。专家放置位置采用 ad-hoc 规则：每隔一层（Fedus et al. 2022b; Kim et al. 2021）、每四层（Zoph et al. 2022），或最后几层（Rajbhandari et al. 2022）。

  **Baseline 全栈执行例子（以 6-layer encoder-decoder SwitchTransformer-Big + 手动 homogeneous MoE 推理一个 token 为例）**：
  - **算法层**: token embedding x → Router: Softmax(x · W_g) → top-1 gating → 每层固定激活 1/4 experts（所有 expert FFN 尺寸相同，intermediate=3072）→ 所有 token 走相同大小的专家计算
  - **系统框架层**: fairseq (PyTorch) → encoder forward（6 layers, 4 experts each）→ decoder autoregressive forward（6 layers, 4 experts each, 200× 每步时间 vs encoder at peak throughput）→ 所有 decoder layer 每步激活相同数量 expert
  - **编译框架层**: 论文未明确说明（PyTorch eager execution + fairseq）
  - **Kernel/运行时调度层**: 论文未明确说明（标准 cuBLAS GEMM 执行 expert FFN）
  - **硬件架构层**: Intel Xeon CPU → decoder latency 占总量 90%+ → 6 decoder layers × 4 experts 激活 → total FLOPs 10.6G, latency ~2199ms (CPU)

  **Baseline 的核心缺陷**：
  1. **Homogeneous 设计导致计算浪费**：所有 expert 尺寸相同，但不同 token 需要不同计算量——对简单 token，"大专家"浪费 FLOPs
  2. **无自适应计算（adaptive compute）**：相同数量/大小的 expert 参数应用到每个输入，不支持"不同 token 使用不同计算量"
  3. **手动设计效率低下**：expert 放置（每层/隔层/每四层）是 ad-hoc 选择，未系统性地优化 FLOPs 和 latency
  4. **MoE 设计不考虑硬件约束**：expert 数量/大小的选择与目标部署硬件（CPU latency, memory）脱节，模型可能在 CPU 上 latency 过高

- 论文方法是什么？如何对应解决Baseline的缺陷？
  
  **论文方法**: AutoMoE = 通过 NAS 在异构 MoE 搜索空间中自动搜索最优架构。核心设计三步：
  1. **异构搜索空间**：每层可变 expert 数量（{1,...,M}）+ 每 expert 可变 FFN 尺寸（{1024,2048,3072}）+ 可变 decoder 层数（{1-6}）→ 形成指数级搜索空间
  2. **Supernet 训练 + 演化搜索**：Supernet 通过 weight sharing 联合训练所有子架构，演化算法以 validation loss 为性能信号、以目标设备 latency 为约束，迭代搜索 Pareto 最优
  3. **自适应计算（Adaptive Compute）**：异构设计使不同 token 通过 routing 自然分配到不同大小的 expert，简单 token 走小 expert（节省计算），复杂 token 走大 expert（保持质量）

  **Defect→Design 映射**:

  | Baseline 缺陷 | AutoMoE 设计选择 | 解决机制 |
  |---|---|---|
  | Homogeneous expert size → 所有 token 相同计算量 | 可变 expert FFN 尺寸（per expert） | Token 路由至不同大小的 expert，"简单" token 走小 expert（节省 FLOPs），"困难" token 走大 expert（保证质量），实现 adaptive compute |
  | 手动 ad-hoc expert 放置 | NAS 自动搜索每层 expert 数和放置 | 演化算法发现最优配置：encoder 中间层（3rd, 5th）分配最多 expert，decoder 首层最多 → encoder 承担 71% 专家 |
  | 不考虑硬件约束 | Latency constraint（CPU ≤ 600ms）作为搜索约束 | 演化搜索在 latency 约束内优化 BLEU，产生的架构天然满足部署硬件要求 |
  | 固定 decoder 层数（6 层）→ decoder 延迟主导 | 搜索可变 decoder 层数（1-6） | 减少 decoder 层数（从 6 → 3 或 4），补偿为增加首层 expert 数。decoder latency 降低 30%+ |
  | Expert 选择缺乏系统性优化 | Supernet weight sharing + 演化搜索 | 联合优化 expert 数量、大小、decoder 层数、attention heads、hidden size 等全部 Transformer 超参数 |

  **AutoMoE 方法全栈执行例子（以 WMT'14 En-De AutoMoE 6-expert 搜索到的架构推理一个 token 为例）**：
  - **算法层**: token x → Router → top-1 gating → encoder: 层层 expert 数分别为 [5,1,1,1,2,1]，expert FFN 尺寸各异（1024-3072）；decoder: 4 layers, experts [1,1,1,1], FFN 尺寸全 3072 → 大多数计算集中在 encoder 中间层（容量大），decoder 轻量化（layer 数从 6 → 4, experts 少）
  - **系统框架层**: fairseq (PyTorch) → encoder forward: 中间层激活多 expert + 大 FFN（处理源语言语义信息）→ decoder forward: 4 layers only, 每层 1 expert（轻量级生成）→ 总 latency 504ms (CPU) vs baseline SwitchTransformer 2199ms
  - **编译框架层**: 论文未明确说明（标准 PyTorch eager execution + fairseq）
  - **Kernel/运行时调度层**: 论文未明确说明。总 FLOPs 从 10.6G → 2.9G (↓3.7×)，expert 激活数大幅减少
  - **硬件架构层**: Intel Xeon CPU → encoder 承担主要计算（中间层多 expert + 大 FFN, latency ~45ms），decoder 极轻（4 layers × 1 expert × 3072 FFN, latency ~459ms）→ 总 latency 504ms = 4.4× speedup

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | 可变 decoder 层数 | decoder 延迟主导（>90%） | FLOPs 随 decoder 层数增加而增加（Fig 3a）；AutoMoE 自动选择 3-4 decoder layers |
  | Encoder 中间层多 expert | Encoder 需要高容量处理语义 | Encoder 3rd/5th layer 分配最多 expert（Fig 3c），encoder 占总 expert 71% |
  | Decoder 首层多 expert | 补偿 decoder 层数减少的容量损失 | Decoder 首层 expert 最多，逐层递减（Fig 3d） |
  | 异质 expert 尺寸（fract-expert） | 实现 adaptive compute | 70% expert layers 有 ≥2 experts，>75% 含可变 expert 尺寸。WMT'14 En-De AutoMoE: BLEU 28.2, FLOPs 2.9G, Latency 504ms |
  | Identity/dummy experts（FFN size=0） | 允许部分 token "跳过" FFN 计算 | BLEU 28.1 (↓0.1), FLOPs 2.7G (↓6.9%) — 质量轻微损失但 FLOPs 显著降低 |
  | Latency constraint（而非仅 FLOPs） | 更严格的硬件控制 | Latency constraint 下模型充分利用 budget 且 FLOPs 更优；FLOPs constraint 下 latency 偏高（Table 6） |

  **创新总结**: AutoMoE 首次将 NAS 引入 MoE 设计，将 MoE 架构从"手动 homogeneous 设计"转变为"自动异构搜索"。其核心洞察是：MoE 架构的各维度（expert 数量、大小、decoder 层数）之间存在复杂的性能-效率 trade-off，通过 Supernet 的 weight sharing 和演化搜索，可以在短时间内（224 GPU-hours vs Evolved Transformer 的 2,192,000 GPU-hours）找到 Pareto 最优的异质配置。异构设计自然实现 adaptive compute——不同 token 路由到不同大小的 expert，无需额外机制即可实现"按需分配计算"。
