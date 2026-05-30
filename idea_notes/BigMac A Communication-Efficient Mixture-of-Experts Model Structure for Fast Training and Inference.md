## BigMac A Communication-Efficient Mixture-of-Experts Model Structure for Fast Training and Inference

- baseline方法是什么？
  Baseline 为 **Fine-Grained MoE（DeepSeekMoE 式）**，基于 CDAC（communicate-descend-ascend-communicate）执行方式：

  在 CDAC 方式下，每个 MoE 层的执行顺序为：
  1. **Communicate**：将 token 通过 All-to-All 发送到对应 expert 所在设备（此时 token 维度为 full hidden dimension h）
  2. **Descend**：各 expert 内部通过 descending projection ($W_{i,\downarrow}$) 将 token 从 h 降维到 FFN intermediate dimension $h_f$
  3. **Ascend**：各 expert 内部通过 ascending projection ($W_{i,\uparrow}$) 将 token 从 $h_f$ 升维回 h
  4. **Communicate**：再次 All-to-All 将各 expert 输出汇集回源设备

  All-to-All 通信量 C = 2 × top_k × (ep-1)/ep × b × s × h，正比于 hidden dimension h。当 top_k 增大时（如从 top-1 到 top-8），通信量线性增长 7.1-7.3 倍，成为训练和推理的主导延迟（占比高达 90%+）。

  **Baseline 全栈执行例子（以 GPT-Fine-Grained, top_k=8, ep=32 推理一个 MoE layer 为例）**：

  - **算法层**：输入 x ∈ R^{batch×seq×2048} → Router: TopK(Softmax(x·W_gate), k=8) → All-to-All dispatch（高维 2048 通信，1,488 GB transfer for ep=32）→ 64 个 expert 各自执行 E_i(x) = σ(x·W_{i,↓})·W_{i,↑}（每个 expert h=2048→h_f→2048）→ All-to-All combine（高维 2048）→ 输出 y。通信占迭代时间 79.2-90.6%（随 top_k 增大而增加）。
  - **系统框架层**：Megatron-LM EP（expert parallelism）=32，跨 32 GPU 分发 64 experts。All-to-All 使用 NCCL 通信原语，token dispatch 和 combine 各一次全交换。Megatron 中 TP-SP 通信还涉及 All-to-All、All-Gather、Reduce-Scatter（在 TP group 内），均在高维度进行。
  - **编译框架层**：论文未明确说明（NCCL + cuBLAS 标准执行）。
  - **Kernel/运行时调度层**：All-to-All kernel（NCCL）→ Expert FFN GEMM kernel（64 experts 分布在 32 GPU 上，每 GPU 2 experts）→ 高维 token 搬运占通信时间主导。
  - **硬件架构层**：32 GPUs（48 GB HBM each, PCIe 4.0 x16），4 节点 × 8 GPU，100 Gbps InfiniBand。All-to-All 通信成为瓶颈，top_k=8 时占总延迟 91.8%（训练）/ 90.6%（推理）。

  **Baseline 的核心缺陷**：
  1. **All-to-All 在高维度进行**：CDAC 模式下，expert 内部降维-升维投影在 All-to-All 之后，导致通信始终在全维度 h 进行，通信量巨大。
  2. **top_k 增大会加剧通信瓶颈**：fine-grained MoE 需要更多 small experts 和更大 top_k 来保证性能，但通信量与 top_k 线性增长，限制模型扩展。
  3. **系统级优化效果有限**：Tutel 的 overlap 和 Lina 的带宽协调等系统优化在 fine-grained MoE 场景下效果有限——计算量被大幅减少后（small experts），overlap 窗口太小，带宽优化空间被压缩。
  4. **压缩方案损害模型质量**：ScheMoE 的 ZFP 压缩等 lossy 压缩方法会降低模型质量，且引入额外压缩/解压计算开销。
  5. **Expert capacity 限制丢 token**：为缓解 imbalanced routing 带来的 straggler 问题，传统 MoE 设置 expert capacity（capacity factor f=1~1.25），超限 token 被丢弃，直接损害模型质量。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **BigMac**，核心创新是将 fine-grained MoE 的 CDAC 执行顺序改为 **DCCA（descend-communicate-communicate-ascend）**，配合重新设计的 expert 结构：

  **DCCA 策略**：
  1. **Descend**：在 All-to-All 通信之前插入 descending projection ($W'_{\downarrow}$)，将 token 从 h 压缩到 r·h（r=0.25，如 2048→512）
  2. **Communicate**：All-to-All dispatch —— 在低维度 r·h 进行
  3. **Communicate**：All-to-All combine —— 继续在低维度 r·h 进行
  4. **Ascend**：ascending projection ($W'_{\uparrow}$) 将 token 从 r·h 恢复到 h

  通信量降至原来的 r 倍（-75%），仅增加 +4.54% FLOPs 和 +1.35% 参数。

  **BigMac Expert 重新设计**：由于 DCCA 已将输入维度降至 r·h，expert 内部改为先升维再降维（$E_i(x) = \sigma(xW_{i,\uparrow})W_{i,\downarrow}$），保证与 fine-grained MoE expert 相同的参数量和计算复杂度。

  **Defect → Design 映射**：

  | Baseline 缺陷 | BigMac 设计选择 | 解决机制 |
  |---|---|---|
  | All-to-All 在高维度进行 | DCCA：先 descend 再 communicate | 通信维度从 h 降至 r·h（-75%), All-to-All 通信量减少至 25% |
  | top_k 增大会加剧通信瓶颈 | 通信量与 top_k 解耦（DCCA 框架下 top_k 的影响被 r 打折扣） | Top8 BigMac 比 Top4 Fine-Grained 快 27.7-55.4% |
  | 系统优化在 fine-grained 场景效力有限 | 算法层直接减少通信量（不依赖系统层 overlap） | Megatron 上 2.45-3.07× training speedup；Tutel 上 1.71-3.09× speedup |
  | 压缩方案损害模型质量 | DCCA 是结构级优化（非 lossy compression），无精度损失 | BigMac 质量与 Fine-Grained 相当或更优（Table 6,7） |
  | expert capacity 丢 token | 通信减少后可移除 capacity 限制实现 dropless routing | 进一步改善模型质量（enables dropless token routing） |

  **BigMac 全栈执行例子（以 GPT-BigMac, top_k=8, r=0.25, ep=32 推理一个 MoE layer 为例）**：

  - **算法层**：输入 x ∈ R^{batch×seq×2048} → Router: TopK(Softmax(x·W_gate), k=8)（在 full dimension 2048 做路由以保证精度）→ Descend: x' = x·W'↓（2048→512, 压缩至 r=0.25）→ All-to-All dispatch（低维 512 通信，仅 372 GB transfer，-75% vs baseline）→ 64 experts 各自执行 E_i(x') = σ(x'·W_{i,↑})·W_{i,↓}（先在 expert 内升维 512→h_f→512，保证 expert 复杂度）→ All-to-All combine（低维 512）→ Ascend: y = y'·W'↑（512→2048, 恢复原始维度）→ 输出。额外优势：可承受更大 top_k（如 Top8）而无通信代价惩罚；可移除 expert capacity 限制实现 dropless routing。

  - **系统框架层**：Megatron-LM / Tutel / DeepSpeed-Inference 无需修改系统逻辑——DCCA 仅改变模型结构（projection 顺序），通信调用方式不变。在 Megatron 中，TP-SP 通信（All-to-All、All-Gather、Reduce-Scatter）也从高维降为低维，减少 1.42-2.34×。

  - **编译框架层**：论文未明确说明。

  - **Kernel/运行时调度层**：All-to-All kernel 搬运的 token 维度从 2048 降至 512（-75% data volume）。Expert FFN 计算量不变（通过先升维保证），但通信 kernel 耗时大幅缩短。Tutel 的 2DH All-to-All + overlap 与 BigMac 正交叠加。

  - **硬件架构层**：32 GPUs × 4 nodes, 100 Gbps InfiniBand。BigMac 将 All-to-All 通信瓶颈从占 91.8% 显著降低，端到端训练延迟加速 1.53-3.09×（跨 Megatron/Tutel），推理吞吐提升 1.62-3.11×（跨 Megatron/Tutel/DeepSpeed-Inference）。

  **关键设计对应关系**：

  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | DCCA（descend before communicate） | 通信在高维度 | All-to-All 从 1,488 GB → 372 GB (-75%) |
  | BigMac Expert（先升后降） | DCCA 降维后 expert 参数减少 | Param: 3.73B → 3.78B (+1.35% only) |
  | r=0.25 downscaling factor | 平衡通信节省与模型质量 | FLOPs +4.54% 换取通信 -75% |
  | Routing at full dimension | 保证门控精度（低维路由会退化） | 选择 x（非 x'）作为 gate input |
  | Dropless routing | 通信减少后移除 capacity 限制 | 进一步提升模型质量（避免丢 token 损失） |

  **创新总结**：BigMac 的核心洞察是——fine-grained MoE 的 CDAC 方式将 All-to-All 放在了最高维度（通信最贵），通过重新排列 projection 和通信的顺序（DCCA），只需增加极少的 FLOPs（+4.54%）和参数（+1.35%），即可将通信量减少 75%。这是一个纯算法/模型结构层面的优化，与系统级优化（Tutel、Lina 等）正交叠加。更重要的是，BigMac 解耦了 top_k 与通信成本的关系，使 fine-grained MoE 可以使用更大的 top_k 以获得更好的模型质量，而无需承受通信代价。
