## Continual Pre-training of MoEs: How robust is your router?

- baseline方法是什么？
  **Baseline 为 Dense Transformer 的持续预训练策略（直接应用于 MoE）**：
  
  从 Ibrahim et al. (2024) 确立的 Dense CPT 策略出发——LR Re-warming + LR Re-decaying + Replay——直接应用于 MoE，不针对 MoE 特有的路由机制做任何修改。具体配置：(1) 从衰减后 checkpoint 开始 CPT，用 cosine decay 重新 warmup 到 $\eta_{max}$ 再 decay；(2) 按固定 replay 比例混合新旧数据。

  **Baseline 全栈执行例子（以 PB Granular MoE, 400B FineWeb → 200B German, decayed checkpoint CPT, 一个训练 step 为例）**：
  
  - **算法层**：输入 1024 sequences × 2048 tokens → 24 层 MoE decoder → 每层：Attention → Router (linear proj [1024, 32] softmax top-3) → 3 细粒度 GEGLU FFN experts + 1 shared expert → weighted sum → Aux Loss + Z-Loss 加入总 loss → next layer
  - **系统框架层**：GPT-NeoX + Megablocks grouped GEMM kernel → 64 A100 GPU Dataparallel + ZeRO-1 → dropless MoE 前向（Megablocks 处理稀疏 expert dispatch）→ AdamW optimizer step
  - **编译框架层**：论文未明确说明（PyTorch eager execution + GPT-NeoX 框架）
  - **Kernel/运行时调度层**：Megablocks grouped GEMM kernel 处理 MoE 稀疏前向（group tokens by expert → batched GEMM）→ NCCL All-Reduce for Data Parallel → ZeRO-1 分散 optimizer states
  - **硬件架构层**：64× A100 GPU，标准数据中心配置，无特殊硬件

  **Baseline 的核心缺陷**：
  1. **路由算法对分布偏移的鲁棒性未知**：MoE 的 PBTk/SBTk 路由是在 IID 预训练数据上设计的，分布偏移可能导致 router 在新旧分布间失衡，加剧遗忘或破坏 expert 负载均衡
  2. **LR Re-warming 对路由的影响未知**：从衰减 checkpoint 大幅 warmup LR 可能导致 router 经历"混沌期"，token 分配剧烈波动
  3. **Replay 对 MoE 路由的影响未知**：replay 旧数据可能干扰 router 对新分布的适应，也可能帮助维持旧分布上的负载均衡
  4. **缺乏 CPT 场景下的路由诊断工具**：没有类似 MRI 的指标来量化分布偏移对 MoE 最坏情况延迟的影响
  5. **MoE CPT 与 Full Re-training 的性能差距未知**：不知道 CPT MoE 能否匹配重训练 MoE 的性能

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：系统性实证研究，不提出新算法，而是(1) 验证现有 Dense CPT 策略（Infinite LR + Replay）对 MoE 的适用性；(2) 量化两种路由算法和两种 MoE 架构在 CPT 下的鲁棒性；(3) 提出 MRI 指标和三种路由行为分析指标来诊断 CPT 中的路由变化；(4) 将 CPT MoE 与 Full Re-training MoE 直接对比。
  
  **Defect → Design/Discovery 映射**：

  | Baseline 缺陷 | 论文方法/发现 | 解决机制 |
  |---|---|---|
  | 路由对分布偏移鲁棒性未知 | 实验证明 PBTk/SBTk 均对分布偏移**惊人鲁棒**——即使 0% replay，MRI 也在 500 step 内恢复到比 SBTk 更好的水平 | PBTk 的 Aux Loss + Z-Loss 足以在 CPT 中维持负载均衡；SBTk 显式平衡更稳定但最终 MRI 更高 |
  | LR Re-warming 对路由影响未知 | 对比 decayed vs non-decayed checkpoint CPT：Non-decayed (CosineInf) 减少遗忘且不牺牲适应，路由混沌期更短 | CosineInf schedule 在 CPT 阶段不 warmup，直接从 $\eta_{const}$ 继续，避免剧烈 LR 变化对 router 的冲击 |
  | Replay 对 MoE 路由影响未知 | Replay 对 MoE 和 Dense 的效果相似：减少遗忘但稍损害适应。对 PBTk 的 MRI spike 有轻微缓解作用 | Replay 保持 router 对旧分布的记忆 → 减少早期层路由变化 → 减少遗忘 |
  | 缺乏 CPT 路由诊断工具 | 提出 **MRI**（Maximum Routing Imbalance）作为最坏情况延迟代理；扩展三个路由行为指标（Router Saturation / Vocabulary Specialization / Expert Co-activation）到 CPT 场景 | MRI = $\max_i$ ( routed tokens to expert i / total tokens)，直接量化延迟风险 |
  | 不知道 CPT MoE 能否匹配 Full Re-training | CPT PB Granular MoE 在 German 和 Code 任务上匹配甚至超越 Full Re-training MoE 性能（<1% accuracy 差距），同时仅用约 1/3 计算量 | MoE 的更大参数量在 CPT 中起正则化作用 → 比 Dense 忘记更少 |
  | Switch MoE 早期层 MRI 不稳定 | 发现 Granular MoE 架构在早期层 MRI 更稳定 → 推荐 CPT 使用 Granular MoE | 细粒度 experts + shared expert 提供更稳定的路由分布 |

  **论文方法全栈执行例子（以 PB Granular MoE, CosineInf + 40% Replay, 400B FineWeb → 200B German CPT, 一个训练 step 为例）**：
  
  - **算法层**：
    1. Data Sampling: 1024 sequences × 2048 tokens，其中 40% (410 seqs) 从 FineWeb replay，60% (614 seqs) 从 German CC
    2. 24 层 decoder 前向：每层 Attention → Router (W_r [1024, 32] · x → softmax → top-3 experts from 31 routed + 1 shared) → 3 细粒度 GEGLU FFN (intermediate=704) + 1 shared GEGLU FFN → weighted combination → 输出
    3. Loss = CrossEntropy(LM) + 0.01×Aux Loss + 0.001×Z-Loss（Aux Loss 鼓励 31 experts 均匀负载，Z-Loss 惩罚大 router logits）
    4. Backward: AdamW (β1=0.9, β2=0.95, wd=0.1, grad clip=1.0), LR=1.65×10^{-4} (CosineInf constant phase)
    5. MRI 记录：per-layer max(load_i / total_tokens)，monitor 最坏情况延迟风险
  
  - **系统框架层**：
    - GPT-NeoX 框架 + Llama3 tokenizer (128K vocab)
    - Megablocks grouped GEMM kernel：将分配到同一 expert 的 tokens 分组 → batched GEMM → dropless 执行
    - 64 A100 GPU：Data Parallel + ZeRO-1 (optimizer states 分片)

  - **编译框架层**：论文未明确说明（PyTorch eager + Megablocks kernel 编译为 CUDA）
  
  - **Kernel/运行时调度层**：
    - Megablocks grouped GEMM：输入 [S, H] tokens → Router → token-to-expert mapping → group tokens by expert → per-expert GEMM (batched) → scatter output to original token positions
    - ZeRO-1 All-Reduce：gradient synchronization across 64 GPUs → optimizer step per GPU → broadcast updated params
    - 关键：Granular MoE 的 forward pass (485ms) 比 Switch MoE (449ms) 慢约 8%，backward 慢约 13%（更多 experts 的 dispatch overhead）
  
  - **硬件架构层**：
    - 64× A100 GPU，标准配置。论文未明确说明互联方式（NVLink/PCIe）或显存容量
    - MoE step time ~1679ms (PB Granular) vs Dense ~880ms → MoE 约 2× slower per step，但因样本效率优势总体更强

  **关键发现与设计指导**：
  | 发现 | 对 CPT MoE 的指导 |
  |---|---|
  | CosineInf (non-decayed) 优于 Cosine Decay (decayed) CPT | CPT 应在预训练阶段就使用 Infinite LR Schedule |
  | Replay 对 MoE 和 Dense 效果相似 | 按 Dense CPT 经验选择 Replay 比例即可 |
  | PBTk 性能始终优于 SBTk（首次在大规模实验中证实） | CPT 优先使用 PBTk (Aux Loss + Z-Loss) 路由 |
  | Granular MoE 优于 Switch MoE，且早期层 MRI 更稳定 | CPT 优先使用细粒度 expert 架构 |
  | 早期 MoE 层（0-2）路由变化最大 → 与遗忘最相关 | 未来可研究对早期层特殊处理（freeze/lower LR）以减少遗忘 |
  | CPT MoE 匹配 Full Re-training MoE，仅需 ~1/3 计算量 | CPT 是 Full Re-training 的高效替代方案 |
  | MRI 在 PBTk 下分布偏移后短暂飙升但 500 step 内恢复 | CPT 中短期 MRI spike 无需特别干预 |
