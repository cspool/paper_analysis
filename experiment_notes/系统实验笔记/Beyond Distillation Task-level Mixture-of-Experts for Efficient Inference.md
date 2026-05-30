## Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **Task-level MoE Sub-Network Extraction for Serving**，核心思想：通过 task-level routing 将 MoE 模型的 decoder 从 "每 token 动态选 expert" 改为 "每 task 静态选 expert"，使得推理时只需加载 task-specific sub-network（K 个 experts），而非全部 E 个 experts。这避免了 token-level MoE 因 decoder 参数过大（远超单加速器内存）而需要的模型并行和跨设备通信。同时避免了蒸馏（distillation only preserves 32% of BLEU gains）。在 WMT 30 language pairs 上 peak throughput 提升 1.87x，decoder 参数从 221M 降至 25M（6.3%）。在 200 language pairs 上 peak throughput 提升 2.6x，decoder 参数从 6.5B 降至 201M（1.6%），communication overhead 从 36% 降至 0.2%。

  实验比较：
  - Task-MoE vs Token-MoE 在不同 batch size 下的推理吞吐量（Figure 2, Figure 4）
  - Task-MoE vs Distillation (Token MoE → Transformer-Base student) 在 8 个语言对上的 BLEU（Table 2）
  - 推理通信开销对比：Task-MoE (0.0%-0.2%) vs Token-MoE (26.9%-36% of step time)
  - Hybrid 策略：Task-MoE 在 decoder only (encoder 用 Token routing) 效果最好

- 硬件平台是什么，配置是什么。
  Cloud TPU V3：WMT 实验吞吐量测量使用 32 cores，大规模实验使用 128 cores。解码 WMT14 En-De test set 测量吞吐量。

- 开源Serving框架是什么。修改了什么。
  论文基于 Google 内部 GShard 框架（Lepikhin et al. 2020, TensorFlow/Lingvo），未开源。论文未修改 Serving 框架本身——其创新在于**路由算法层面**使得 MoE 模型天然适合高效 serving：task-level routing 使 decoder 的 expert 选择与 token 无关，因此每个 task 仅需加载 task-specific experts，无需模型并行或 all-to-all 通信。

  **Token-MoE Serving 的问题**：
  - 每个 token 独立选 expert → 不同 token 可能路由到不同加速器上的不同 expert → 需要动态加载 experts（host↔device 通信）或模型并行（inter-device all-to-all 通信）
  - 自回归解码的每一步都需要跨设备通信 → 通信开销被乘以 decoding steps
  - 小 batch 时只有部分 expert 被激活 → 设备利用率低

  **Task-MoE Serving 的解决方案**：
  - 相同 task 的所有 token 路由到相同 experts → 每个 task 只需预加载 K=2 个 experts
  - 不同 task 可独立、并行地在不同加速器上解码
  - 无跨设备通信（Task-MoE: 0.0%-0.2% vs Token-MoE: 26.9%-36%）

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  **开源情况**: 论文未公开代码。Google Research 内部基于 GShard/TensorFlow 实现。

  **Task-MoE Serving 全过程（以 WMT En→Fr 翻译为例，32 TPU V3 cores）**：

  1. **输入/请求**: 源语言句子 (En) → 目标语言 (Fr) 的翻译请求到达。目标语言 Fr 作为 task 标识。

  2. **Task-Specific Sub-Network 选择**: 根据 task_id (Fr) 查询 task embedding table → Router(task_emb) → Top-2 experts（如 expert 5 和 expert 17）→ 仅将这 2 个 experts 的权重加载到 TPU 内存。对于 32 expert 模型，decoder 仅用 2/32 experts → 25M params vs 221M full decoder。

  3. **Encoder 前向（Token-level routing）**: 源句子 tokens 经 encoder 处理。Encoder 使用 token-level MoE → 每个 token 独立 router → 动态选择 top-2 experts → encoder 输出源语言表示。（Encoder 推理成本可忽略：decoder 每步时间是 encoder 的 200x）。

  4. **Decoder 前向（Task-level routing）**: 自回归解码每步：
     - 所有 decoder tokens 因同属 task "Fr" → router 返回相同 top-2 experts（expert 5, 17）
     - Decoder MoE layer: x_s → FFN_5(x_s) + FFN_17(x_s)（加权）
     - 无需 all-to-all 通信（expert 5 和 17 已在同一设备）
     - 无 expert 动态加载（task 开始时预加载，持续复用）

  5. **多 Task 并行**: 不同 task（如 En→Fr, En→De）可分配到不同 TPU cores 或 core groups，各自加载自己的 task-specific experts 子网络，完全独立解码，无跨 task 通信。

  6. **输出**: 解码完成 → 输出 Fr 翻译结果。

  **与蒸馏对比（Table 2）**: Distillation 将 Token-MoE (533M) 蒸馏到 Dense Transformer-Base (142M) → BLEU 26.9（仅保留 32% MoE 增益）。Task-MoE (Token encoder + Target decoder) → BLEU 29.0（保留 100% MoE 增益 + 额外 +2.1 BLEU），同时 decoder 参数量 25M << 142M。

  **通信开销对比**: Token-MoE 解码时 26.9%-36% step time 用于跨设备通信（WMT/Large-scale）。Task-MoE 解码时 0.0%-0.2% step time 用于通信（可忽略），因为所有 token 路由到相同设备上的相同 experts。
