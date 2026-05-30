## Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference

- baseline方法是什么？
  **Baseline 为 Token-level MoE (Token-MoE)**：在 Multilingual NMT 中，MoE Transformer 的 encoder 和 decoder 都使用 token-level routing——每个输入 token x_s 独立经过 router GATE(x_s) 计算 top-2 experts，不同 token 可能路由到不同的 experts。模型使用 top-2 gating，E 个 experts 的 FFN 替换 Transformer 的 alternate layers。训练时使用 auxiliary load balancing loss 确保 expert 利用率均衡。

  **Baseline 全栈执行例子（以 Token-MoE 32 experts WMT En→Fr 翻译为例）**：
  - **算法层**: 输入 token "Bonjour" → GATE(x_s) = TopK(Softmax(W_g · x_s), k=2) → 不同 token 独立选 expert（"Bon"→expert 3,7; "jour"→expert 12,5; "##s"→expert 1,8）→ y_s = G[3]·FFN_3(x_s) + G[7]·FFN_7(x_s)。Decoder 自回归每步：每个新 token 重新 router → 可能路由到不同的 expert，需要加载全部 32 experts 的动态通信。
  - **系统框架层**: GShard (TensorFlow/Lingvo) → Expert Parallelism: 32 experts 分布在多个 TPU 设备上 → all-to-all dispatch tokens → expert FFN 计算 → all-to-all combine → 自回归解码每步重复 all-to-all → 通信开销占总 step time 26.9%。
  - **编译框架层**: 论文未明确说明（Google 内部 TensorFlow/XLA 编译）。
  - **Kernel/运行时调度层**: 每个 TPU core 加载部分 experts → Router kernel → All-to-all token dispatch kernel → FFN GEMM kernel → All-to-all combine kernel → 每 decoding step 重复。
  - **硬件架构层**: 32 Cloud TPU V3 cores → Decoder 221M params 需跨多 TPU 设备 → dynamic routing 导致跨设备通信 → 解码器每步时间是 encoder 的 200x → peak throughput 1.3×10^5 tokens/s。

  **Baseline 的核心缺陷**：
  1. **Decoder 参数膨胀**: 全部 E experts 需常驻或可访问 → decoder 221M-6.5B params，超出单加速器内存 → 必须模型并行。
  2. **自回归解码通信放大**: 每 decoding step 都需要 all-to-all 通信 → 通信开销 26.9%-36% × N_decoding_steps。
  3. **小 batch 设备利用不足**: 小 batch 下只有少量 expert 被激活 → 大量设备空闲。
  4. **蒸馏损失**: 蒸馏 Token-MoE → Dense 模型仅保留 32% BLEU 增益。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**: Task-level MoE (Task-MoE) = 将 MoE 路由从 token-level 改为 task-level，使同一 task（language pair / target language）的所有 token 路由到相同的 experts。配合 hybrid 策略：encoder 用 token-level routing（保持灵活性），decoder 用 task-level routing（dominant inference cost）。

  **Defect→Design 映射**：

  | Baseline 缺陷 | Task-MoE 设计选择 | 解决机制 |
  |---|---|---|
  | Decoder 参数膨胀需模型并行 | Task-level routing: 每 task 仅需 K=2 experts | Decoder 参数从 221M→25M (↓88%), 6.5B→201M (↓97%)，单加速器即可容纳 |
  | 每 decoding step all-to-all 通信 | 所有 token 路由到相同 experts（同设备） | 通信开销从 26.9%→0.0% (WMT), 36%→0.2% (large-scale) |
  | 小 batch 设备利用率低 | 每 task 独立部署 sub-network | 不同 task 可在不同设备上独立并行解码 |
  | 蒸馏仅保留 32% BLEU 增益 | Sub-network extraction 保留 100% 增益 | Task-MoE BLEU 29.0 vs Distillation 26.9 (+2.1) |
  | Token-MoE encoder 灵活性丧失 | Hybrid: Token encoder + Task decoder | Encoder 保持 token-level 灵活性（Xx-En 更好），decoder 获得 task-level 效率（decoder 占 200x 时间） |

  **Task-MoE 全栈执行例子（以 WMT En→Fr 翻译，32 experts，hybrid Token/Target 策略为例）**：

  - **算法层**: 
    Encoder: x_s (source token) → GATE(x_s) = TopK(Softmax(W_g · x_s), k=2) → token-level routing → 不同 source token 可走不同 expert → 输出 source hidden states。Decoder: target language "Fr" → task_emb = Embedding("Fr") → GATE(task_emb) = TopK(Softmax(W_g · task_emb), k=2) → 如 expert 5, 17 → 所有 decoder tokens 走 expert 5 + 17 → y_s = G[5]·FFN_5(x_s) + G[17]·FFN_17(x_s)。Task boundary 由 target language 定义（French→English 和 German→English 同 task "English" 选相同 experts），或由 language pair 定义（各自独立选 experts）。

  - **系统框架层**: Task-specific sub-network 提取 → 每个 task 仅加载 K=2 experts 到单 TPU device → 无 all-to-all → 无跨设备通信 → Decoder 前向: Router (task_emb) 一次计算，所有 decoder step 复用 → Expert FFN 计算（仅 2 experts）→ peak throughput 2.3×10^5 tokens/s。

  - **编译框架层**: 论文未明确说明（Google 内部 TensorFlow/XLA）。

  - **Kernel/运行时调度层**: Router kernel: task_emb lookup → Softmax → TopK（仅需运行一次 per task，非 per token）。FFN kernel: 仅 2 experts GEMM（非全部 32/128）。无 all-to-all dispatch/combine kernel。Decoder 自回归每步 kernel 执行路径缩短 → peak throughput 1.87x-2.6x。

  - **硬件架构层**: 32-128 Cloud TPU V3 cores → 每 task 分配专用 sub-network → 不同 task 可在不同 core groups 独立解码 → communication overhead 0.0%-0.2% → decoder step time 大幅缩短。

  **路由决策分析（Section 5.4）**：对 Token-MoE 的 gating decisions 可视化发现——decoder 中 task-level 决策自然出现（related languages share similar expert distributions，如 Spanish-Catalan, Russian-Ukrainian），encoder 中所有 Xx-En 任务偏好相同的少数 experts。这为 hybrid 策略（Token encoder + Task decoder）提供了实证支持。

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | Task-level routing on decoder | Decoder 参数从 221M→25M (WMT), 6.5B→201M (200 langs) | Peak throughput 1.87x (WMT), 2.6x (200 langs) |
  | Hybrid Token encoder + Task decoder | Encoder 保持灵活性，decoder 获得效率 | BLEU 23.6 (vs Token/Token 22.6, +1.0) |
  | Target language 作为 task boundary | 最大化 transfer: xx→En 共享同组 experts | Target/Target BLEU 22.9 vs LanguagePair/LanguagePair 21.4 |
  | Sub-network extraction 替代蒸馏 | 100% 保留 MoE BLEU 增益 vs 蒸馏 32% | Task-MoE 29.0 vs Distillation 26.9 BLEU |
  | Per-task expert 不共享 | 不同 task 加载不同 sub-network 独立解码 | No communication overhead (0.0-0.2% vs 26.9-36%) |

  **创新总结**: Task-MoE 的核心洞察是——在 Multi-task learning（MNMT）场景中，task boundary 是已知先验，可以直接利用来替代 token-level routing 的 dynamic selection。通过将路由决策从 "per token" 提升到 "per task"，将不可控的 token 级动态通信转化为可控的 task 级静态部署，从而在不蒸馏、不量化、不剪枝的情况下实现高效推理。其局限在于仅适用于 task boundary 明确的多任务场景（如 multilingual NMT），不适用于通用单任务 LLM。
