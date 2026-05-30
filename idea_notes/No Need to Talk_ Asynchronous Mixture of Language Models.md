## No Need to Talk: Asynchronous Mixture of Language Models

- baseline方法是什么？
  Baseline 是标准同步分布式数据并行训练的 dense LLM（Standard Synchronous Distributed Data Parallel Training）：
  - 算法层：单个 dense Transformer decoder 模型，所有参数同时参与训练和推理。
  - 系统框架层：分布式训练使用 PyTorch/JAX 的数据并行，每个 training step 后进行 all-reduce 梯度同步。对于 1.3B 参数模型，每步每节点需传输约 10.4GB 梯度数据（float32）。推理时整个模型需全部加载到 GPU 内存。
  - 编译框架层：论文未明确说明（使用标准 PyTorch eager mode / JAX XLA 编译）。
  - Kernel调度层：论文未明确说明（使用框架默认 kernel 实现，如 cuBLAS、FlashAttention 等标准算子）。
  - 硬件架构层：依赖高带宽互联（如 NVLink、InfiniBand）进行梯度同步，8-128 GPUs 集群。
  Baseline 的核心痛点：(1) **训练通信瓶颈**——每次迭代需要 all-reduce 同步梯度，数据量巨大（1.3B 模型每步 ~10.4GB/节点），严重依赖高速互联硬件；(2) **推理参数冗余**——推理时所有参数均需驻留在 GPU 内存中并参与计算，无法稀疏激活；(3) **异步训练方案性能退化**——已有的异步 SGD、Local SGD 等减少同步频率的方法会导致 perplexity 显著低于每步同步的 baseline；(4) **先前 MoE 方案依赖 token 级路由**——Switch Transformer 等 MoE 方法虽能稀疏激活参数，但路由决策在每个 token 上做出，要求所有 expert 常驻 RAM 且仍需高通信开销进行梯度同步。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 SMALLTALK LM——一种序列级硬混合专家（Hard MoE）方法，通过小型路由 LM 实现数据分区和独立 expert 训练：
  - 算法层：将训练分解为两个解耦阶段。**Stage 1——Router 训练**：使用 E 个极小的语言模型（4.4M 参数，仅为 expert 的 1.3%）作为 router，通过 EM 算法（交替优化 router 的 NLL 和 balanced assignment）学会将数据按 prefix（256 token）分配到不同 expert。关键设计是 **balanced assignments**——按 min log-likelihood 排序后贪心分配，确保每个 expert 获得等量数据。**Stage 2——Expert 训练**：每个 expert 在自己的数据子集上完全独立训练，无需任何梯度同步。推理时，router 对输入 prefix 评分并选择单个 expert 执行自回归生成。
  - 系统框架层：训练框架从同步分布式变为**完全独立并行**——router 训练用 PyTorch（仅需少量 all-gather 通信，每次 <6MB，总计约 100 次），expert 训练用 JAX 独立运行，零通信。推理时仅需加载被选中的单个 expert（总参数的 1/E），其他 expert 可以 offload 或驻留在不同节点。对比 baseline：1.3B × 32 experts 模型，训练通信从 ~10.4GB/步 降至 router 训练期间约 100 次 <6MB 通信，推理参数从全量 1.3B 降至单 expert 1.3B（相同的 inference FLOPs）。
  - 编译框架层：论文未明确说明（使用标准 PyTorch eager mode / JAX 编译）。
  - Kernel调度层：论文未明确说明（使用框架默认 kernel）。
  - 硬件架构层：由于 expert 训练和推理完全独立，**不再需要高带宽互联**——每个 expert 可在独立的低带宽节点上训练，甚至可以在不同时间、不同地理位置训练。对比 baseline 需要紧密互联的 GPU 集群，SMALLTALK LM 可在松耦合的异构节点上运行。

  核心设计如何解决 baseline 痛点：
  1. **训练通信瓶颈** → Router 完成数据分区后，expert 训练**零梯度同步**，仅 router 训练期间有少量 loss 值通信（<100 次，每次 <6MB）。
  2. **推理参数冗余** → 序列级路由使推理时仅激活 **1 个 expert**，参数量与 dense baseline 相同，但混合模型总容量为 E 倍。
  3. **异步训练性能退化** → 不同于 Local SGD 的方案（因梯度延迟导致性能下降），本方法不降级是因为每个 expert 在自己的不相交数据子集上做**标准同步训练**（子集内），不存在跨 expert 的梯度 staleness 问题。最终 1.3B × 32 experts 的 perplexity 比 dense baseline 低 17.56%。
  4. **Token 级 MoE 的高通信/内存需求** → 序列级路由使 expert 间完全解耦，无需在推理时为每个 token 做跨 expert 路由决策，也无需所有 expert 常驻 RAM。

- baseline方法是什么？
  Baseline 是 LLaVA-1.5 的全参数微调方法（以及对 MoE 模型做全参数微调的 MoExtend-Full / MoE-LLaVA）。核心缺陷：
  (1) **Catastrophic Forgetting（灾难性遗忘）**：全参数微调让 LLM 在学会视觉理解的同时遗忘原有的文本知识。Mixtral 8x7B 全参数微调后（MoExtend-Full），在纯文本 benchmark 上平均下降 3.30 分（Avg. drop），而 MoE 架构对全参数微调尤为敏感——MoE-LLaVA 的 Avg. drop 高达 7.86 分。
  (2) **高昂的训练成本**：全参数微调 Mixtral 8x7B 需要 ~200 小时的 instruction tuning（8×A800），而随着模型规模增长，这一成本愈发不可承受。
  (3) **模态间隙未有效弥合**：使用少量线性投影层或 LoRA 等参数高效方法虽然减少遗忘，但无法让 LLM 充分理解新模态，限制了多模态能力。

  全栈执行例子：一张图片 + 一个问题文本输入 LLaVA-1.5 → CLIP ViT 编码 visual token (P×D) → project 层映射到 LLM hidden space → 与 text token 拼接为 (N+P)×D → 进入 Vicuna-13B（dense LLM）逐层 attention + FFN 前向 → 全参数微调时所有 13B 参数参与梯度更新 → 问题：原有 MMLU/GSM8K 等文本知识在 FFN 权重中被覆盖 → 推理文本任务时使用被覆盖的 FFN 权重，性能下降。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoExtend**，一种专为 MoE 模型设计的新模态扩展方法。核心思路：**冻结所有原有 MoE 参数，只在关键层新增 expert，仅训练新增部分**。

  **对应缺陷 1（Catastrophic Forgetting）→ 冻结原有参数 + Calibration Module**
  - 不修改原有 MoE 模型的任何参数（expert FFN、router、attention 全部冻结），原有知识完全保留。
  - 新增 expert 后 softmax 概率分布会变化：s(x)_j' = e^{f(x)_j} / (Σ e^{f(x)_h} + e^{f(x)_{m+1}}) ≤ s(x)_j，原有 expert 被选概率下降，forward 输出分布漂移。Calibration Module 通过 s_c(x) 修正每个 expert 的输出权重：MoE(x) = Σ s(x)_j · [1+s_c(x)] · FFN(x)_j，s_c 初始零输出来保证初始一致性。

  **对应缺陷 2（高昂训练成本）→ 仅训练新增参数**
  - 新增 expert 仅添加到 50% 的 MoE 层（由 Extender 自动决定），每层仅加 1 个 expert。对于 Mixtral 8x7B（32 层，8 experts/层），仅训练 16 个新 expert + 对应 router 列 + 轻量 Calibration modules。
  - 激活训练参数量仅 ~3B，训练时间 ~30 小时（Alignment ~15h + Fine-tuning ~30h），对比全参数微调 ~200 小时，加速约 6 倍。

  **对应缺陷 3（模态间隙）→ 新增专用 expert 而非投影层微调**
  - 新增的 expert 是完整的 FFN 层（而非 LoRA adapter 或投影层），拥有足够容量学习新模态的特征变换。
  - 新 expert 初始化策略：复制该层对视觉数据响应最活跃的原有 expert 权重，使得新 expert 从"最接近视觉理解"的参数空间出发训练，加速收敛并保证选中概率。

  全栈执行例子：一张图片 + 问题文本输入 MoExtend → CLIP ViT 编码 visual token → MLP project 对齐 → 拼接为 (N+P)×D → 进入 Mixtral 8x7B，逐层 attention + MoE：在未扩展层（16层），router 从 8 个原有 expert 中选 top-2 计算（与原始模型完全一致，无遗忘风险）；在扩展层（16层），router 从 9 个 expert（8 个冻结原有 + 1 个新增可训练）中选 top-2 → 若新增 expert 被选中，其 forward 输出经过 Calibration s_c(x) 修正后加权求和 → 新增 expert 专门处理视觉 token，原有 expert 处理文本 token → 仅新增 expert 的 FFN 参数 + v_new router 列 + s_c 参与梯度更新（~3B params vs 46.7B 全参数）→ 推理时与原始 MoE 流程一致，无额外开销。
