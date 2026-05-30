## Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

- baseline方法是什么？
  Baseline 是传统的 KD (Sanh, 2019) 和 GKD (Agarwal et al., 2024)。KD 使用 forward KL divergence 最小化 teacher 和 student 在 token 级别的分布差异（使用 teacher 的原始 Top-k routing 选择 activated experts）。GKD 则使用 student 生成的 on-policy 数据 + reverse KL divergence，但同样仅依赖 MoE teacher 的 Top-k routing 机制。

  **Baseline 全栈执行例子（以 Llama-MoE-3.5B (4/16) → Sheared-Llama-1.3B 蒸馏为例）**：
  - **算法层**：MoE 教师使用 Noise Top-k Gating 计算 gate logits → softmax → Top-k selection。对于每个输入 token，仅激活 k=4 个 expert，gate probabilities 中 non-activated experts 的总概率 >50%（大部分层的 activated experts gate prob sum <50%），意味着大部分 expert knowledge 未被利用。教师输出 logits 与 student 之间计算 KL divergence 作为蒸馏损失。
  - **系统框架层**：论文未明确说明（标准 PyTorch/HuggingFace Transformers 训练循环，无特定 serving 框架修改）
  - **编译框架层**：论文未明确说明（使用 SynapseAI 编译框架，Intel Gaudi v2 加速器后端）
  - **Kernel 调度层**：论文未明确说明（标准 MoE FFN 前向传播 kernel，expert selection 通过 KeepTopK 掩码实现）
  - **硬件架构层**：4 × Intel Gaudi v2 加速器，路由选择在 Gaudi 上执行，无法将不同 expert 的激活分配到不同设备上以利用所有 expert

  **Baseline 核心痛点**：
  1. MoE 教师仅有 Top-k expert 参与知识生成，non-activated experts 不参与。但 non-activated experts 的 gate probabilities 总和超过 50%，意味着大量有价值的知识被浪费。
  2. KD 和 GKD 均为 dense-to-dense 场景设计，不感知 MoE 的 expert routing 特性。在 dense teacher 和 MoE teacher 性能相当的情况下，dense teacher 竟然作为更好的教师（学生 ROUGE-L 更高），说明现有方法未能有效利用 MoE 的分布式知识。
  3. Load balancing 使同一输入的不同训练迭代可能激活不同 expert 集合，知识被分散到多个 expert 中，但 conventional KD 每次只取 Top-k，无法覆盖完整知识。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出两种 MoE 专用的 KD 方法：

  1. **Knowledge Augmentation (KA)**：每次迭代对同一输入进行 M 次教师前向传播，每次以概率 λ 从 gate probability 分布中随机采样 N-1 个 expert（以 1-λ 概率取 Top N-1 个），生成 M 份增广知识。使用 student 生成的 pseudo-target + reverse KL 进行蒸馏。
  2. **Student-Aware Router (SAR)**：先以 student 反馈（reverse KL + auxiliary load balancing loss β=0.01）训练 MoE 教师的路由器，再使用更新后的路由器激活所有 expert 并加权聚合输出进行蒸馏。

  **论文方法全栈执行例子（与 baseline 对比）**：
  - **算法层**：
    KA 模式下：MoE 教师前向 → gate logits H(x) 计算 → softmax → 以 λ=0.05 概率从 gate prob 采样 N-1=15 个 expert（以 0.95 概率取 Top 15）→ 激活 15 个 expert → 加权聚合 → 重复 M=2 次，每次不同 expert 组合（通过采样随机性）→ 每次均与 student on-policy 输出计算 reverse KL → 参数更新。
    SAR 模式下：student 前向生成 pseudo-target → teacher 前向（激活所有 16 个 expert，使用完整 gate prob 加权）→ 计算 reverse KL + β·L_b → 仅更新 router 参数 W_g, W_noise → router 更新后，teacher 再次前向（激活所有 expert，用更新后 gate prob 加权）→ 计算 student reverse KL → 更新 student 参数。

  - **系统框架层**：论文未明确说明（标准训练框架，与 baseline 相同）
  - **编译框架层**：论文未明确说明（与 baseline 相同，使用 SynapseAI 在 Gaudi v2 上执行）
  - **Kernel 调度层**：论文未明确说明。KA 在同一迭代内对同一输入执行 M 次教师前向（M=2），每次选 N-1 个 expert，计算量增加但 non-activated expert 减少。SAR 激活所有 expert（全激活），每次迭代多一次 router 更新前向，计算开销通过 β=0.01 的轻量 load balancing 控制。
  - **硬件架构层**：论文未明确说明（与 baseline 相同，4 × Intel Gaudi v2）

  **三个缺陷的对应解决**：
  | Baseline 缺陷 | 论文解决方案 |
  |---|---|
  | Non-activated experts 知识未被利用（gate prob sum >50% 的 expert 被丢弃） | KA 将激活 expert 从 k 扩展到 N-1，覆盖几乎全部 expert；SAR 激活所有 N 个 expert，100% 覆盖 |
  | Dense teacher 比 MoE teacher 更好（现有 KD 不适用于 MoE） | KA 和 SAR 在 MoE teacher 下均超过 dense teacher + GKD 的效果（如 KA: 25.71 avg vs GKD dense: 24.89 avg） |
  | Load balancing 导致知识分散但 KD 只取单次 Top-k | KA 通过 M 次采样聚合不同 expert 组合的知识，模拟多次 routing 的多样性；SAR 通过训练 router 动态调整 expert 权重，让 router 感知 student 需求 |

  实验效果：(a) KA 和 SAR 在所有 MoE teacher 配置下均超过 KD 和 GKD baselines，最高提升 KA +4.8 ROUGE-L over KD；(b) ALL（直接全激活）优于 KD/GKD 但不如 KA/SAR，验证 router 训练的价值；(c) KA 的 M=2 最佳，M 过大导致过度多样性反而降低性能；(d) SAR 的 KL divergence 随层深增加而增加，说明 student-friendly router 调整在深层累积效果明显；(e) λ=0.05 取得最佳 trade-off（随机采样与确定性选择的平衡），过大 λ 导致性能下降。
