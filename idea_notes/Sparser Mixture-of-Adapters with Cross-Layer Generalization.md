## Sparser Mixture-of-Adapters with Cross-Layer Generalization

- baseline方法是什么？
  - **Mixture of LoRA (MoL) / MultiLoRA**：传统 MoA 方法，每层维护独立的 N 个 LoRA adapter 专家池，由每层路由器（router）将输入 tokens 路由到该层的专家，专家的路由权重通过 token-expert 相似度或固定权重确定。每层专家之间不共享，所有 N×L 个专家始终处于激活状态（100% 利用率）。
  - 全栈执行例子（以 Phi-2 在 BoolQ 上推理为例）：
    - **算法 pipeline 层**：输入 token x 进入 layer l，该层的 router 计算 x 与该层 N=8 个 adapter expert embedding 的相似度，分配路由权重 u_n，输出为 y' = Vx + Σ_{n=1}^{N} u_n B_n^l A_n^l x。每个 adapter 只接收来自该层的 tokens 训练，不与其他层共享。
    - **系统框架层**：LoRA adapter 作为 HuggingFace PEFT 模块插入 target modules（如 q_proj, k_proj, v_proj）。每个 layer 的 adapter 参数独立存储和加载，总 trainable params = N×L×2dr（对 Phi-2: 8×32×2×2560×16 ≈ 33M，占 1.19%）。
    - **编译框架/Kernel调度/硬件架构层**：论文未明确说明，使用标准 PyTorch forward pass + NVIDIA A100 GPU 执行。
  - **核心缺陷**：Section 3 冗余分析揭示了四层冗余——(a) 同层专家之间冗余：mask 80% 同层专家性能几乎无下降；(b) backbone-expert 冗余：mask 全部同层专家（100%）仅降 ~0.14%；(c) 跨层冗余：同时 mask 多层专家性能下降极小（仅 mask all layers 才从 74.98% 降至 52.55%）；(d) 专家未充分利用：极端情况下单层专家超过全部专家。因此 baseline 的 adapter 缺乏专业化分工，未能充分利用 MoA 架构容量。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **SMOA (Sparser Mixture-of-Adapters)**：三个核心设计解决 baseline 缺陷：
    1. **跨层共享适配器池**：将 N 个 adapter 放入全局池，所有 L 层共享，每个 adapter 训练时接收来自不同层的 tokens。直接解决"跨层冗余"——通过强制共享消除每层独立训练导致的冗余。
    2. **全局路由器稀疏选择（Sparse Expert Selection）**：通过全局 router + 多数投票选出每层 top-n_l 个专家（而非全部 N 个），稀疏激活提升了专家利用率（Phi-2 利用率从 100% 降至 12.73%），解决"专家未充分利用"。
    3. **专家-冗余正则化 + Backbone Expert**：将 backbone 作为额外"专家"，通过正则项 R（式(8)）增大 backbone 路由权重 v_{l,i}，迫使 adapter 只学习 backbone 无法处理的残差知识。解决"backbone-expert 冗余"，同时促进 adapter 专业化（Figure 4 显示 SMOA 专家有明确任务偏好）。
    4. **课程学习（Specialization-to-Generalization）**：初始阶段 adapter 专注于特定层（深度专业化），逐渐允许跨层共享（泛化），平衡 specialization 与 generalization。
  - 全栈执行例子（以 SMOA + Phi-2 在 BoolQ 上推理为例，对比 baseline）：
    - **算法 pipeline 层**：
      - Baseline：每层独立计算 Δy = Σ_{n=1}^{8} u_n^l B_n^l A_n^l x，32 层共 256 个 adapter 参数独立。
      - SMOA：全局池 N=8 个 adapter，全局 router 计算所有 tokens 对 8 个 expert 的分数 w_{n,i} = softmax_n(<x_i, e_n>)，多数投票选出 top-n_l=8 个（即全选），重新归一化得 u_n，同时计算 backbone 相对适合度 v_{l,i}，输出 y' = Vx + (1 - v_l) · Σ_{n∈A_l} u_n · B_n A_n x。8 个 adapter 被 32 层共享复用，总 adapter 参数从 33M 降至仅需 8×2dr（但论文报告 trainable params 仍为 ~33.5M，与 MoL 相近，增量来自 embedding e_n 和 c_l）。
    - **系统框架层**：PEFT + PyTorch 实现，每个 adapter 的 LoRA 矩阵 B_n, A_n 全局存储一份，每层的前向计算通过全局 router 动态路由选择 adapter 子集并加权合并。训练时 curriculum learning 控制 adapter 可被哪些层选择。
    - **编译框架/Kernel调度/硬件架构层**：论文未明确说明。训练在 NVIDIA A100 上完成，wall-clock time per batch 38.54s（vs MoL 42.08s，MultiLoRA 31.85s），说明动态路由开销可控。
  - 关键结果：SMOA 在 4 个 base LLM 上全面超越 baseline：
    - Phi-2: 75.61% vs MoL 74.15%（+1.46%），adapter 利用率仅 12.73%
    - Phi-3: 82.23% vs LoRA 81.36%（+0.87%），利用率 58.75%
    - Gemma: 39.99% vs MultiLoRA 37.24%（+2.75%），利用率 60.39%
    - OLMo: 38.32% vs LoRA 36.82%（+1.50%），利用率 76.34%
    - OOD (Phi-2 MMLU): 56.19% vs MultiLoRA 55.19%（+1.00%）
    - 仅需 2 个激活专家即可达到接近 8 个专家的性能（Table 7）。
