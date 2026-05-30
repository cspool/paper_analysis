## Branch-Train-MiX Mixing Expert LLMs into a Mixture-of-Experts LLM

- baseline方法是什么？
  BTX 对比两类 baseline：

  **(1) Branch-Train-Merge (BTM)**：复制 seed 模型为多个 domain expert，各 expert 在领域数据上独立训练。推理时通过 TF-IDF 计算输入与各 expert 训练数据的相似度，选 Top-k expert 模型的输出 logits 做平均 ensemble。执行流程：输入 prompt → TF-IDF 嵌入 → cosine similarity 选 expert → 各选中 expert 独立 forward → 输出 logits 加权平均 → 预测 token。BTM 是特殊的 BTX（100% compute 给 expert training，0% 给 MoE finetune）。

  **(2) Sparse Upcycling**：从 seed dense checkpoint 将每层 FFN 复制为多个 identical expert，随机初始化 router，然后在混合数据上做 MoE 训练。这是 BTX 的另一特殊形式（0% expert training，100% MoE finetune）。执行流程：seed dense FFN → 复制为 4 个 identical expert → + 随机 router → Top-2 MoE training on mixed data → 统一 single model。

  **Baseline 全栈执行例子（以 BTM 处理 math/code 混合输入为例）**：
  - **算法层**：输入 prompt → TF-IDF 嵌入 → cosine sim 选 Top-2 expert（Math + Code）→ Math expert 独立 decoder forward（32层 Llama-2 FFN）→ Code expert 独立 decoder forward（32层）→ 两个 logits 向量直接平均 → argmax→输出 token。各 expert 之间无信息交换，无 token 级路由。
  - **系统框架层**：论文未明确说明（标准 PyTorch forward，无特殊 serving 框架）。
  - **编译框架层**：论文未明确说明。
  - **Kernel/运行时调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明（Meta FAIR 训练集群，GPU 细节未给出）。

  **Baseline 的核心缺陷**：
  1. **BTM 无统一模型**：expert 独立存在，不能做 SFT/RLHF 等后续微调（对齐 LLM 的关键步骤）
  2. **BTM 路由粗糙**：TF-IDF 整句级路由，无法做 token 级细粒度 routing，不同 token 需要的 expert 组合不同
  3. **BTM 无学习路由**：TF-IDF 相似度是静态的，无法学习最优 token→expert 映射
  4. **Sparse Upcycling 同步训练**：全部 compute 用于 MoE 训练，all-to-all 通信成本随 expert 数增长，训练吞吐低
  5. **Sparse Upcycling 无领域专门化**：expert 从同一 checkpoint 复制，无 domain specialization，性能不均衡

- 论文方法是什么？如何对应解决Baseline的缺陷？
  BTX 将 embarrassingly parallel 的 expert training 与 MoE finetune 结合：先异步独立训练 domain expert（继承 BTM 的训练效率），再将 expert FFN 组合为 MoE 层并 finetune router（解决 BTM 的无统一模型和无学习路由问题），同时 attention 参数平均（假设 attention 层 domain specialization 弱于 FFN）。

  | Baseline 缺陷 | BTX 设计选择 | 解决机制 |
  |---|---|---|
  | BTM 无统一模型，无法 SFT/RLHF | MoE finetune 阶段将所有 expert 合并为统一 MoE LLM | 最终模型是标准的 MoE Transformer，可直接做 SFT/RLHF |
  | BTM TF-IDF 整句路由粗糙 | MoE router 做 token 级 Top-k routing: g(W_l x) | 每个 token 独立选 top-2 expert，同一序列可用所有 domain expert |
  | BTM 路由无学习 | 随机初始化 W_l 然后 MoE finetune 训练 | Router 通过 80B tokens finetune 学习最优 token→expert 映射 |
  | Sparse Upcycling 同步训练通信高 | Expert 训练阶段 embarrassingly parallel | 无 all-to-all 通信直至 MoE finetune 阶段，训练吞吐线性 scaling |
  | Sparse Upcycling 无 domain specialization | 每个 expert 在独立领域数据上单独训练 | Math expert 在 math 任务上从 2.5→18.8 (MATH)，Code expert 从 12.8→31.7 (HumanEval) |
  | Expert 训练导致 catastrophic forgetting | 保留 Seed 模型作为 generalist expert + MoE finetune 混合所有数据 | BTX 在 reasoning 上 63.5 vs seed 63.3（无退化），Knowledge 41.0 vs 37.4（+3.6） |

  **BTX 方法全栈执行例子（以 Llama-2 7B seed + 4 experts + Top-2 routing 推理一个 token 为例）**：

  - **算法层**：
    token x 进入 layer l → Attention: 使用 4 expert 的平均 attention 权重 (W_q, W_k, W_v, W_o 均平均) → Router: logits = x @ W_l [4096, 4] → TopK(logits, k=2) → SoftMax(top2_vals) → 激活 2/4 FFN experts → y = w_1·FFN_math(x) + w_2·FFN_code(x) → 输出。Math 输入时 router 偏好 Math/Code expert，Knowledge 输入时偏好 Wiki/Llama-2 expert，Reasoning 时均衡使用 Math/Llama-2 expert。
  - **系统框架层**：论文未明确说明（标准 PyTorch MoE forward）。
  - **编译框架层**：论文未明确说明。
  - **Kernel/运行时调度层**：MoE 层仅计算 top-2 experts 的 FFN GEMM（4 个 expert 中 2 个），激活参数 11.1B，总参数 4×7B=28B。
  - **硬件架构层**：论文未明确说明（Meta FAIR 训练集群，GPU 细节未给出）。

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | Embarrassingly parallel expert training | Sparse upcycling 同步训练通信开销大 | Training time 7.8 GPU-days, BTX 训练 533B tokens vs sparse upcycling 252B tokens (同 compute) |
  | Expert FFN → MoE layer (仅 FFN 做 expert) | Attention 层 domain specialization 弱 | Freeze experts 后仅训练 router+attention: 性能几乎不变 (34.7 vs 34.7) |
  | Self-attention 权重平均 | 避免 attention 参数膨胀 | 无新增 attention 参数，与 seed model 结构完全兼容 |
  | Top-2 routing with load balancing (α=0.01) | Dead expert 问题（Code expert 无负载均衡时几乎不被激活） | Load balancing 使 Code expert 从 dead→在 math/code domain 主导 |
  | 保留 seed model 为 generalist expert | 防止 catastrophic forgetting，保留原有 general knowledge | Knowledge 41.0 vs seed 37.4 (+3.6), Reasoning 63.5 vs 63.3 (+0.2) |
  | MoE finetune 80B tokens (vs expert training 200B+ tokens each) | 仅用少量 compute 学会 router 和调优平均的 attention | BTX 47.9 vs BTM 43.4 (+4.5 average), BTX 47.9 vs Sparse Upcycling (DM) 46.3 (+1.6) |

  **创新总结**：BTX 的核心洞察是将 LLM 能力提升分解为两个解耦阶段——(1) expert training 阶段通过 embarrassingly parallel 训练最大化 compute 投入产出（线性 scaling），(2) MoE finetune 阶段将分离的 knowledge source 通过可学习的 token-level router 融合为统一模型。这使 BTX 同时获得 BTM 的训练效率和 MoE 的统一模型优势，避免了 BTM 的"无统一模型"和 sparse upcycling 的"同步训练通信高"两个极端。Blending experts 实验（将各 domain FFN 分块再混合）导致的性能大幅下降（Average 34.7→22.2）验证了"保留 domain specialization + 学习 router"而非"强制混合 domain knowledge"的设计决策正确性。
