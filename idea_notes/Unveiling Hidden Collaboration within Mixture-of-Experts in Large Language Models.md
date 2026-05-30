## Unveiling Hidden Collaboration within Mixture-of-Experts in Large Language Models

- baseline方法是什么？
  Baseline 是**传统 MoE LLM 的分析与剪枝方法，将专家视为独立实体**。具体包括：(1) Router 分析——仅研究单个 router 的 top-k 选择行为（如输出 norm 偏好、token ID 关联），不揭示专家间的协作关系；(2) 独立专家剪枝——SEER-MoE 基于路由分数高低剪枝（Muzio et al., 2024），GEM 基于 |x-f(x)| 差异识别输出影响最小的专家（Zhang et al., 2024），Random 随机删除专家。这些方法均将每个专家独立评估，忽略了跨层专家之间的协作与互补关系。

  全栈执行例子（以 DeepSeek-MoE-16B 上 GEM 剪枝 25% 专家为例）：
  - **模型推理算法层**：对每个 MoE 层的每个专家独立计算其对最终输出的影响度（|x - f_expert(x)|），按影响度排序后删除影响最小的 25% 专家，不考虑跨层专家共激活模式。剪枝后剩余专家保留原始路由权重，Router 仍按 top-k 选择（可能选到对特定任务已不完整的专家组合）。在 HellaSwag 上 accuracy 从约 0.69 降至 0.658。
  - **系统框架层**：基于 HuggingFace Transformers 加载 DeepSeek-MoE-16B，通过直接修改模型权重（删除 experts）实现剪枝。评估使用 EleutherAI LM Harness 框架，normalized zero-shot accuracy。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出**基于层级稀疏字典学习（HSDL）发现专家协作模式，并基于协作模式进行贡献感知专家剪枝（CAEP）**的方法，通过两个关键设计解决 Baseline 缺陷：

  1. **HSDL 揭示跨层专家协作模式**——不同于 Baseline 将专家视为独立实体，HSDL 从 expert activation matrix X 出发，通过多层稀疏字典学习递归分解，发现跨层专家之间的共激活协作模式（如 Layer 5 Expert 21 + Layer 6 Expert 3 频繁同时激活）。字典每个 atom 编码一组协作者，稀疏编码 R 控制各模式在不同样本上的参与度。实验验证：(a) 60% 的字典模式对应穷举搜索中 Top 10% 最高频组合；(b) 层级语义标注显示高层字典捕获粗粒度类别（"数学计算"），深层字典细化为子任务（"日期/符号识别"）；(c) 语义相近领域（数学/物理/计算机科学）的专家激活分布相似度高，语义不同的领域（数学/法律）分布差异大。

  2. **CAEP 基于协作模式贡献剪枝**——不同于 Baseline 按单个专家的路由分数或输出影响力独立排序，CAEP 结合字典矩阵 D 和稀疏编码 R 计算每个专家的综合贡献分数 e = Σ D_sum[:,i]，在迭代中优先移除最少被使用的协作模式（pattern），而非仅按个体分数截断。这确保了剪枝后保留的专家仍然形成完整的协作组合，维持任务处理能力。实验表明：CAEP 在剪枝 25% 专家后平均 accuracy 0.612，优于 SEER-MoE (0.5872) 和 GEM (0.5870)，如在 OBQA 上从 0.420 提升至 0.473，HellaSwag 上从 0.658 提升至 0.691。

  全栈执行例子（以 DeepSeek-MoE-16B 上 CAEP 剪枝 25% 专家为例）：
  - **模型推理算法层**：Step 1——在 MMLU 128 个样本上收集 expert activation matrix X（每个 token 的 router 分配 α 按句子求和 v_{i,j,k} = Σ α(i)_{t,j,k}）。Step 2——HSDL 多层分解 X 得到字典 D（编码跨层专家协作模式）和稀疏编码 R。Step 3——计算每个专家的贡献分数 e，取 k_1-分位数作为阈值生成初始 mask，迭代移除最少使用的 pattern 并更新 mask 直至达到 25% 剪枝比。与 GEM 不同，CAEP 保留了如 {L5-E21, L6-E3} 等共激活组合中的双方专家，避免因单独删除一个专家而破坏协作模式。剪枝后模型在 HellaSwag 上 accuracy 为 0.691（vs GEM 0.658），OBQA 上 0.473（vs SEER-MoE 0.420）。
  - **系统框架层**：基于 HuggingFace Transformers 加载 DeepSeek-MoE-16B，通过 mask vector m ∈ {0,1}^{N_e} 标记保留/删除的 normal experts（shared experts 保留），剪枝后参数量 = 16.4 - 14.7 × k% B（式(10)）。评估使用 EleutherAI LM Harness 框架。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。
