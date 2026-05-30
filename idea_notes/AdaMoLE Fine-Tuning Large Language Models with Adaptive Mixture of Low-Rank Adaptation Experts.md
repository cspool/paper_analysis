## AdaMoLE Fine-Tuning Large Language Models with Adaptive Mixture of Low-Rank Adaptation Experts

- baseline方法是什么？
  **Baseline: MoLE (Mixture of LoRA Experts) with 静态 top-k 门控**。在每层 Transformer 的 self-attention 权重矩阵（Wq, Wk, Wv, Wo）上，用 N 个 LoRA 专家（各 rank=r）替代单个 LoRA（rank=N×r），路由器计算 Softmax(W_g x) 得到 N 个专家权重，通过固定 top-k（k=2 或 k=3）选择最高的 k 个专家，其余权重置零，归一化后加权求和。或者使用固定阈值 τ=1/N 的硬阈值策略。

  **Baseline 缺陷**：
  1. **static top-k 对所有 token 同等对待**：无论是简单还是复杂的 token/任务，始终激活固定数量专家，无法根据输入复杂度灵活调整资源分配。
  2. **固定阈值 τ=1/N 缺乏上下文感知**：阈值无法随输入语义变化，不能区分何时需要更多专家（如复杂推理）或更少专家（如简单语法）。
  3. **资源浪费或欠利用**：简单 token 激活过多专家浪费计算，复杂 token 可能激活不足导致精度损失。

  **Baseline 全栈执行例子（以 Llama-2-7B + MoLE top-2，处理单个 token x 为例）**：
  - **算法层**：输入 x 经 router 计算 8 个专家权重 p = Softmax(W_g x) → TopK(p, k=2) 选出权重最高的 2 个专家 → 输出 h = W_0 x + (p_1 E_1(x) + p_2 E_2(x)) / (p_1 + p_2)，每个 E_i(x) = B_i A_i x ∈ R^d
  - **系统框架层**：HuggingFace Transformers + PEFT → 替换 self-attention 四矩阵的 LoRA adapter 为 8 个 LoRA expert → forward 时 router 和 top-k 在 PyTorch eager 模式执行 → 所有 expert 的 A_i, B_i 矩阵都已加载到 H100 GPU 显存
  - **编译框架层**：论文未明确说明（PyTorch eager execution）
  - **Kernel/运行时调度层**：无论输入简单或复杂，始终启动 2 个 LoRA expert 的 GEMM kernel（每个 expert = r×k · d×r 两次矩阵乘法）→ 简单 token 浪费 1 个 expert 的 kernel 计算 → 复杂 token 可能 2 个 expert 仍不足
  - **硬件架构层**：单张 NVIDIA H100 GPU → top-2 的 2 次 LoRA GEMM 消耗固定 CUDA core 和显存带宽 → 简单 token 下 SM 做无效计算

- 论文方法是什么？如何对应解决Baseline的缺陷？

  **论文方法**: AdaMoLE = LoRA + 自适应 MoE，引入可学习的动态阈值网络替代静态 top-k 选择。关键设计：
  1. **动态阈值网络**：τ = τ_max · σ(W_τ x + b_τ)，τ_max = 1/N，单层线性层 + sigmoid 使 τ 随输入 x 自适应变化
  2. **基于阈值的专家选择**：激活所有 p_i ≥ τ 的专家，而非固定 k 个
  3. **可导门控公式**：用 (p_i - τ) 替代原始 p_i，确保 τ 参与反向传播梯度计算，使阈值网络可学习
  4. **参数等价性**：8 个 rank-4 专家 = 单个 rank-32 LoRA，总参数量相同（除门控和阈值网络额外参数）

  **Defect→Design 映射**：

  | Baseline 缺陷 | AdaMoLE 设计选择 | 解决机制 |
  |---|---|---|
  | static top-k 对所有 token 同等对待 | 动态阈值 τ(x) = τ_max · σ(W_τ x + b_τ) | 每个 token 根据自身特征计算独立阈值，复杂 token 得到较低 τ（激活更多专家），简单 token 得到较高 τ（激活更少专家） |
  | 固定阈值 τ=1/N 缺乏上下文感知 | 阈值网络以输入 x 为条件 | τ 成为 x 的函数，不同语义输入自然产生不同阈值和专家激活数 |
  | 简单 token 浪费计算 | 高 τ 只激活极少数专家 | Table 4 显示 τ∈[0,3/(2N)] 时平均仅激活 1.26 专家（CommonsenseQA），远少于 top-2 的固定 2 个 |
  | 复杂 token 精度不足 | 低 τ 允许更多专家参与 | τ∈[0,1/(2N)] 时平均激活 6.59 专家（CommonsenseQA），远多于 top-2，准确率 78.95% vs top-2 的 77.15% |

  **论文方法全栈执行例子（以 Llama-2-7B + AdaMoLE，阈值范围 [0, 1/N]，处理单个 token x 为例）**：
  - **算法层**：输入 x → router 计算 8 维权重 p = Softmax(W_g x) → 阈值网络计算 τ = (1/N) · σ(W_τ x + b_τ) → 激活集合 S = {i | p_i ≥ τ} → 输出 h = W_0 x + Σ_{i∈S} (p_i - τ) B_i A_i x / Σ_{j∈S} (p_j - τ) → S 的大小随 x 动态变化，平均 3.46（CommonsenseQA）或 4.56（COPA）
  - **系统框架层**：HuggingFace Transformers + PEFT → 额外加入阈值网络（单层 Linear + Sigmoid）→ forward 时 router、阈值计算、条件专家激活均在 PyTorch eager 执行 → 所有 8 个 LoRA expert 矩阵常驻 H100 显存
  - **编译框架层**：论文未明确说明（PyTorch eager execution）
  - **Kernel/运行时调度层**：简单 token（τ 高）→ 仅 1-2 个 expert 的 GEMM kernel 启动 → 复杂 token（τ 低）→ 可能 6-8 个 expert GEMM kernel 启动 → 实际 kernel launch 数动态变化，理论计算量与输入复杂度成正比
  - **硬件架构层**：单张 NVIDIA H100 GPU → 简单 token 少用 CUDA core/显存带宽，复杂 token 多用 → 相比 top-2 固定 2 expert，AdaMoLE 能对简单 token 节省 ~40% 计算（1.26 vs 2 expert），对复杂 token 额外利用更多 expert 提升精度

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | τ = τ_max · σ(W_τ x + b_τ) 动态阈值 | 替代 static top-k 实现上下文自适应专家选择 | CommonsenseQA: 78.71% vs MoLE top-2 77.15% (+1.56%); COPA: 94.00% vs 92.00% (+2.00%) |
  | (p_i - τ) 可导门控公式 | 使阈值网络可端到端训练 | 阈值网络参数可通过交叉熵 loss 反向传播学习 |
  | Threshold sensitivity τ∈[0,1/(2N)] vs [0,1/N] | 调整计算-精度权衡 | τ∈[0,1/(2N)]: CommonsenseQA 78.95%, 平均激活 6.59 专家; τ∈[0,1/N]: 78.71%, 平均激活 3.46 专家 |
  | 较低层更多专家激活 | 利用 LLM 层次特征：低层处理多样基础特征需更多专家 | Figure 2: Layers 1-10 激活更多专家，Layers 25-32 激活较少（CommonsenseQA & COPA） |
  | Hyperparameter robustness (N×r) | 方法对各配置鲁棒 | N=4×r=4: 78.38%; N=8×r=4: 78.71%; N=16×r=4: 78.13%; 均超各自 MoLE baseline |
  | Cross-model generalization | 方法适用于不同基础模型 | Gemma-7B: 81.00% vs LoRA 80.51% (+0.49%); Llama-2-13B: 81.74% vs LoRA 79.77% (+1.97%) |
