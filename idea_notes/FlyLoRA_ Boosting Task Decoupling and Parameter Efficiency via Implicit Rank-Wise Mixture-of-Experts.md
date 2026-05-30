## FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts

- baseline方法是什么？
  - **LoRA(r=8) / LoRA(r=32)**：标准低秩适配，将参数更新 ΔW 分解为 ΔW = B_{m×r}·A_{r×n}，其中 A 和 B 均可训练。全栈执行例子：输入 token x → 预训练权重 W0·x + (α/r)·B·(A·x)。存在 **intra-task interference**：同一 LoRA 组件内不同 rank 间梯度相关（full covariance），rank 间参数耦合导致 suboptimal 性能，且高 rank 需要更多训练参数。存在 **inter-task interference**：多任务模型合并时，task_i 的 B_i·A_i 与 task_j 的 B_j·A_j 在参数空间中重叠（因可训练 A 之间无正交性保证），合并出现性能崩塌（如 ScienceQA 上 LoRA(r=8) 合并后 Δ=-60.34%）。
  - **Split-LoRA(4×8)**（MoE-based LoRA 代表）：将 rank r=32 分解为 4 个 expert × 8 rank，router G(x)=top-k(sigmoid(W_g·x)) 选择激活 expert。引入显式 router 参数 W_g∈R^{N×n}。全栈执行例子：输入 x → router 计算 softmax(W_g·x) 选 top-k expert → Σ_i G(x)_i·B_i·(A_i·x)。痛点：(1) router 参数开销随 expert 数 N 线性增长（N=32 experts 时 router 成为瓶颈）；(2) 单 expert 内仍存在 rank 间 interference；(3) 多任务合并无本质改善（ScienceQA Δ=-54.74%），因 A 仍可训练、Expert 更新不天然正交。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **FlyLoRA 方法**：基于果蝇嗅觉回路的三层设计。(1) 矩阵 A 替换为**冻结的稀疏随机投影**（每行 p 个非零 ~N(0,1/r²)），既是下投影又是隐式 router；(2) 在 B 矩阵中执行 **rank-wise top-k 专家激活**——r=32 个 rank-1 专家，每 token 仅激活 top-k=8 个；(3) 通过 **loss-free 负载均衡偏置 d** 辅助专家选择。
  - 解决 baseline 缺陷的对应机制：
    1. **消除显式 router 参数**：A 的稀疏随机投影天然保留了 pairwise 距离（Theorem 3.1, Johnson-Lindenstrauss 延伸），语义相似 token 被路由到相似 expert，实现与 hash router 等效的轻量级隐式路由。不再需要 W_g，彻底消除 router 参数开销。激活训练参数仅 d·k（vs LoRA 的 2·d·k 和 Split-LoRA 的 2·d·k+d·N）。
    2. **Intra-task 去耦合**：top-k 稀疏激活使不同 B 列的梯度仅在同时被选中时才产生协方差。Theorem 3.3 证明 off-diagonal 梯度协方差按 k²/r² 因子缩减。当 k=8, r=32 时，off-diagonal 协方差约为 dense 训练（k=r）的 1/16。图 3(b-c) 的梯度相关热力图验证了 FlyLoRA 显著稀疏的正交模式。
    3. **Inter-task 去耦合**：独立初始化的冻结随机投影 A_i 和 A_j 天然近似正交。Theorem 3.4/Collorary 3.5 证明 <B_i·A_i, B_j·A_j>_F ≈ 0，不同任务的参数更新位于近似正交子空间，合并时无破坏性 interference。模型合并后 MMLU 仅降 2.02%（vs LoRA 降 4.91-6.48%，Split-LoRA 降 4.86%），ScienceQA 降 43.05%（vs LoRA 降 60.34%）。
  - 全栈执行例子（FlyLoRA 单 task 训练+合并推理）：
    - **算法层**：Token x → 冻结 A 做 sparse projection y=A·x（每行 p 稀疏非零，O(r·p)）→ 加负载偏置 d → top-k argmax 选 I_topk → 仅激活 B 的 k 列 → ΔW·x = (α/r)·Σ_{i∈I_topk} b_i·(a_i·x)。Backward 仅更新 B：grad_B 中非 top-k 列为 0。
    - **系统框架层**：论文使用标准 PyTorch + Transformers 训练框架（AdamW optimizer），未修改 serving 框架。PEFT 注入方式：对 LLM 所有 linear 层（q,k,v,o,gate,down,up}_proj）替换为 FlyLoRA 模块。训练内存：FlyLoRA 在 Llama-3.1-8B 上最低 10.6GB（vs LoRA(r=8) 12.5GB），因 A 冻结减少了激活值存储。
    - **编译框架层**：论文未明确说明（使用标准 PyTorch 编译栈，无自定义编译 pass 或 kernel）。
    - **Kernel 调度层**：论文未明确说明（top-k 操作、稀疏 mask 乘法均使用 PyTorch 原生算子，未自定义 CUDA kernel）。
    - **硬件架构/芯片设计层**：论文未明确说明（使用 consumer GPU RTX 3090 / A100，无自定义 RTL 或硬件修改）。
  - 关键数据（Llama-3.1-8B）：FlyLoRA(k=8) 在 MMLU 上 40.88% vs LoRA(r=32) 38.93% vs Split-LoRA(4×8) 38.44%，且激活训练参数仅 0.13%（vs LoRA(r=32) 1.03%, Split-LoRA 0.33%）。训练时间：FlyLoRA MMLU 上 4.73h（8×RTX3090）vs LoRA(r=32) 5.09h。多任务合并后 MMLU 仅降 2.02% vs LoRA(r=8) 降 6.48%。
