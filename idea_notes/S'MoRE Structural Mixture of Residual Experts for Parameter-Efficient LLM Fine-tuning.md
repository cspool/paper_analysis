## S'MoRE Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning

- baseline方法是什么？
  **Baseline: MoLRE (Mixture of Low-Rank Experts, 即 MixLoRA) 和 MoMOR (Mixture of Multi-Order Residues)**
  
  LoRA 将预训练权重的更新限制在低秩空间 ΔW = B·A（B∈R^{d×r}, A∈R^{r×d}），参数高效但模型容量受限于单个低秩矩阵。MixLoRA（MoLRE）将其扩展为 x' = Σ_{i=1}^s ROUTE(x)^i · B^i · A^i · x，即多个低秩专家的加权组合。然而这种"扁平"结构存在两个缺陷：(1) 路由灵活性有限——每个 token 仅从 s 个专家中选 k 个，总路由组合数为 C(s,k)，增加专家数虽然能提高灵活性，但会导致专家利用不均和路由开销增大；(2) 结构无关性——同一组被激活的专家无论按何种结构连接，都输出相同的结果（因为等价于简单的加权求和），这意味着模型的表达能力完全取决于"激活哪些专家"，而无法从"如何连接专家"中获益。
  
  全栈执行例子（以 MixLoRA 推理一个 token 为例）：
  - 算法层：输入 token embedding x(4096d) → Router 计算 ROUTE(x)^i = softmax(W_gate·x) → top-k 选择 2 个专家 → 并行执行 B^i·A^i·x（8个秩16矩阵）→ 加权求和输出 x'(4096d) → 加到 pre-trained FFN 输出。整个过程是"选择+线性组合"，无结构化信息。
  - 系统框架：论文未明确说明（使用标准 PyTorch 训练框架 LLaMA-Factory，无自定义 serving 修改）
  - 编译框架：论文未明确说明
  - kernel调度：论文未明确说明（标准 PyTorch CUDA kernel 执行矩阵乘法）
  - 硬件架构：论文未明确说明

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **S'MoRE: Structural Mixture of Residual Experts**
  
  S'MoRE 的核心创新是将"扁平专家选择"升级为"分层结构化专家组合"。它没有增加专家数量，而是通过将专家组织成多层结构并利用非线性层间传播，使得同一组参数可以形成指数级的不同前向路径（非树形结构），从而极大提升模型容量。具体设计对应 baseline 的两个缺陷：
  
  **缺陷1"路由灵活性有限"→ 解决：树形分层路由**
  - 将 s 个专家按 L 层排列，路由器自顶向下逐层条件化选择：P(child | parent_ancestors, x) = softmax(⟨k_child, MLP(concat(x_down, ancestor_keys))⟩)
  - 每层 fanout f_ℓ，总激活节点 F_ℓ = Π_{i=ℓ}^{L-1} f_i
  - 1层 MixLoRA 路由组合数 ≤ C(s, k)；2层 S'MoRE 在相同参数下结构灵活性 Γ = Π_ℓ C(s_ℓ, f_ℓ)^{F_{ℓ+1}}（组合数作为指数的指数级增长）
  
  **缺陷2"结构无关性"→ 解决：非线性层间聚合 + GIN 启发的传播**
  - 每层聚合 x_{ℓ+1}^i = Σ α · σ(B·A·x + W·x_prev)，其中 σ 是非线性激活（ReLU）
  - 理论证明：无 σ 时，多层的线性组合可坍缩为等价单层（MoMOR）；加入 σ 后，L 层传播模拟 L 轮 WL 同构测试，不同树结构的输出不同
  - 图3示例：同一组激活专家 {"0,1","0,2","0,3","0,4","1,1","1,2"} 按三种不同树结构连接（非树形）→ MixLoRA/MoMOR 输出相同 → S'MoRE 输出三种不同结果
  
  全栈执行例子（以 S'MoRE L=2 推理一个 token 为例）：
  - 算法层：输入 x(4096d) → [路由阶段 自顶向下] 层2 Router MLP_1(concat(x_down)) → softmax(⟨k_1^i, q_1⟩) → top-2 选择专家 "0,1"和"0,2" → 对于每个选中父节点，层1 Router MLP_0(concat(x_down, k_parent)) → 分别选子节点（"0,1"选{"1,1","1,2"}，"0,2"选{"1,3","1,4"}）→ 构建两棵残差树 → [聚合阶段 自底向上] 层1：对每组子节点计算 σ(B_0^n·A_0^n·x)（带 skip connection 和 ReLU），求和生成 x_1^{parent} → 层2：对两个 x_1 输出计算 W_1·x_1 + B_1·A_1·x，经 ReLU 后求和 → 最终 W_proj 映射回 4096d。路由选择的树结构不同，即使专家集合相同，非线性传播路径不同，输出也不同。
  - 系统框架：LLaMA-Factory 训练框架，OpenCompass 评测框架。adapter 插入 FFN 和 attention 模块，与 LoRA 使用方式一致（论文未修改 serving 框架）
  - 编译框架：论文未明确说明
  - kernel调度：标准 PyTorch 矩阵乘法，论文未涉及自定义 kernel
  - 硬件架构：论文未明确说明（NVIDIA GPU 训练，具体型号未披露）
  
  关键理论保证：Theorem 3.4 证明 S'MoRE 的结构灵活性 Γ_{S'MoRE} = Π C(s_ℓ, f_ℓ)^{F_{ℓ+1}}，在 s_ℓ=4, f_ℓ=2, L=2 时远超 MoMOR 的上界 Γ_{MoMOR} ≤ C(s, f)。本质原因是：MoMOR 中激活专家组合数仅与选择相关（线性累加 C(s,i)），而 S'MoRE 中每个节点独立选择子节点，且非线性传播使不同树结构输出可区分，因此灵活性按"每个节点的组合选择"的乘积指数增长。
