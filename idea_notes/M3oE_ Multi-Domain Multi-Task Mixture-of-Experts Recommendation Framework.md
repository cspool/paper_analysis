## M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework

- baseline方法是什么？
  - Baseline 为单域单任务 MLP、多任务方法（ShBot-MTL, PLE-MTL, MMoE-MTL, AdaTT）、多域方法（STAR, ShBot-MDL, MMoE-MDL, PLE-MDL）、以及多域多任务方法（ShBot-MDMT, MMoE-MDMT, PLE-MDMT, M2M）。以 MMoE-MDMT 为例，它在每个域上复用 MMoE 的 shared expert + task-specific gate/tower 结构，所有 task gate 对所有 expert 输出进行加权求和后送入各 task tower 预测。该全栈执行路径为：
    - 算法层：输入特征 x_d → 共享 embedding 层 → N 个 shared expert（MLP + ReLU）→ 每个 task t 的 gate 对 expert 输出做 softmax 加权 → task-specific prediction tower（2 层 MLP + Sigmoid）→ y_hat_{d,t}。所有域共享同一套 expert 参数，gate 在每个 domain 独立运行但结构相同。
    - 系统框架层：论文未明确说明（标准 PyTorch 训练，无 Serving 改造）。
    - 编译框架层：论文未明确说明。
    - kernel 调度层：论文未明确说明。
    - 硬件架构层：论文未明确说明。
  - MMoE-MDMT 的核心缺陷在于：(1) 域间信息和任务间信息均通过同一套 shared expert 隐式学习，缺乏对 domain-specific 和 task-specific 模式的显式建模；(2) 融合方式单一（仅 gate 加权），无法精确控制域/任务/共享信息的贡献比例；(3) 出现 MDMT seesaw 现象——同一多域信息传递方法无法泛化到不同任务，同一多任务优化平衡策略无法泛化到不同域。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - M3oE 通过三个解耦的专家模块 + 两级自适应融合机制解决上述缺陷：
    - 算法层全栈执行路径：输入 x_d → **Domain Representation Extraction**（W_d ⊙ W_sh 元素乘捕获域特定+共享模式 → W_c 映射到统一空间 + f_DA 域无关映射）→ h_d → 并行送入三个模块：
      1. **Shared Expert Module S**：N 个 expert，每个 = ReLU(LayerNorm(W_e h_d + b_e))，D×T 个独立 gate 做 softmax 加权求和 → S_{d,t}(h_d)，捕获跨域跨任务共性。
      2. **Domain Expert Module D**：D 个 expert（每域一个），f_E^d(h_d) 为对应域 expert 输出，β_d 控制当前域 vs 其他域的加权融合 → D(h_d)，显式捕获域特定偏好。
      3. **Task Expert Module T**：T 个 expert（每任务一个），f_E^t(h_d) 为对应任务 expert 输出，β_t 控制当前任务 vs 其他任务的加权融合 → T(h_d)，显式捕获任务特定偏好。
      → **两级融合**：Level-1（β_d/β_t 控制域间/任务间融合）+ Level-2（α_d/α_t 控制域/任务/共享模块间的贡献平衡）→ h̄_d = S(h_d) + α_d·T(h_d) + α_t·D(h_d) → D×T 个独立 prediction tower → y_hat_{d,t}。
      → **AutoML Bi-Level Optimization**：α_d, α_t, β_d, β_t 由 Sigmoid(可训练标量 e_w) 生成，与模型参数交替优化（外层更新 W，内层更新 α/β），自适应确定每对 (d,t) 的最优融合权重。
    - 系统框架层：论文未明确说明（标准 PyTorch 训练）。
    - 编译框架层：论文未明确说明。
    - kernel 调度层：论文未明确说明。
    - 硬件架构层：论文未明确说明。
  - 对比 baseline 的改进映射：
    - MMoE 只有一个 shared expert 模块 → M3oE 新增 domain expert 和 task expert 模块，显式建模域/任务特定信息，解决"无法泛化到不同域/任务"的 MDMT seesaw。
    - MMoE 用统一 gate 融合 → M3oE 用两级融合（β 控制专家内部平衡 + α 控制模块间平衡），实现更精细的信息贡献控制。
    - MMoE 固定架构 → M3oE 通过 AutoML 自适应学习融合权重，无需人工为不同数据集调参，提升泛化能力。
