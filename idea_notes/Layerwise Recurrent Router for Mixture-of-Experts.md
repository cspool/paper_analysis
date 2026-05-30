## Layerwise Recurrent Router for Mixture-of-Experts

- baseline方法是什么？
  - **标准 SMoE (Switch Transformer)**：每层路由由独立的线性层 + softmax + top-k 构成。每层 router 参数 G_i ∈ R^(h, N)，对输入 token hidden state x_i 计算 gating score 后选择 top-k experts 进行 FFN 计算。不同层的 router 独立决策，不共享跨层路由信息。论文指出 (1) token 的 hidden state 虽然通过残差连接可以隐式传递跨层信息，但路由相关信息可能被 LM loss 的优化"淹没"；(2) 单个线性层 router 表达能力有限，token embedding 容易 collapse 到 expert embedding 附近（representation collapse）；(3) 早期 router 梯度主要来自 load balancing loss（而非 LM loss），导致早熟收敛到次优路由策略。
  - 全栈执行例子（Baseline SMoE，decoder-only transformer）：
    - **算法 Pipeline**：token x_i → Linear(G_i) → softmax → top-k → sparse FFN(selected experts) → output y_i → 残差 + LayerNorm → x_{i+1}。每层独立路由，无跨层信息。
    - **系统框架**：论文未明确说明，小实验用 PyTorch 原生实现，大实验用 Megablocks 框架。
    - **编译框架**：论文未明确说明。
    - **Kernel 调度**：论文未明确说明。Megablocks 本身提供 block-sparse kernel 加速 MoE 计算。
    - **硬件架构**：NVIDIA A100 GPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **RMoE**：在每层 router 前插入跨层共享的 GRU 单元，将路由决策从独立逐层计算改为跨层循环依赖。核心公式：x_i' = Proj_i(x_i), h_i = GRU(x_i', h_{i-1}), y_i = sum_n g_n(h_i; G_i, k) * E_n(x_i)。同时额外提供 Recurrent Gradient 路径优化 router 训练。
  - 对应解决 Baseline 缺陷：
    1. **跨层信息共享**：GRU 显式传递历史路由决策（h_{i-1} → h_i），使当前层 router 知道 token 在之前层被分配到哪些 experts，支持跨层协作。实验证明 RMoE 的跨层 mutual information 显著高于 SMoE/XMoE/HyperMoE。
    2. **Representation collapse 缓解**：Proj_i 投影 + GRU 将 hidden state 从 expert embedding 空间分离，类似于 XMoE 的低维投影策略，减少 token embedding collapse。
    3. **Router 梯度优化**：GRU 提供额外的 Recurrent Gradient 路径（跨层反向传播），使 router 优化不再被 LB loss 主导。实验显示 SMoE 的 linear router 梯度早期被 LB loss 主导，而 RMoE 的 GRU router 梯度持续由 LM loss 主导，达到更好的 LM/LB 权衡。
    4. **Moderate flat gating scores**：跨层信息共享使 gate score 分布呈现适度平坦（高熵但非随机），在 exploration vs exploitation 之间取得更好平衡，避免早熟收敛。RMoE 的 Top-1/Top-2 比率和 Outer Balance 均显著低于 SMoE。
    5. **正交兼容**：GRU 路由作为一个新的计算阶段，可与 XMoE、DeepSeekMoE 等现有方法组合，实验验证 XMoE+GRU router 在 3 种配置下均优于纯 XMoE。
  - 全栈执行例子（RMoE，decoder-only transformer）：
    - **算法 Pipeline**：token x_i → Proj_i(x_i) 降维 → x_i' → GRU(x_i', h_{i-1}) 结合历史 → h_i → Linear(G_i) + softmax + top-k → sparse FFN → y_i → 残差 + LayerNorm → x_{i+1}。跨层 GRU 提供前向路由信息和反向 Recurrent Gradient。
    - **系统框架**：论文未明确说明，小实验用 PyTorch 原生实现，大实验用 Megablocks 框架。
    - **编译框架**：论文未明确说明。
    - **Kernel 调度**：论文未明确说明。
    - **硬件架构**：NVIDIA A100 GPU。RMoE 仅增加 ~3.5M 参数（相对于 0.91B 模型），训练速度仅从 48.87 s/step 增加到 49.07 s/step（+0.4%），GPU 内存从 48.00GB 增加到 48.69GB（+1.4%），开销可忽略。
