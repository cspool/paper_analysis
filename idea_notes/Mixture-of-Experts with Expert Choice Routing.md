## Mixture-of-Experts with Expert Choice Routing

- baseline方法是什么？
  **Token-Choice Routing（以 GShard Top-2 gating 为例）**：传统 MoE 路由策略中，每个 token 通过 softmax gating 独立选择得分最高的 top-k 个专家（通常 k=1 或 k=2）。全栈执行例子如下：
  - **算法层**：输入 token X ∈ R^{n×d} → 计算 gating score S = Softmax(X · W_g) ∈ R^{n×e} → 对每个 token 取 TopK(S, k)，k=2 → 每个 token 分配到 2 个专家 → 各专家独立计算 FFN → 加权合并输出。k 对所有 token 固定。
  - **系统框架层**：MoE 层通过 "shuffle" 阶段将 token 按 expert ID 聚集（all-to-all dispatch），FFN 计算后 "unshuffle" 回原始顺序。为缓解负载不均，引入 auxiliary load balancing loss（如 Switch Transformer 的 load balance loss），但该 loss 需仔细调权以不压倒主 loss。
  - **编译框架层**：论文未明确说明（标准 TPU XLA 编译 + GSPMD 2D sharding）。
  - **kernel 调度层**：einsum 操作执行 expert FFN 的批量矩阵乘法。Token dispatch/gather 通过 TPU 的 collective communication 原语实现。负载不均导致最繁忙 expert 的 step latency 成为瓶颈（step time 比 EC 慢约 20%）。
  - **硬件架构层**：Google TPU V4（512 chips for 8B/64E），利用 2D torus 拓扑做 GSPMD sharding。负载不均导致部分 TPU core 过载、部分闲置。
  - **Baseline 核心缺陷**：
    1. **负载不均（Load Imbalance）**：token-choice 独立路由导致某些专家接收远超容量的 token。auxiliary loss 无法保证均衡，尤其在训练早期，过容量比率可达 20%-40%，大量 token 被丢弃。
    2. **专家欠专业化（Under Specialization）**：过大的 auxiliary loss 倾向负载均衡但路由效果差，导致专家冗余或不够专精。在负载均衡与专业化之间取得平衡极为困难。
    3. **每个 token 固定计算量**：所有 token 精确分配 k 个专家，无论其复杂度。重要 token 和简单 token 获得相同计算资源，浪费且不灵活。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Expert Choice Routing**：反转路由方向，让每个专家独立选择 top-k 个 token（而非 token 选专家）。全栈执行路径如下：
  - **算法层（Expert Choice Routing）**：
    1. 计算 token-to-expert affinity S = Softmax(X · W_g) ∈ R^{n×e}（与 baseline 相同）
    2. 反转：对 S^T ∈ R^{e×n} 的每一行（每个专家）取 TopK，k = n×c/e（固定专家容量）→ G, I = TopK(S^T, k)
    3. 排列矩阵 P = OneHot(I) ∈ R^{e×k×n}，将 token 按专家分组：X_in = P · X ∈ R^{e×k×d}
    4. 各专家独立 FFN：X_e[i] = GeLU(X_in[i] · W_1[i]) · W_2[i]^T
    5. 反排列 + 门控加权：X_out[l,d] = Σ_{i,j} P[i,j,l] · G[i,j] · X_e[i,j,d]
    - 每个 expert 恰好处理 k 个 token，负载天然完美均衡。每个 token 可被 0~e 个专家选中，实现可变计算分配。
  - **系统框架层**：与 baseline 相同的 MoE 层结构（每两层 Transformer 替换一层 FFN 为 MoE），shuffle/unshuffle 阶段不变。关键区别：无需 auxiliary load balancing loss——负载均衡由算法本身保证。容量系数 c=2 匹配 GShard top-2 的 per-token 计算量。
  - **编译框架层**：论文未明确说明（标准 TPU XLA + GSPMD 2D sharding，与 baseline 一致）。
  - **kernel 调度层**：einsum 操作与 baseline 相同。因负载完美均衡，所有 expert 计算时间一致，step latency 由均匀负载决定（无 straggler expert），step 时间比 GShard top-2 快约 20%。
  - **硬件架构层**：Google TPU V4（512 chips for 8B/64E），与 baseline 一致。负载均衡使 TPU 利用率更高，无闲置 core。

  **缺陷 → 方法设计直接映射**：
  - **负载不均 → 专家选 token + 固定专家容量 k**：每个专家恰好接收 k 个 token（k = n×c/e），从设计上消除负载不均。无需 auxiliary loss，训练早期即保持均衡。效果：EC-CF2 收敛速度比 GShard top-2 快 2× 以上，且每步快 20%。
  - **欠专业化 → 学习到的 token-expert affinity 不受 load balance loss 干扰**：无 auxiliary loss 意味着 gating 网络的学习目标纯粹是最大化 token-expert affinity，自然产生更专业化的专家。效果：下游 GLUE/SuperGLUE 11 任务平均 accuracy 提升 2%+（8B/64E: EC-CF2 92.6 vs GS Top-2 90.3 vs ST Top-1 88.9）。
  - **固定计算量 → 可变数量专家 per token**：每个 token 可被 0~e 个专家选中（实际分布：约 77% tokens 被 1-2 个专家选中，23% 被 3-4 个，3% 被 >4 个）。Ablation 验证：限制每个 token 最多 2 个专家（EC-CAP2）导致 accuracy 下降 0.8 points，证实可变专家数有效。

  **可选约束扩展（EC-CAP）**：通过熵正则化线性规划 + Dykstra 交替投影算法限制每个 token 最多 b 个专家。EC-CAP3（b=3）达到与无约束 EC-CF2 相当的 accuracy，验证了可变分配的有效性同时提供了可控性。
