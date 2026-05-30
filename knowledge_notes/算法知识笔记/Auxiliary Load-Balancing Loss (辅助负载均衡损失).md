## Auxiliary Load-Balancing Loss (辅助负载均衡损失)

术语解释
Auxiliary Load-Balancing Loss 是 MoE 训练中添加到主语言模型损失上的辅助损失项，用于鼓励 gate network 将 token 均匀分配到各 expert，避免某些 expert 过载（overloaded）而其他 expert 闲置（underutilized）。首次由 Shazeer et al. (2017) 在 Sparsely-Gated MoE 中引入，后由 Switch Transformer (Fedus et al., 2022) 推广。

术语是什么？
Load-balancing loss 的标准形式（Switch Transformer）：
$$L_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

其中 f_i = 分配给 expert i 的 token 比例，P_i = router 分配给 expert i 的平均 gate probability，N = expert 总数，α = auxiliary loss coefficient（典型值 10^-2 ~ 10^-5）。

符号化辅助损失的变体（DeepSeek-V3, Wang et al., 2024）直接在 router scores 中注入 expert-level bias，而非修改 loss function，实现 auxiliary-loss-free load balancing。

从算法pipeline角度拆解术语：
MoE 训练 loss 组成：
```
L_total = L_LM + α × L_aux
# L_LM: cross-entropy language modeling loss
# L_aux: load balancing penalty
# α: coefficient controlling the tradeoff

L_aux = E × Σ_i (f_i × P_i)
# f_i = (1/T) × Σ_t 1[token t routed to expert i]  # fraction of tokens
# P_i = (1/T) × Σ_t softmax(gate_logits[t])[i]      # avg routing probability
```

术语一般如何实现？如何使用？
- **α 调优至关重要**：SYMI 论文 Figure 11 显示 DeepSpeed 需要高 α (~10^-1) 才能将 token drop 从 ~40% 降至 ~10%，但高 α 干扰主 loss 收敛
- **SYMI 的发现**：有了 adaptive expert replication 后，SYMI 在任何 α 下均保持 ~10% token drops，auxiliary loss 从"系统必需项"降级为"质量调节旋钮 (quality knob)"
- 替代方案：Expert-Choice routing (Zhou et al., 2022) 天然负载均衡无需 auxiliary loss；BASE Layers (Lewis et al., 2021) 用线性分配保证均衡；DeepSeek 的 auxiliary-loss-free 策略 (Wang et al., 2024) 直接在 router 中注入 bias；ARIA 的 group-level load balancing（见 Group-Level Load Balancing Loss）将 per-expert 约束松弛为 per-group 约束；**DSMoE 刻意不引入 load balancing loss**（见下文）
- **DSMoE 的"无 Load Balancing"设计**：DSMoE 明确不引入 load balancing loss，因为其目标不是 expert 均匀使用，而是学习输入自适应的稀疏激活模式。DSMoE 的 sigmoid 门控（非 softmax）使 expert 激活决策互不依赖，配合 L1 sparse loss 施加稀疏压力，形成"STE 让所有 expert 保持可训练 + sparse loss 鼓励选择性激活"的对抗训练机制。论文在 10B tokens 继续预训练后未观察到严重的 expert 负载不均，且 W 形层间激活模式表明不同层自然形成不同的激活水平
- **AquilaMoE 实践**：AquilaMoE 训练 8×16B MoE 时使用 α=0.001 的 load balancing loss + α=0.01 的 max z-loss，两者均以乘法系数形式施加于最终训练目标，用于防止训练崩溃并维持 expert 负载均衡。Scale-Out 阶段每 token 激活 top-2/8 experts（约 30B 激活参数）

**Micro-Batch vs Global-Batch LBL（Demons in the Detail, 2025）**：

Qiu et al. (2025) 揭示了 LBL 计算粒度对 MoE 性能和 expert specialization 的关键影响：

- **Micro-batch LBL (LBL_micro)**：主流框架（DeepSpeed-MoE, Tutel, MegaBlocks, Megatron-Core）的默认行为。每个 parallel group（即每个 GPU 的 micro-batch）内独立计算 f_i 和 P_i，然后 all-gather 平均。在大模型训练中 micro-batch 仅含极少序列（数千 tokens），LBL 退化到序列级均衡——强制每个序列内的 token 均匀分配到所有 expert。这抑制了 domain-level expert specialization。
- **Global-batch LBL (LBL_global)**：跨并行组同步专家选择频率 f_i（仅 N_E 维向量），用全局 f̄_i 替换本地 f_i 计算 LBL。约束从"每序列内均匀"放松为"全语料库均匀"。额外通信开销 <1%。
- **Buffer 近似机制**：当计算节点有限（微批总和 < 全局批大小），在 GA 各步缓冲累积同步后的 c_i 逐步逼近 global f̄_i。
- **Balance BSZ**：论文引入的度量指标，表示计算专家选择频率时考虑的总 token 数。实验证明 Balance BSZ 从 2 增加到 512，PPL 持续下降 (~0.185)。
- **Shuffle LBL_micro 消融**：通过 all-gather token-expert selection matrix G 并随机抽取等量 token 计算 LBL（保持 token 数与 micro-batch 相同但分布等同 global-batch），证实性能提升来自 **token 多样性**而非 token 数量（方差降低）。
- **缓解局部负载不均**：Global-batch LBL 可能导致局部计算不均（~5.8% slowdown）。加微量 micro-batch LBL（1% weight of global-batch）可将速度恢复至仅 2.6% 慢于 baseline，性能损失极小。

涉及论文标题：
- SYMI: Accelerating Mixture-of-Experts Training with Adaptive Expert Replication
- AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies
- Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models
- Continual Pre-training of MoEs How robust is your router（CPT 中 PBTk routing 使用 α=0.01 的 aux loss + λ=0.001 的 z-loss。CPT 分布偏移时 PBTk 经历短暂 MRI spike 后 ~500 steps 恢复，aux loss 学习新的负载均衡模式。与 SBTk 的显式均衡相比，PBTk 的 penalty-based 方法恢复后的 MRI 更低）

涉及论文标题：
- Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)
- A Survey on Mixture of Experts in Large Language Models
- AdaMOE Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models
- AdaMoLE Fine-Tuning Large Language Models with Adaptive Mixture of Low-Rank Adaptation Experts
- Adaptive Gating in Mixture-of-Experts based Language Models
- Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference
- DSMoE Matrix-Partitioned Experts with Dynamic Routing for Computation-Efficient Dense LLMs
- Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models

**Task-MoE 的负载均衡（Kudugunta et al., EMNLP 2021）**：
Task-MoE 使用 task-level routing，所有 token 按 task 预先分组 → 同 task 的 token 必然路由到相同 experts → task 内的 expert load 天然均衡。负载均衡仅在跨 task 之间需要（确保不同 task 不过度集中到少数 experts）。论文使用 standard auxiliary load balancing loss (α=0.001) with top-2 gating。128 expert 配置下 decoder 仅 2/128 experts per task，专家负载天然跨 task 分散。

Li et al. (EMNLP 2023) 对标准 load balancing loss 做了最简洁的修改以适应 flexible expert count：

**核心修改**：由于 adaptive gating 中 token 可能使用 1 或 2 个 expert，load balancing loss 仅对 top-1 gating 决策施加软约束，top-2 gating 决策完全自由：

$$L_i = E_i \sum_{e \in E} f_e^1 \cdot p_e$$

其中 $f_e^1$ 为 top-1 gating token 中分派到 expert e 的比例（而非所有 token），$p_e$ 为所有 token 对 expert e 的平均门控概率，$E_i$ 为第 i 层 expert 数量。

**设计理由**：top-2 决策代表 token 确实需要双专家处理的"困难"情况，对这些 token 施加负载均衡约束会干扰其学习。仅约束 top-1 决策在保证基本负载均衡的同时，给予 router 在困难 token 上完全的路由自由度。

与其他负载均衡变体的对比：
- Standard (Switch Transformer): 对所有 token 施加约束 → 不适用灵活 expert 数
- AdaMOE ℓ_null: null experts 间不做负载均衡 → 适用于 null expert 范式
- Adaptive Gating: 仅 top-1 决策施加约束 → 最简修改，配合阈值门控

### AdaMOE 的 Null Expert Load Balancing Loss (ℓ_null)

AdaMOE 对标准 load balancing loss 做了关键修改以适应 null experts：

**修改 1 — 不对 null experts 之间做负载均衡**：
由于所有 m 个 null experts 在功能上完全相同（均为 zero mapping），对它们之间做负载均衡区分会施加不必要的约束。AdaMOE 将 null experts 的负载因子 f_j 替换为均值：

$$\tilde{f}_i = \begin{cases} f_i & \text{if } i \leq n \text{ (true expert)} \\ \frac{1}{m} \sum_{j=n+1}^{n+m} f_j & \text{if } i > n \text{ (null expert)} \end{cases}$$

$$\ell_{null} = \alpha \cdot (n+m) \cdot \sum_{i=1}^{n+m} \tilde{f}_i \cdot P_i$$

实验验证 ℓ_null 显著优于对所有 null experts 做负载均衡的 ℓ_bal：RTE accuracy 67.51 vs 56.68, COLA 85.01 vs 83.68。

**修改 2 — α annealing 策略**：
- Epoch 1: α=0.02（大 α）→ 严格负载均衡，确保 tokens 不全部涌向 true experts
- Epoch 2: α=0.0001（小 α）→ 释放 token 自由度，让 router 根据任务需求自由分配
- 效果: WINO accuracy 从 epoch 1 的 76.24 提升至 epoch 2 的 81.93 (+5.69%)，Load 几乎不变 (1.65→1.66)

**修改 3 — Normalization 策略**：
仅对 top-k 中选中的 true experts 做 Softmax normalization（option 2），而非对所有 k 个选中的 expert（含 null）做 Softmax（option 1）。保证加权输出与 vanilla MoE 数值尺度一致。SIQA: option 2 accuracy 81.27 vs option 1 80.19。

---
