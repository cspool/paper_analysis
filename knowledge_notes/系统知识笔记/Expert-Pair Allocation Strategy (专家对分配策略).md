## Expert-Pair Allocation Strategy (专家对分配策略)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert-Pair Allocation Strategy 是 MoDSE（Mixture of Diverse Size Experts）论文提出的 GPU 负载均衡策略。在 MoDSE 中，同一 MoE layer 内的 expert 具有不同的 hidden dimension（大专家参数量大、计算量大，小专家参数量小、计算量小），直接部署到 GPU 会导致 GPU 间负载不均——分配到大专家的 GPU 计算负载显著高于分配到小专家的 GPU。Expert-Pair Allocation 将 expert 按对分组 $(E_{i_k^1}, E_{i_k^2})$，每对满足参数量之和相等（$\hat{h}_{i_k^1} + \hat{h}_{i_k^2} = 2h$），并将每对 expert 放置在同一 GPU 上。这样每个 GPU 上的总参数量和总计算量完全一致，实现 GPU 间负载均衡。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
以 300M×8 MoDSE 模型（8 expert, 4 GPU）为例的 expert-pair allocation 流程：
```
# Expert 配置（8 experts, 4 pairs）
# Pair 0: E_0(h=6912, 4.5×), E_1(h=768, 0.5×)  -> GPU 0
# Pair 1: E_2(h=6144, 4.0×), E_3(h=1536, 1.0×) -> GPU 1
# Pair 2: E_4(h=4608, 3.0×), E_5(h=3072, 2.0×) -> GPU 2
# Pair 3: E_6(h=3840, 2.5×), E_7(h=3840, 2.5×) -> GPU 3
# 每对参数量: 2*dim*(h_i1+h_i2) = 2*dim*7680, 所有 GPU 相等

# 每个 MoE Layer 的 forward pass:
# Step 1: All-to-All dispatch (token routing)
# Router 决定每个 token 的 top-2 experts
# Token 通过 all-to-all 发送到对应 expert 所在的 GPU

# Step 2: Expert computation (各 GPU 独立)
# GPU 0: 计算 E_0 (大expert, 高计算量) 和 E_1 (小expert, 低计算量)
# GPU 1: 计算 E_2 (大expert) 和 E_3 (小expert)
# GPU 2: 计算 E_4 (中expert) 和 E_5 (中expert)
# GPU 3: 计算 E_6 和 E_7 (两中等 expert)
# 由于每对 expert 的总参数量相同 + 负载均衡损失保证 token 均匀分配
# → 每个 GPU 上的总计算量基本一致

# Step 3: All-to-All combine
# Expert 输出通过 all-to-all 发送回原始 token 所在的 GPU
# Weighted sum 得到最终 MoE layer output
```

关键设计：
- 依赖于负载均衡辅助损失（Switch Transformer 式 L_aux = α·N·Σ f_i·P_i）确保每个 expert 被路由的频率相近
- 即使单个 expert 尺寸不同，每个 GPU 上分配的 expert 对的总参数量保持一致
- 推理时同理：expert-pair allocation 使得 MoDSE 的推理速度与 baseline 几乎相同（论文 Table 4 验证）

术语一般如何实现？如何使用？
- 实现：在模型初始化和分布式部署时，将 expert 对映射到 GPU rank。在 PyTorch DDP/FSDP 或 Megatron-LM 等分布式框架中，通过自定义 expert placement 逻辑将 Expert 对绑定到同一 device。
- 需要与 all-to-all 通信配合：MoE 的 token dispatch 通常使用 all-to-all 将 token 发送到 expert 所在 GPU，expert-pair allocation 确保 all-to-all 后每个 GPU 的计算负载均衡。
- 该策略的前提是 expert 数量为偶数（可成对分组），每对 expert 内一大一小互补，且负载均衡损失收敛良好（训练后期 max/min token ratio 从 >3.0 降至 <3.0）。
- 局限性：论文仅在小规模 MoE（300M×8, 700M×8）上验证，大规模 MoE 的 expert-pair allocation 在更多 GPU 下的通信模式和负载均衡性质未知。

涉及论文标题：
- Mixture of Diverse Size Experts
