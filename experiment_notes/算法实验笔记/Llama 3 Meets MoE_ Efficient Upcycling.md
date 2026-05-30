## Llama 3 Meets MoE: Efficient Upcycling

- 属于算法pipeline的实现是什么？实验比较什么？
  - 论文提出从预训练 dense checkpoint 初始化 MoE 模型的 **Sparse Upcycling** 算法 pipeline：
    1. **Upcycling Technique**：将 dense checkpoint 中指定 FFN 层的权重复制 N 次，初始化 MoE layer 的 N 个 expert（每个 expert 是原始 FFN 的完整副本），同时添加随机初始化的 router。其余权重（embedding、attention 等）直接从 dense checkpoint 复制。
    2. **Online Upcycling in NeMo**：在分布式训练框架 NeMo 中实现在线 upcycling。根据并行训练配置将 dense checkpoint 按设备分片（shard），各设备独立完成权重 upcycling，无需跨设备权重复制，解决因总参数量激增导致的内存超限问题。
    3. **MoE Parallel Folding**：提出异构混合并行策略，解耦 Attention 和 MoE 组件的并行映射。Attention 层使用 TP×CP×DP×PP 四维并行；MoE 层使用 Expert-TP×EP×Expert-DP×PP 四维并行。将 Attention 和 MoE 层中通信密集的并行操作折叠到 NVLink 高带宽域内，减少跨节点通信开销。
    4. **5-D Hybrid Parallelism**：基于 Megatron-Core，同时使用 Tensor Parallelism (TP)、Expert Parallelism (EP)、Pipeline Parallelism (PP)、Context Parallelism (CP)、Data Parallelism (DP with ZeRO-1) 五种并行策略。
    5. **Router Algorithm 选择**：对比 Mixtral-type router（KeepTopK→Softmax，确保 upcycling 后初始前向输出与 dense 模型一致）和 ST-type router（Softmax→KeepTopK），选择收敛更快的 Mixtral-type。
  - 实验比较：
    - Llama 3-8B Base vs Llama 3-8B E8T2 (upcycled 8-Expert Top-2 MoE)：MMLU (0-shot/5-shot), TruthfulQA, PIQA, SciQ, LogiQA, BoolQ, OpenBookQA
    - Capacity Factor (CF) 消融：CF=1, 2, 4, Dropless (无限 CF) 下的 MFU 和 MMLU 准确率
    - Base Model CT (Continued Training) vs upcycled MoE 的 MFU 和 MMLU
    - 不同并行配置 (TP/CP/EP/PP) 下的 TFLOPS/GPU 和 MFU
    - Router 类型：Mixtral-type vs ST-type 训练 loss 曲线对比

- 硬件平台是什么，配置是什么。
  - 主训练：512× NVIDIA H100 GPUs，使用 bfloat16 精度
  - MFU 调优实验：128× NVIDIA H100 GPUs（不同 TP/CP/EP/PP 配置），含 NVLink 域内通信
  - 训练框架：NeMo (https://github.com/NVIDIA/NeMo) + Megatron-Core (https://github.com/NVIDIA/Megatron-LM)
  - 分布式并行：5-D Hybrid Parallelism (TP+EP+PP+CP+DP with ZeRO-1)

- 模型是什么。数据集和bench分别是什么。
  - **Base 模型**：Llama 3-8B（Meta 预训练 dense checkpoint）
  - **Upcycled 模型**：Llama 3-E8T2（34.4B 总参数，11.8B 激活参数，8 Experts Top-2 routing，FLOPs 约 dense 的 1.6×）
  - **训练数据集**：
    - RedPajama V2（经 CCNet pipeline 按 n-gram perplexity 分桶，取最低 perplexity 桶，约 0.89T tokens）
    - Academic data blend（多种开源学术 benchmark 数据集混合，约 2.7B tokens）
    - 两源混合比例 7:3
  - **训练量**：100B tokens（主实验），27B tokens（CF 消融实验）
  - **Benchmarks**（使用 lm-evaluation-harness 评估）：
    - MMLU (5-shot & 0-shot), TruthfulQA (0-shot), PIQA (0-shot), SciQ (0-shot), LogiQA (0-shot), BoolQ (0-shot), OpenBookQA (0-shot)
  - **训练超参**：初始 LR=3e-5，余弦退火至 3e-7，100 warmup steps，主实验 CF=4, EP=8, TP=2, PP=4, VPP=8

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源**：NeMo (https://github.com/NVIDIA/NeMo) 中已集成 online upcycling 功能；Megatron-Core (https://github.com/NVIDIA/Megatron-LM) 提供 5-D 并行训练支持
  - **Upcycling 算法伪代码**：
```
# === Sparse Upcycling: Dense → MoE 初始化 ===
# 输入: dense checkpoint Θ_dense, 目标 expert 数 N, Top-K
# 输出: MoE checkpoint Θ_moe

def upcycle_dense_to_moe(Θ_dense, N, K, moe_layer_indices):
    Θ_moe = copy(Θ_dense)  # 复制所有非 MoE 权重

    for layer_idx in moe_layer_indices:
        # 1. 复制 FFN 权重 N 次初始化 experts
        W_orig = Θ_dense[layer_idx].ffn  # 原始 FFN 权重
        for n in range(N):
            Θ_moe[layer_idx].expert[n] = copy(W_orig)

        # 2. 随机初始化 router
        Θ_moe[layer_idx].router.W_g = random_init()
        Θ_moe[layer_idx].router.W_noise = random_init()

    return Θ_moe
```

  - **MoE Layer 前向传播（Mixtral-type router）**：
```
# 输入: x [B, S, d_model]
# Router:
H(x) = x @ W_g + StandardNormal() * Softplus(x @ W_noise)  # [B, S, N]
G(x) = Softmax(KeepTopK(H(x), K=2))                        # TopK 后 Softmax

# Expert FFN (每个 expert E_i 为 SiLU-gated FFN):
for each token with selected experts (i1, i2):
    y = G(x)_i1 * E_i1(x) + G(x)_i2 * E_i2(x)

# Expert capacity 控制:
expert_capacity = (tokens_per_batch / N) * CF
# 超出容量的 token 被跳过，直接传递到下一层
```

  - **MoE Parallel Folding 配置示例**：
```
# Attention layer: TP=2, CP=2, DP×PP
# MoE layer: EP=8, TP=1
# 效果: Attention 的 TP×CP group (4 GPUs) 折叠到 MoE 的 EP group (8 GPUs)
#        Attention 的 2×2 TP×CP 在单节点 8 GPU 内通过 NVLink 完成
#        避免跨节点通信开销扩大
```

  - **关键训练调优实践**：
    1. TP 和 EP 保持在 NVLink 域内以最小化延迟；MoE 层 EP 通常优于 TP
    2. AllToAll-based token dispatcher 对 TopK=1-4 更高效（vs AllGather-based）
    3. CP 配合 GQA 可重叠通信与计算，减小 KV 通信量
    4. 跨节点扩展用 PP+DP，VPP 减少 pipeline bubble
    5. 早期训练阶段对 MoE 层启用 recomputation，缓解负载不均导致的 OOM
