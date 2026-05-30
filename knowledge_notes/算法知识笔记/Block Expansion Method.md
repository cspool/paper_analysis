## Block Expansion Method

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block Expansion Method 是一种将预训练模型扩展为更大模型的技术，由 LLaMA Pro（Wu et al., 2024, ACL 2024）首次系统提出。核心思想：在预训练 LLM 的 transformer block 序列中以交错方式（interleaved）插入新 block 的副本，将 output projection 层（o_proj 和 down_proj）零初始化使新 block 初态表现为恒等映射，然后仅训练新 block 或分阶段训练以注入新知识。这种方法通过复用预训练权重避免从头训练，同时交错插入使新容量分布于各抽象层级（而非仅堆在顶层或底层），比 LoRA 等参数高效方法保留了更完整的模型表达力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Block Expansion 在 RWKV-X 中的两阶段流程：
```
# Stage 0: Model Expansion (预训练 checkpoint → 扩展模型)
model_rwkv7 = load_checkpoint("RWKV-7")  # L layers
new_model = copy(model_rwkv7)
# 交错插入新层：将 L 层分为 N 组，每组后插入 1 个新 block
for group_idx in range(N):
    insert_pos = (group_idx + 1) * (L // N) + group_idx
    new_block = copy_block(model_rwkv7.layers[insert_pos - 1])
    # Zero-init output projections for identity mapping
    new_block.time_mixing.wkv_output.weight = 0
    new_block.channel_mixing.output.weight = 0
    # 对 Sparse Attention block: W_O 零初始化
    new_block.attn.W_O.weight = 0
    new_model.insert_layer(insert_pos, new_block)

# Stage 1: Alignment Pretraining (MiniPile, ctx=1024, 1.5B tokens)
# RWKV-7 blocks frozen, only new blocks trainable
for batch in MiniPile:
    loss = LongCE(new_model(batch))  # only new block params get gradients

# Stage 2: Long-context Continual Pretraining (ProLong-64K, ctx=64K, 1B tokens)
# All parameters unfrozen
new_model.unfreeze_all()
for batch in ProLong:
    loss = LongCE(new_model(batch))  # all params updated
```

参数配置（RWKV-X 3.6B）：L=32 original layers, 每 4 层插入 1 个 Sparse Attention block, 共 8 个新层, 总 40 层。Alignment phase: batch=1.024M tokens, ctx=4096, 4 GPU hours on H20。Long-context phase: batch=8.192M tokens, ctx=64K, 80 GPU hours on H200。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LLaMA Pro 开源：https://github.com/TencentARC/LLaMA-Pro。RWKV-X 基于此方法扩展至混合架构领域（RNN + Attention hybrid）。关键实现细节：(1) 只零初始化 output projection（o_proj/down_proj/W_O），不能零初始化 RMSNorm（会导致梯度完全消失）；(2) 采用交错插入而非堆叠于顶层——消融证明交错方式显著优于仅堆叠顶层或底层；(3) alignment stage 仅训练新 block，保留原始模型 general knowledge 防止 catastrophic forgetting；(4) 新 block 初始为原 block 的副本（非随机初始化），零初始化 output projection 使其恒等映射，training 过程中逐渐学习非零 output。适用于：将预训练 LLM/RWKV 模型扩展以注入新领域知识或增强特定能力（如长上下文），同时保留原始通用能力。

涉及论文标题：
- RWKV-X__A_Linear_Complexity_Hybrid_Language_Model

---
