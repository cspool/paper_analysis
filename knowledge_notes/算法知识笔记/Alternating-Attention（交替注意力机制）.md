## Alternating-Attention（交替注意力机制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Alternating-Attention 是 VGGT (Visual Geometry Grounded Transformer) 和 StreamVGGT 中使用的核心 Transformer 架构设计，由 Wang et al. (CVPR 2025) 提出。每层 Transformer block 中包含两种交替执行的 attention 操作：(1) **帧内空间自注意力（intra-frame spatial self-attention）** 对单帧内的所有 token（camera token + register tokens + patch tokens）执行标准 self-attention，捕获帧内空间结构；(2) **时序因果注意力（temporal causal attention）** 以当前帧 token 为 Query，跨所有历史帧的 token 为 Key/Value 执行 causal attention，聚合时序信息。

交替设计的动机：同时建模空间结构（同一帧内 patch 之间的关系）和时序动态（不同帧之间的运动/变化），而非将两种 attention 融合在一起（如 TimeSFormer 的 divided space-time attention），使每层可以独立优化空间和时序的 token 交互。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
StreamVGGT 中 Alternating-Attention 的计算流程（每层 ℓ）：

```
# 输入: Input_t^(ℓ-1) ∈ R^{(1+R+N)×C} (当前帧 token 序列)
#       Cache = {K_{1:t-1}^(ℓ), V_{1:t-1}^(ℓ)} (历史帧 KV cache)

# === Phase 1: 帧内空间自注意力 (Intra-frame Spatial Self-Attention) ===
Q_spatial = Input_t^(ℓ-1) · W_Q_spatial^(ℓ)
K_spatial = Input_t^(ℓ-1) · W_K_spatial^(ℓ)
V_spatial = Input_t^(ℓ-1) · W_V_spatial^(ℓ)
H_spatial = Softmax(Q_spatial · K_spatial^T / √d) · V_spatial
# 注：此处无 KV cache 参与，仅当前帧内 token 交互

# === Phase 2: 时序因果注意力 (Temporal Causal Attention) ===
Q_temporal = H_spatial · W_Q_temporal^(ℓ)     # 使用 spatial 输出
K_temporal = H_spatial · W_K_temporal^(ℓ)
V_temporal = H_spatial · W_V_temporal^(ℓ)

# 拼接历史 cache
K_all = concat(Cache.K_{1:t-1}^(ℓ), K_temporal)
V_all = concat(Cache.V_{1:t-1}^(ℓ), V_temporal)

# Causal attention（mask 防止访问未来帧）
Out_t^(ℓ) = FlashAttn(Q_temporal, K_all, V_all, causal_mask=True)

# 仅 temporal attention 维护 KV cache
Cache.K_{1:t}^(ℓ) ← K_all
Cache.V_{1:t}^(ℓ) ← V_all

# XStreamVGGT: 在此处对 temporal cache 执行 pruning + quantization
```

设计约束：仅 temporal global attention 模块维护 KV cache → 仅该部分受 cache growth 影响 → KV cache pruning 仅需处理 temporal attention 的 cache。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VGGT 基于 DINOv2 ViT-L backbone（24 layers），在其上构建 Alternating-Attention 架构。StreamVGGT 通过将 temporal attention 改为 frame-wise causal 版本适配流式推理。PyTorch 实现中 spatial attention 使用 `torch.nn.MultiheadAttention` 或 FlashAttention，temporal attention 额外传入 `attn_mask` (causal) 和 `past_key_value` (cache)。XStreamVGGT 在此基础上插入 pruning + quantization 逻辑。代码开源：VGGT (https://github.com/facebookresearch/vggt)，StreamVGGT (https://github.com/DongZhuo/StreamVGGT)。

涉及论文标题：
- XStreamVGGT__Extremely_Memory-Efficient_Streaming_Vision_Geometry_Grounded_Transformer_with_KV_Cache_Compression
