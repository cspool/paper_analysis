## Mamba / Structured State Space Model (SSM) / 结构化状态空间模型

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mamba 是一种基于 Structured State Space Model (SSM) 的序列模型架构（Gu and Dao, 2023），作为 Transformer attention 的替代方案，以 O(N) 线性复杂度处理序列。核心机制是 selective scan：输入序列通过 input-dependent 参数 (Δ, B, C) 将连续时间 SSM 离散化，逐 token 更新 hidden state，避免了 Transformer attention 的 O(N²) 计算和 KV cache。关键创新：(1) Input-dependent 参数化——Δ, B, C 由当前输入 x_t 通过线性投影生成，使模型具备选择性（根据输入内容决定保留/丢弃哪些信息）；(2) 硬件感知算法——通过 kernel fusion 和 parallel scan 在 GPU 上高效实现；(3) 无 KV cache——状态隐式编码为固定大小 hidden state。Mamba 在长序列上效率极高，但 In-Context Learning 和复杂检索能力弱于 Transformer attention。LongLLaVA 在 Hybrid 架构中 Mamba 层占比 7/8。

**Mamba-2 (Structured State Space Duality, SSD)**：Mamba-2 (Dao and Gu, 2024) 将 SSM 重新形式化为矩阵乘法，核心公式化简为 h_t = A_t·h_{t-1} + B_t·x_t，y_t = C_t^T·h_t，其中 A_t, B_t, C_t 为离散化参数。d_state 从 16 增至 128。TimeViper 在 hybrid backbone 中使用 27 层 Mamba-2，每层通过遗忘-记忆门控将历史序列信息编码入固定大小的隐式 hidden memory。Mamba-2 的 attention pattern 可通过 row-wise L1 normalized 累乘矩阵可视化，揭示其多样化的注意力模式——sparsity（选择性关注关键 token）、locality（邻域聚焦）、globality（全局均匀关注）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Mamba-2 SSM (TimeViper formulation)
# h_t = A_t * h_{t-1} + B_t * x_t
# y_t = C_t^T * h_t

def MambaBlock(x):  # x: [B,L,D]
    x_proj, z = split(Linear(x))              # each [B,L,D_inner]
    x_conv = SiLU(Conv1D(x_proj))             # local mixing
    B = Linear(x_conv); C = Linear(x_conv)    # [B,L,d_state]
    dt = softplus(Linear(x_conv) + dt_bias)   # [B,L,D_inner]
    A_bar = exp(dt ⊗ A)                       # discretize
    B_bar = dt ⊗ B
    h = selective_scan(A_bar, B_bar, x_conv)  # parallel scan O(N)
    y = (h @ C.T) * SiLU(z)                   # gated output
    return Linear(y)                           # [B,L,D]

# Mamba-2 attention pattern (for interpretability):
# y_i = sum_{j=1}^{i} C_i^T * (prod_{k=j+1}^{i} A_k) * B_j * x_j
# M'_{i,j} = |C_i^T * (prod_{k=j+1}^{i} A_k) * B_j|  -- "attention score"
```

Annotations: d_state = 16 (Mamba-1) or 128 (Mamba-2)；D_inner = D * expand_factor (2 or 4)；SiLU gate 类似 LSTM 输出门；selective scan 使用并行前缀扫描；Mamba-2 的"attention score" M'_{i,j} 非显式计算，而是从 SSM 参数推导得到，用于模型可解释性分析。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Mamba 官方实现 https://github.com/state-spaces/mamba，HuggingFace `MambaModel`。Mamba-2 (Dao and Gu, 2024) 提出 SSD (Structured State Space Duality)，将 SSM 形式化为矩阵乘法加速，d_state 增至 128。Falcon-mamba (7.3B) 是最大纯 Mamba LLM；Jamba 首次 hybrid Mamba-Transformer + MoE；Cobra 扩展 Mamba 到多模态 LLM。LongLLaVA 中 Mamba 层无 KV cache，使 hybrid architecture 在 100K tokens 时 Throughput 37.6 (2.6× vs Transformer)。

**Mamba 在长视频理解中的独特作用**：TimeViper 发现 Mamba-2 layers 通过遗忘-记忆机制隐式建模视频的时序位置信息——即使只使用 SigLIP 的 positional embedding（无 MRoPE 等显式时间戳建模），TVG 任务 mIoU 仍达 40.5，与显式使用 MRoPE 的 Qwen2.5-VL-7B (43.6) 差距不大。Mamba-2 的 O(1) KV-cache 使 TimeViper 在 32K input tokens (≈2K frames × 16 tokens/frame)、1K output tokens、batch_size=32 时，每秒生成 token 数比 Qwen3 高 40.1%。

涉及论文标题：
- LongLLaVA__Scaling_Multi-modal_LLMs_to_1000_Images_Efficiently_via_Hybrid_Architecture
- TimeViper__A_Hybrid_Mamba-Transformer_Vision-Language_Model_for_Efficient_Long_Video_Understanding
