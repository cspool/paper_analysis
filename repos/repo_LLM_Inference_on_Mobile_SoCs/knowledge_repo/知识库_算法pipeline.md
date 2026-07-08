## W4A16 Weight-Only Quantization（仅权重量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

W4A16 是一种 LLM 权重量化方案，其中权重（Weights）以 INT4（4-bit 整数）存储，激活值（Activations）保持为 FP16（16-bit 浮点）进行计算。这与同时量化权重和激活的 W4A4/W8A8 方案（如 llm.npu 使用的 INT8 混合精度、Qualcomm-AI 使用的 INT4/8）形成对比，后者在端侧小模型上可能导致显著的模型准确率损失（Qualcomm-AI 报告 20% 以上的精度退化）。

核心算术流程：模型权重 W（原始为 FP16）按 group（通常 group_size=128）量化——每组内的 FP16 权重值通过 scale factor（浮点缩放因子）和 zero point 映射到 INT4 范围 [-8, 7]。推理时从内存读取 INT4 权重 + scale factor → on-the-fly dequantize 恢复为 FP16 → 与 FP16 激活进行矩阵乘法。这比 FP16 存储减少约 75% 的权重内存占用（16 bit → 4 bit），而计算精度保持在 FP16。

量化方法包括 GPTQ（基于 Hessian 的逐列误差补偿）和 AWQ（基于激活幅度的显著通道保护），两者均为 post-training quantization (PTQ)，无需反向传播训练。HeteroInfer 论文未明确指定使用的具体 GPTQ/AWQ 变体，仅声明使用 W4A16 方案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 HeteroInfer 的 LLM 推理 pipeline 中，W4A16 的运作方式：

```python
# W4A16 权重量化在推理 pipeline 中的位置
# 阶段 0：离线量化（模型加载前）
def offline_quantization(model_weights_fp16):
    for layer in model.layers:
        for weight_matrix in layer.matrices:  # W [M, N]
            # 按 group_size=128 分组
            for g in range(0, N, group_size):
                w_group = weight_matrix[:, g:g+group_size]
                scale = max(abs(w_group)) / 7.0  # INT4 范围 [-8,7]
                w_int4 = round(w_group / scale).clip(-8, 7)
                store(w_int4, scale)  # 存储 INT4 权重 + FP16 scale
    # 存储结果: 权重占 4bit/元素, scale 占 16bit/group

# 阶段 1：在线推理 — Dequantize + Compute
def w4a16_matmul(activation_fp16, w_int4_stored, scales):
    # activation_fp16: [M, K] (FP16)
    # w_int4:        [K, N] (INT4 packed)
    # scales:        [N/group_size] (FP16)
    
    # Step 1: Dequantize 权重 INT4 → FP16
    w_fp16 = dequantize(w_int4_stored, scales)  # [K, N] FP16
    
    # Step 2: 标准 FP16 矩阵乘法
    output = matmul_fp16(activation_fp16, w_fp16)  # [M, N]
    return output
```

在 HeteroInfer 的 decoding 阶段，W4A16 的特殊处理（论文 §5.1 脚注 1）：
- Decoding 阶段仅使用 NPU 的 TOPS（INT8 计算能力），因为 NPU 当前不支持 W4A16 的 decoding 模式。
- Prefill 阶段 GPU 和 NPU 均可利用 W4A16 进行 FP16 计算。
- 这解释了为什么 decoding 阶段 GPU-dominant（GPU 对 W4A16 decoding 支持更好）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

**通用实现框架**：
- GPTQ (Frantar et al., 2022): `auto_gptq` 格式，按输入维度打包（K // pack_factor, N）的 INT4 排列。使用 Hessian 矩阵的逆进行逐列量化误差补偿。
- AWQ (Lin et al., 2023): `auto_awq` 格式，按输出维度打包（K, N // pack_factor）的非标准 nibble 顺序。通过激活幅度识别 ~1% 显著通道并对其进行 rescale 保护。
- GGUF (llama.cpp): 支持多种量化级别（Q4_0, Q4_K_M 等），在 llama.cpp/ollama 生态中广泛使用。

**HeteroInfer 的特化使用**：
- 权重以 INT4 存储以降低内存占用（端侧内存通常 8-24 GB，7B 模型 FP16 约 14 GB，W4A16 约 3.5 GB + KV Cache）。
- 所有计算以 FP16 进行，保证模型准确率不受量化影响（与 llm.npu/PowerInfer-2 的 INT 计算形成对比）。
- 论文明确声明"避免使用激活量化和稀疏技术，因为这些技术可能降低模型推理准确率"。

涉及论文标题：
- Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference

## LLM Prefill / Decoding Phase（LLM 预填充/解码阶段）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

LLM 推理分为两个物理特性截然不同的阶段：

**Prefill 阶段（预填充）**：一次性处理用户输入的所有 token（prompt），生成第一个输出 token。此阶段对所有输入 token 并行计算注意力，核心运算是矩阵-矩阵乘法（GEMM），属于计算密集型（compute-bound）。其延迟指标为 TTFT (Time to First Token)。在 HeteroInfer 论文的评估中，prefill 序列长度从 54（multi-turn dialogue）到 1787（long-text processing）不等。

**Decoding 阶段（解码）**：自回归地每次生成一个 token。每步仅计算新 token 对所有历史 token（包括 KV Cache 中的历史 token）的注意力，核心运算是矩阵-向量乘法（GEMV），属于内存密集型（memory-bound）。其延迟指标为 TPOT (Time per Output Token)。移动端 SoC 的有限内存带宽（Snapdragon 8 Gen 3: 68 GB/s 理论，~61.9 GB/s 实际最大）是 decoding 性能的主要瓶颈。

HeteroInfer 的核心设计决策源于两个阶段的不同瓶颈：prefill 阶段目标是最大化 SoC 计算吞吐（NPU-dominant，GPU 补充 NPU 性能退化场景），decoding 阶段目标是最大化内存带宽利用（GPU-dominant，GPU+NPU 并发使带宽从 ~43.3 GB/s 提升至 ~59.5 GB/s）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# LLM 推理两阶段 pipeline（以单个 decoder layer 为例）

# === Prefill Phase: compute-bound ===
def prefill_step(input_tokens, layer_weights, layer_kv_cache):
    # input_tokens: [seq_len, d_model]  (如 [320, 4096])
    
    # 1. Multi-Head Attention (并行计算所有 token)
    Q = matmul(input_tokens, W_q)  # GEMM [seq_len, d_model] x [d_model, d_head*n_heads]
    K = matmul(input_tokens, W_k)  # GEMM — 计算密集
    V = matmul(input_tokens, W_v)  # GEMM
    # 存储 K, V 到 KV Cache
    layer_kv_cache.store(K, V)
    # Attention: softmax(Q @ K^T / sqrt(d_head)) @ V
    attn_out = scaled_dot_product_attention(Q, K, V)  # [seq_len, d_model]
    
    # 2. FFN (Feed-Forward Network)
    ffn_up   = matmul(attn_out, W_up)     # GEMM [seq_len, d_model] x [d_model, d_ff]
    ffn_gate = matmul(attn_out, W_gate)   # GEMM — 主要计算量
    ffn_hidden = ffn_up * silu(ffn_gate)
    ffn_out = matmul(ffn_hidden, W_down)  # GEMM [seq_len, d_ff] x [d_ff, d_model]
    
    return ffn_out  # → next layer
    
# === Decoding Phase: memory-bound ===
def decode_step(new_token_embedding, layer_weights, layer_kv_cache):
    # new_token_embedding: [1, d_model]  (单 token)
    
    # 1. Multi-Head Attention (仅计算新 token)
    Q = matmul(new_token_embedding, W_q)  # GEMV [1, d_model] x [d_model, d_head*n_heads]
    K_new = matmul(new_token_embedding, W_k)  # GEMV — 内存密集
    V_new = matmul(new_token_embedding, W_v)  # GEMV
    # 追加到 KV Cache
    K = layer_kv_cache.append(K_new)  # [seq_len+1, d_head*n_heads]
    V = layer_kv_cache.append(V_new)
    # Attention with full KV Cache
    attn_out = scaled_dot_product_attention(Q, K, V)  # 内存带宽密集
    
    # 2. FFN (同上，但 seq_len=1)
    ffn_up   = matmul(attn_out, W_up)     # GEMV [1, d_model] x [d_model, d_ff]
    ffn_gate = matmul(attn_out, W_gate)   # GEMV
    ffn_hidden = ffn_up * silu(ffn_gate)
    ffn_out = matmul(ffn_hidden, W_down)  # GEMV [1, d_ff] x [d_ff, d_model]
    
    return ffn_out
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

**Prefill 优化策略**：
- Prefix Caching（vLLM, SGLang）：复用共享前缀的 KV Cache，避免重计算
- Chunked Prefill（llm.npu, Sarathi-Serve）：将长 prefill 序列拆分为多个 chunk，与 decoding 交错调度
- GPU-NPU 并行（HeteroInfer）：tensor-level 分区将 prefill 的 GEMM 拆分到 GPU 和 NPU 并行计算

**Decoding 优化策略**：
- KV Cache 量化：将 KV Cache 压缩为 INT8/INT4（vLLM FP8 KV Cache）
- Speculative Decoding：使用 draft model 预测多个 token 后并行验证
- 内存带宽聚合（HeteroInfer）：GPU+NPU 并发执行以最大化 SoC 内存带宽

**移动端特殊考量**：
- Decoding 阶段由于端侧单用户场景，batch size = 1，无 continuous batching 收益
- 内存带宽是 decoding 性能的硬上限（HeteroInfer 达到 96% 最大可用带宽后继续提升空间有限）
- CPU 不作为计算后端（能效比低），仅用于控制面（同步、kernel 调度）

涉及论文标题：
- Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference

## Activation Outliers in LLM（LLM 激活异常值）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Activation outliers（激活异常值）是 LLM 推理中某些激活通道（channel）的值显著大于其他通道的现象——通常超出典型量化范围（如 INT8 的 [-127, 128]）几个数量级。在 llm.npu 的实验中，Qwen1.5-1.8B 模型在 wikitext 数据集上推理时，仅 0.1%–0.3% 的通道在一次推理中产生异常值（约 5–15 个通道），但这些少数异常值对 per-tensor 量化造成灾难性精度损失：如果强制将所有值量化到 INT8 范围，异常值被截断（clipping），导致 >3.9% 的精度下降。

异常值的分布具有两个关键特性：(1) **通道偏斜（channel skew）**——虽然异常值在长 prompt 处理中可能出现在广泛位置（78% 的通道至少出现过一次），但出现频率高度偏斜：不到 3% 的"热通道"贡献了超过 80% 的异常值出现；(2) **逐层重要性差异**——靠近输入和输出的 layer 中异常值对精度影响更大（输入层受 token 差异影响波动大，输出层累积浅层误差），中间层异常值可以被剪枝而不显著影响精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 llm.npu 的量化流水线中，异常值的检测和处理过程：

```python
# 异常值检测（在线，每次推理时执行）
def detect_outliers(activation_fp16, quant_scale_s):
    """
    activation_fp16: [seq_len, hidden_dim]
    quant_scale_s:   量化尺度（离线确定）
    """
    # 量化后的理论值
    quantized_val = activation_fp16 / quant_scale_s
    
    # 异常值 = 超出 INT8 范围 [-127, 128] 的值
    outlier_mask = (abs(quantized_val) > 128)  # [seq_len, hidden_dim], boolean
    
    # 提取异常值所在通道
    outlier_channels = unique(outlier_mask.nonzero()[:, 1])  # ~5-15 个通道
    # 异常值仅占 0.1%-0.3% 总通道数
    
    return outlier_channels, outlier_mask

# 异常值的重要性度量（离线，一次完成）
def measure_outlier_importance(layer_activations_over_calibration_corpus):
    """
    对每个 layer l, 用校准语料统计异常值重要性
    """
    for layer_l in model.layers:
        # 收集该校准语料上该层的所有激活值
        all_acts = collect_activations(layer_l, calibration_corpus)
        quant_scale_s = max(abs(all_acts)) / 127  # 对称量化尺度
        
        # 对每个通道 c:
        for channel_c in range(hidden_dim):
            max_outlier_ch = max(abs(all_acts[:, channel_c]))
            # 重要性 = 最大异常值 / 量化尺度
            # 比值越大 → 分布越分散 → 量化误差越大 → 越重要
            importance[c] = max_outlier_ch / quant_scale_s
        
        # 该层重要性 = max_c(importance[c])
        layer_importance[layer_l] = max(importance)
    
    # 按重要性对所有层排序, 剪枝 top 85% 不重要的层
    sorted_layers = sort_by(layer_importance, ascending=True)
    pruned_layers = sorted_layers[:int(num_layers * 0.85)]
    # 仅保留 15% 层需要 shadow execution
    return pruned_layers
```

**Annotated Example**（Qwen1.5-1.8B, prompt_len=2048, wikitext）：
- 总 hidden_dim = 2048 channels/layer
- 单次推理中异常值通道: 5–15 个（0.24%–0.73%）
- 长期统计中 78% 的通道至少出现过一次异常值，但 <3% 的通道（~60 个）占据了 >80% 的异常值出现次数
- 输入附近 layer 的 importance 值: ~10–50× quant_scale（高度重要）
- 中间 layer 的 importance 值: ~1–2× quant_scale（可安全剪枝）
- 输出附近 layer 的 importance 值: ~5–20× quant_scale（高度重要）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

**LLM 量化中处理异常值的四种主流方法**：
1. **Per-group quantization (K-Quant, AWQ)**：将权重/激活按 group_size（如 128）分成多个子组，每组独立量化尺度——将异常值隔离在少数组内而不影响其他组。代价：NPU 上需将大 MatMul 拆分为多个子 MatMul + FP16 累加，性能损失 8.1–10.7×。
2. **SmoothQuant (per-tensor + smoothing)**：离线计算 per-channel smoothing factor，将激活的量化难度"迁移"到权重（对权重也施加 per-channel scaling），实现数学等价的 per-tensor 量化。代价：精度损失 3.9–8.4%（Qwen1.5-1.8B/LlaMA-2-7B on HellaSwag）。
3. **LLM.Int8() (mixed-precision)**：将异常值所在的列（columns）以 FP16 精度计算，其余列以 INT8 计算——直接在 MatMul 层面混合精度。代价：FP16 列的 GPU/NPU 利用率无法与 INT8 列重叠优化。
4. **Shadow Outlier Execution (llm.npu)**：per-tensor W8A8 的 INT8 MatMul 在 NPU 执行作为主路径 + 提取异常值通道在 CPU 以 FP16 执行影子 MatMul → 结果合并。代价：需维护热通道权重副本（~0.6-1% 额外内存）、CPU-NPU 同步开销（通过 85% 层剪枝消除）。

涉及论文标题：
- Fast On-device LLM Inference with NPUs

## Shadow Outlier Execution（影子异常值执行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Shadow Outlier Execution 是 llm.npu 提出的 NPU 友好型 per-tensor 量化 LLM 推理方法。核心思想是：将激活中超出 INT8 范围的异常值通道提取为紧凑子张量，在 CPU/GPU 上以 FP16 精度并行执行影子 MatMul，结果与 NPU 上 per-tensor INT8 MatMul 的主路径结果相加——既保持了 NPU per-tensor MatMul 的高效率（无需拆分为 per-group 子 MatMul），又通过 FP16 精度恢复了异常值导致的精度损失。

数学公式：
$$\frac{x}{s} \odot w = \operatorname{clamp}\left(\frac{x}{s}, -127, 128\right) \odot w \quad \text{(NPU INT8, 主路径)}$$
$$+ \operatorname{extract}\left(\left\lfloor\frac{x/s}{128}\right\rfloor \times 128\right) \odot w \quad \text{(CPU/GPU FP16, 影子路径)}$$

其中 $x$ 为 FP16 激活，$w$ 为 INT8 权重，$s$ 为量化尺度，$\odot$ 为矩阵乘法，$extract(\cdot)$ 为提取异常值通道并压缩为紧凑张量的函数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```python
# ========== Shadow Outlier Execution 完整算法 ==========

# 离线准备阶段 (preparation stage, 一次完成)
def preparation_stage(model_fp16, calibration_corpus, prune_rate=0.85):
    """
    输出: INT8 量化权重, 热通道索引, 剪枝层列表, 量化尺度
    """
    # 1. Per-tensor 对称量化
    for layer in model_fp16.layers:
        for matmul_op in [W_Q, W_K, W_V, W_O, W_gate, W_up, W_down]:
            s_w = max(abs(matmul_op.weight)) / 127
            matmul_op.W_INT8 = round(matmul_op.weight / s_w)  # clip to [-127, 127]
    
    # 2. 热通道识别
    hot_channels = {}  # 逐层
    for layer in model_fp16.layers:
        channel_outlier_count = zeros(hidden_dim)
        for sample in calibration_corpus:
            acts = forward_through_layer(layer, sample)
            for ch in range(hidden_dim):
                if max(abs(acts[:, ch])) / s_act[layer] > 128:
                    channel_outlier_count[ch] += 1
        # 保留频率最高的热通道 (覆盖 80% 异常值出现)
        sorted_ch = argsort(channel_outlier_count, descending=True)
        cumsum_ratio = cumsum(channel_outlier_count[sorted_ch]) / sum(channel_outlier_count)
        top_k = argmax(cumsum_ratio >= 0.8)
        hot_channels[layer] = sorted_ch[:top_k]  # 通常 <3% 通道
    
    # 3. 逐层重要性 + 剪枝决策
    layer_importance = measure_outlier_importance(...)  # 见上条术语
    num_prune = int(num_layers * prune_rate)
    pruned_layers = argsort(layer_importance)[:num_prune]
    
    # 4. 复制热通道权重到 CPU 内存
    cpu_weight_cache = {}
    for layer in model_fp16.layers:
        if layer not in pruned_layers:
            for ch in hot_channels[layer]:
                cpu_weight_cache[(layer, ch)] = W_FP16[:, ch]  # FP16 副本
    
    return quantized_model, hot_channels, pruned_layers, cpu_weight_cache

# 在线推理阶段 (对每次 MatMul 操作)
def shadow_matmul(x_fp16, W_INT8, s, layer_idx, hot_channels, pruned_layers, cpu_w_cache):
    """
    x_fp16:  [seq_len, hidden_dim]  FP16 激活
    W_INT8:  [hidden_dim, out_dim]   INT8 权重
    返回:    [seq_len, out_dim]       FP16 结果
    """
    # === NPU 主路径 (per-tensor INT8 MatMul) ===
    x_quant = round(clip(x_fp16 / s, -127, 128)).to_int8()  # 量化到 INT8
    y_npu = npu_matmul_int8(x_quant, W_INT8)                 # NPU INT8 MatMul
    y_npu_fp = y_npu.to_fp16() * s                           # 反量化
    
    # === 快速路径: 该层被剪枝 → 直接返回 NPU 结果 ===
    if layer_idx in pruned_layers:
        return y_npu_fp
    
    # === CPU 影子路径 (FP16 异常值补偿) ===
    # Step 1: 检测异常值通道
    outlier_mask = abs(x_fp16 / s) > 128  # [seq_len, hidden_dim]
    outlier_ch = unique(where(any(outlier_mask, axis=0)))  # ~5-15 个通道
    
    if len(outlier_ch) == 0:
        return y_npu_fp  # 无异常值, 跳过影子执行
    
    # Step 2: 提取异常值的超出部分为紧凑张量
    x_over = floor(x_fp16[:, outlier_ch] / s / 128) * 128  # 仅超出部分
    # x_over shape: [seq_len, ~10], 远小于原张量
    
    # Step 3: 获取影子权重 (热通道优先)
    W_shadow = []
    for ch in outlier_ch:
        if ch in hot_channels:
            W_shadow.append(cpu_w_cache[(layer_idx, ch)])  # CPU 内存直接取
        else:
            W_shadow.append(load_weight_from_disk(layer_idx, ch))  # 磁盘加载 (rare)
    W_shadow = stack(W_shadow, axis=0)  # [~10, out_dim], FP16
    
    # Step 4: CPU FP16 MatMul (影子执行)
    y_shadow = cpu_matmul_fp16(x_over, W_shadow)  # [seq_len, ~10] x [~10, out_dim]
    # 执行时间 << NPU INT8 MatMul (因为异常值通道极少)
    
    # Step 5: 结果合并
    return y_npu_fp + y_shadow  # FP16 向量加法, ~μs 级
```

**Annotated Execution Timeline**（Qwen1.5-1.8B, FFN Down [1024,2048] × [2048,8192]）:
```
NPU:  |====== INT8 MatMul (2.0ms) ======|
CPU:     |outlier detect| |shadow FP16 MatMul| |merge|
         |   <0.01ms    | |    <0.1ms        | |~μs |
Time: ──────────────────────────────────────────────→
        CPU shadow 执行完全被 NPU 主路径覆盖 (overlap)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

**llm.npu 中的具体实现**：
- 基于 max-min 对称量化 [Jacob et al., 2018] 作为 per-tensor 量化基础
- 该设计与任何 per-tensor 量化方法兼容（论文声称可替换为其他 per-tensor 方案）
- 热通道权重副本存储在 CPU 内存空间（与 NPU 共享的 unified memory 但需维护独立映射）
- 冷通道权重按需从磁盘加载，I/O 延迟与 NPU MatMul 执行重叠
- 实现于 MLLM 框架之上（C/C++ + 汇编），约 10K 行代码

**关键参数**：
- 默认异常值层剪枝率: 85%（基于离线 importance profiling）
- 热通道保留比例: ~3% 通道覆盖 >80% 异常值出现
- Shadow 执行内存开销: 总内存的 0.6%–1%（仅热通道 FP16 权重副本）
- 热通道优化减少 shadow 内存: 34.3%

**与其他方法的对比**：
| 方法 | NPU效率 | 精度 | 额外内存 | 同步开销 |
|------|---------|------|----------|----------|
| Per-group (K-Quant) | 低 (8-11× 慢) | 中 | 低 | 低 |
| SmoothQuant | 高 | 低 (-3.9~8.4%) | 低 | 低 |
| LLM.Int8() | 中 | 高 (<0.1% loss) | 中 | 中 |
| Shadow Outlier (0% prune) | 高 | 最高 (~1% loss) | 中 (0.6-1%) | 中高 |
| Shadow Outlier (85% prune) | **最高** | 高 (~1% loss) | **最低** | **最低** |

涉及论文标题：
- Fast On-device LLM Inference with NPUs

## Per-tensor vs Per-group Quantization for LLM（张量级 vs 分组级 LLM 量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

这是 LLM 量化推理中两种根本不同的激活量化粒度策略：

**Per-tensor quantization（张量级量化）**：整个激活张量使用一个全局量化尺度（scale factor）和零点（zero point）。INT8 范围 [-127, 128] 均匀覆盖整个张量的值域。优点：硬件友好——NPU 可直接执行单个大 MatMul（数据复用最大化），无需拆分为子操作。缺点：异常值通道的存在导致有效量化精度严重降低——量化步长由最大绝对值决定，大部分正常值被压缩到极少数 INT8 级别中。

**Per-group quantization（分组级量化）**：将激活和权重沿某维度分割为 group（典型 group_size=128），每组拥有独立的量化尺度。优点：异常值的影响被限制在其所在的 group 内，不污染其他 group 的量化精度。缺点：(a) 无法在 NPU 上直接计算——需将 MatMul 拆分为多个 group_size 的子 MatMul → 损失 NPU 的数据复用和并行度优势；(b) 子 MatMul 的中间结果需用 FP16 累加（不同 group 的 scale 不同，INT32 累加不足），引入大量 FP 操作。

两者的精度-效率权衡准确公式：
$$\text{Per-group MatMul} = \sum_{g=0}^{G-1} \left( s_g^{\text{act}} \cdot s_g^{\text{weight}} \right) \cdot \left( X_{\text{INT8}}^{(g)} \odot W_{\text{INT8}}^{(g)} \right)$$
其中 $G = N / \text{group\_size}$ 为组数，每组的 scale 因子 $s_g$ 不同使得无法合并为单次 MatMul。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# ===== Per-tensor MatMul (NPU 友好, 如 SmoothQuant, llm.npu 主路径) =====
def pertensor_matmul_on_npu(X_fp16, W_int8, s_x, s_w):
    """
    X_fp16:  [seq_len, hidden_dim]
    W_int8:  [hidden_dim, out_dim]
    s_x, s_w: 标量量化尺度
    """
    # 一次量化: 整个张量
    X_int8 = round(clip(X_fp16 / s_x, -127, 128)).to_int8()
    
    # 一次 NPU MatMul: 单个大操作, NPU 完全利用
    Y_int32 = npu_single_matmul(X_int8, W_int8)  # [seq_len, out_dim]
    
    # 一次反量化: 整个结果
    Y_fp16 = Y_int32.to_fp16() * s_x * s_w
    return Y_fp16


# ===== Per-group MatMul (精度优先, 如 K-Quant, AWQ) =====
def pergroup_matmul_on_npu(X_fp16, W_int8_grouped, s_x_groups, s_w_groups):
    """
    X_fp16:         [seq_len, hidden_dim]
    W_int8_grouped: [hidden_dim // group_size, group_size, out_dim]
    s_x_groups:     [hidden_dim // group_size]  每组独立 scale
    s_w_groups:     [hidden_dim // group_size, out_dim // group_size]
    """
    G = hidden_dim // group_size  # 组数, 如 hidden_dim=2048, group=128 → G=16
    Y_fp16 = zeros(seq_len, out_dim)
    
    for g in range(G):
        # 对每组: 独立量化
        X_int8_g = round(clip(
            X_fp16[:, g*group_size : (g+1)*group_size] / s_x_groups[g],
            -127, 128
        )).to_int8()  # [seq_len, group_size]
        
        # 对每组: 独立 NPU MatMul (小矩阵! NPU 低效)
        Y_int32_g = npu_matmul(X_int8_g, W_int8_grouped[g])  # [seq_len, out_dim]
        
        # 反量化 + FP16 累加 (额外 FP 操作)
        Y_fp16 += Y_int32_g.to_fp16() * s_x_groups[g] * s_w_groups[g]
    
    return Y_fp16
    # 问题: G 次小 MatMul + G 次 FP16 累加 → NPU 效率 8.1-10.7× 损失
```

**具体性能对比（llm.npu 论文 Table/Figure 数据）**：
- Per-tensor (SmoothQuant) NPU 执行: 1× 基准速度 → 精度损失 3.9-8.4% on HellaSwag
- Per-group (K-Quant) NPU 执行: 8.1-10.7× 慢于 per-tensor → 精度损失 <1%
- Per-tensor + Shadow Outlier (llm.npu): ~1× 基准速度（shadow 执行被重叠）+ 精度损失 ~1%

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

**在主流移动端推理框架中的使用**：
- llama.cpp (CPU): K-Quant per-group, group_size 根据 Q-level 变化（Q4_K_M 等），CPU SIMD 对子 MatMul 有一定加速
- llm.npu (NPU+CPU): per-tensor 主路径 + shadow outlier → NPU 保持 per-tensor 高效 + CPU 影子 FP16 补偿
- PowerInfer-V2 (NPU): per-tensor INT4 激活 + INT8 权重, 通过局部稀疏性控制精度损失
- MLC-LLM (GPU): 使用 FP16 而非 INT 量化以避免 per-group 在 GPU 上的复杂性
- HeteroInfer (GPU+NPU): W4A16 weight-only（激活保持 FP16），完全避开激活量化的精度-效率权衡

涉及论文标题：
- Fast On-device LLM Inference with NPUs
- Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

**来自 "Scaling LLM Test-Time Compute with Mobile NPU on Smartphones"（EUROSYS '26）的补充 — Tile-Group Quantization**：

该论文提出了一种新的 per-group 量化方案来解决 per-group 量化与 NPU 矩阵单元不兼容的问题。传统 per-group 量化（如 AWQ, group_size=128）在 column-major 权重上沿列维度连续 group → 反量化后元素在 NPU 的 HMX tile layout（32×32 tile，每两行 permute）中处于非连续位置 → 需要 scatter 写入 TCM（HVX 上极其昂贵）。直接 transpose 权重矩阵也不解决问题，因为 HMX 的多级 tile layout 导致 scatter 仍存在。

论文的 Tile-Group Quantization 方案（三阶段离线变换）：
1. **Pre-quantization permutation**：将 FP16 权重按 HMX tile layout 重新排列（外层 column-major tiles，内层每两行 cross-lane shuffle）
2. **Tile-group quantization**：在 permuted 布局上以 2×16 tile（32 elements, = group_size）为单位执行 round-to-nearest 量化
3. **Post-quantization super-group coalesce**：8 group → 1 super-group（256 INT4 = 128 bytes = 1 HVX 寄存器）

由于预训练权重近似零均值高斯分布，tile 内 reshuffle 不显著改变 group 内统计特性 → 量化误差与常规 group quantization 可比。量化后权重已在 HMX layout 中，运行时反量化结果可连续写入 TCM（消除 scatter），加速 9.65–19.04×。

涉及论文标题：
- Fast On-device LLM Inference with NPUs
- Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

## KV Cache（Key-Value Cache，键值缓存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache（Key-Value Cache）是 Transformer decoder 在自回归推理中用于避免重复计算的中间激活缓存。在 LLM 的自回归生成过程中，每个新 token 的预测需要与所有历史 token 进行 attention 计算。若每次迭代都对全部历史 token 重新计算 Key 和 Value 张量，计算复杂度为 $O(n^2)$（n 为序列长度）。KV Cache 的核心思想是将每层每个 head 的 Key 和 Value 张量缓存下来，后续迭代只需计算新 token 的 Q/K/V，然后将新 K/V 追加到缓存中，attention 计算仅需新 token 的 Q 与全部缓存的 K 进行点积。KV Cache 将 decode 阶段的每次迭代计算量从 $O(n^2)$ 降至 $O(n)$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

KV Cache 在 LLM 自回归推理 pipeline 中的运作流程：

```
# ========== LLM 自回归推理 with KV Cache ==========
# 输入: prompt tokens [t1, t2, ..., tP]
# 输出: 生成的 tokens [t_{P+1}, t_{P+2}, ..., t_{P+G}]

# 初始化: KV Cache 为空
K_cache = []  # 每层每 head 的 Key 缓存
V_cache = []  # 每层每 head 的 Value 缓存

# ===== Iteration 1: Prefill 阶段 (处理所有 prompt tokens) =====
for layer l in 0..L-1:
    # 输入 tokens 经 embedding + 前置层处理后得到 hidden states X [P, d_model]
    
    for head h in 0..H-1:
        # 计算 Q, K, V（一次性计算所有 P 个 token）
        Q_lh = X @ W_Q_lh  # [P, d_head]
        K_lh = X @ W_K_lh  # [P, d_head]
        V_lh = X @ W_V_lh  # [P, d_head]
        
        # 存入 KV Cache
        K_cache[l][h] = K_lh  # [P, d_head]
        V_cache[l][h] = V_lh  # [P, d_head]
        
        # Attention 计算
        attn_scores = softmax(Q_lh @ K_lh^T / sqrt(d_head))  # [P, P], 下三角 causal mask
        attn_out = attn_scores @ V_lh  # [P, d_head]
    
    # O 投影 + FFN → 输出 next layer input
    
# 取最后一个位置的 hidden state → 预测 first token t_{P+1}
output_token = argmax(lm_head(last_hidden_state))

# ===== Iteration 2+: Decode 阶段 (逐 token 生成) =====
while output_token != EOS:
    # 仅对当前 token 计算 embedding
    x = embed(output_token)  # [1, d_model]
    
    for layer l in 0..L-1:
        for head h in 0..H-1:
            # 仅计算新 token 的 Q, K, V
            Q_new = x @ W_Q_lh  # [1, d_head]
            K_new = x @ W_K_lh  # [1, d_head]
            V_new = x @ W_V_lh  # [1, d_head]
            
            # 追加到 KV Cache（关键：避免重计算历史 token）
            K_full = concat(K_cache[l][h], K_new)  # [seq_len+1, d_head]
            V_full = concat(V_cache[l][h], V_new)  # [seq_len+1, d_head]
            K_cache[l][h] = K_full
            V_cache[l][h] = V_full
            
            # Attention: 仅新 Q 与全部 K 点积
            attn_scores = softmax(Q_new @ K_full^T / sqrt(d_head))  # [1, seq_len+1]
            attn_out = attn_scores @ V_full  # [1, d_head]
        
        # O 投影 + FFN
    
    output_token = argmax(lm_head(last_hidden_state))
```

KV Cache 内存占用公式（以 Llama2-7B 为例）：

$$M_{KV} = 2 \times L \times H \times S \times d_{head} \times B$$

其中 $L=32$（层数），$H=32$（每层 head 数），$S$（序列长度），$d_{head}=128$（每 head 维度），$B=2$ bytes（FP16），系数 $2$ 表示 K 和 V 各一份。$S=4096$ 时：$M_{KV} = 2 \times 32 \times 32 \times 4096 \times 128 \times 2 = 2.147 \text{ GB}$。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

KV Cache 在主流 LLM 推理框架中的实现方式：
- **HuggingFace Transformers**: 使用 `past_key_values` 参数在每次 `forward()` 调用时传入并返回更新后的 cache。Cache 以 tuple of tuples 结构存储：`((K_L0_H0, V_L0_H0), (K_L0_H1, V_L0_H1), ..., (K_L31_H31, V_L31_H31))`。
- **vLLM (PagedAttention)**: 将 KV Cache 划分为固定大小的 block（如 16 tokens），使用类似 OS 虚拟内存的 page table 管理。Block 可以不连续分配，减少碎片化。
- **llama.cpp**: 在 GPU/CPU 显存中分配连续 buffer，按 token 维度追加。支持 FP16/INT8 量化 KV Cache 以减少内存。
- **LLMS（本论文）**: 在 chunk 粒度（16 tokens）管理 KV Cache，支持 chunk-wise 压缩和磁盘交换。通过 Pickle 序列化实现内存-磁盘之间的 chunk 传输。

KV Cache 的典型使用场景：
1. 多轮对话（Chatbot）：保存历史对话的 KV Cache，新消息仅需 prefill 新消息 + decode 响应，无需每次重建完整对话历史。
2. Prefix Caching：多个请求共享相同 system prompt 时，预计算并缓存公共前缀的 KV Cache，后续请求直接复用。
3. LLM Serving 的 Context Management：如本论文 LLMS，在移动端多 app 共享 LLM 场景下，管理多个 persistent context 的 KV Cache 压缩、交换和生命周期。

涉及论文标题：
- LLM as a System Service on Mobile Devices

## Tolerance-Aware KV Cache Compression（容忍度感知 KV Cache 压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Tolerance-Aware Compression 是 LLMS 论文提出的 chunk-wise KV Cache 差异化压缩方法。核心观察是：LLM context 中不同 token（及 token 组成的 chunk）对推理精度的贡献不均等——携带关键信息的 token（如名词、动词）比填充性 token（如 "and so on"）更难被压缩。传统 KV Cache 量化方法（如 SmoothQuant INT8）对所有 token 等量压缩，未利用这种不均等性。LLMS 的 Tolerance-Aware Compression 使用 attention scores 作为信息密度度量，为每个 chunk 分配不同的压缩级别（如 INT8/INT4/INT2），在保证全局平均压缩比的前提下最大化 context 的整体信息保留。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Tolerance-Aware Compression 的算法 Pipeline：

```
# ========== 输入 ==========
# KV Cache: C = {chunk_0, chunk_1, ..., chunk_{N-1}}
#   N = seq_len / chunk_size (e.g., 4096/16 = 256 chunks)
# Attention scores: A_lh [seq_len, seq_len] for each layer l and head h
# 全局平均压缩比: ratio_global (e.g., 0.50)
# 可用压缩级别: {ratio_w} = {1.0, 0.5, 0.25} (INT8/INT4/INT2)

# ========== Step 1: 计算 Chunk 信息密度 D_i ==========
for each chunk_i in 0..N-1:
    p = i * chunk_size
    q = (i + 1) * chunk_size - 1
    
    # 对 chunk 内每个 token 位置 col
    token_density_sum = 0
    for col in p..q:
        # 对每个 layer 和 head，计算该 token 被后续 token "关注" 的平均 attention
        layer_head_sum = 0
        for l in 0..L-1:
            for h in 0..H-1:
                # A_lh[row, col] = token_row 对 token_col 的 attention 权重
                # 行归一化: 每行 sum = 1.0
                col_attention = mean over rows > col of A_lh[row, col]
                layer_head_sum += col_attention
        token_density[col] = layer_head_sum / (L * H)
    
    # 聚合为 chunk-level 密度
    D_i = mean(token_density[p:q+1])

# ========== Step 2: 排序并分配压缩级别 ==========
Rank = percentile_rank({D_i})  # 0%~100%

# 优化问题 (Equation 3):
#   maximize: ctxInfo = Σ_w (1/ratio_w) * Σ_{chunk with ratio_w} D_i
#   s.t.:     Σ_w ratio_w * (σ_w - σ_{w+1}) = ratio_global
#   其中 σ_w 是使用压缩级别 w 的 chunk 的累计百分位边界

# 求解 (三种压缩级别时可以解析求解):
#   σ_{INT8} = top 30% (保持 INT8)
#   σ_{INT4} = middle 40% (从 INT8 压缩到 INT4)
#   σ_{INT2} = bottom 30% (从 INT8 压缩到 INT2)

# ========== Step 3: 执行 Chunk-wise 压缩 ==========
for each chunk_i in 0..N-1:
    rank = Rank[i]
    
    if rank > 0.70:  # top 30%
        # 保持 baseline INT8 量化结果
        chunk_i_compressed = chunk_i_INT8
        chunk_i_ratio = 1.0
    
    elif rank > 0.30:  # middle 40%
        # Channel-wise INT4 量化
        for each channel c in chunk_i:
            s_c = max(abs(chunk_i_INT8[c])) / 7  # INT4 scale
            chunk_i_compressed[c] = round(chunk_i_INT8[c] / s_c)
            chunk_i_compressed[c] = clip(chunk_i_compressed[c], -7, 7)
        chunk_i_ratio = 0.5
        
        # Bit-packing: 2x INT4 → 1x INT8
        packed[j] = (compressed[2j] & 0x0F) | ((compressed[2j+1] << 4) & 0xF0)
    
    else:  # bottom 30%
        # Channel-wise INT2 量化
        for each channel c in chunk_i:
            s_c = max(abs(chunk_i_INT8[c])) / 1  # INT2 scale
            chunk_i_compressed[c] = round(chunk_i_INT8[c] / s_c)
            chunk_i_compressed[c] = clip(chunk_i_compressed[c], -1, 1)
        chunk_i_ratio = 0.25
        
        # Bit-packing: 4x INT2 → 1x INT8

# 压缩后总内存:
# M_compressed = M_INT8 * Σ_i (chunk_i_ratio) / N
# 示例: 2048 MB * (0.3*1.0 + 0.4*0.5 + 0.3*0.25) = 2048 * 0.575 = 1178 MB

# ========== 解压 (推理时) ==========
# INT4 解包: unpack → 乘以 scale → 用于 attention 计算
# INT2 解包: unpack → 乘以 scale → 用于 attention 计算
```

信息密度的物理含义示例：
```
attention score matrix (简化版, 3 tokens "You are a"):
        You   are    a
  You   1.0   0     0      ← "You" 只看自己 (causal mask)
  are   0.2   0.8   0      ← "are" 20%关注"You", 80%关注自己
  a     0.3   0.5   0.2    ← "a" 30%→You, 50%→are, 20%→自己

token_density["You"] = (0.2+0.3)/2 = 0.25  ← 被后续 token 平均关注度
token_density["are"] = (0.5)/1    = 0.50  ← 最高
token_density["a"]   = (0)/0     = N/A    ← 无后续 token 关注

→ "are" 信息密度最高 → 压缩容忍度最低 → 保持高精度
→ "You" 信息密度中等 → 可适度压缩
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LLMS 中的实现方式（§4 Implementation）：
- 在 LLM 推理框架现有 KV cache 量化之上叠加。LLMS 原型中基线量化使用 LMDeploy 的 INT8 KV cache 量化方法。
- 对于 INT4 和 INT2 量化，使用 channel-wise 线性量化（每通道独立 scale）。
- 由于 LLM 推理框架仅原生支持 INT8，sub-byte（INT4/INT2）数据通过并行 bit-shift 操作打包为 INT8 格式：`packed[i] = (low_byte) | (high_byte << 4)`。
- 解压时 unpack 回 INT8 → 乘以各通道 scale → 用于 attention Score 计算（Q·K^T）。
- 信息密度计算利用前向推理中已产生的 attention scores（无需额外 forward pass），仅需对 attention scores 做列均值聚合——计算开销可忽略。

实现注意事项：
1. 压缩/解压仅在 context switching（chunk swap-in/out）和 token generation 完成（chunk writeback）时执行，不影响正常 LLM 推理的 decode 循环。
2. 三种压缩级别的阈值通过线性规划（简单微分即可求解）确定——对于三个压缩级别，仅需确定两个分割点。
3. 该方法与底层 KV cache 量化方法正交——可在 SmoothQuant/LMDeploy 等 INT8 量化之上工作，也可适配 4-bit 基线方法（提供 2-bit 和 1-bit 额外压缩）。

与静态量化的对比（WikiText-2, Llama2-7B）：
- 动态 INT4 全量: accuracy loss up to 59%（因 extreme outliers）
- 动态 INT2 全量: accuracy loss up to 99%（不可用）
- LLMS tolerance-aware (ratio_global=50%): accuracy ≈ INT8 baseline, memory ≈ INT4 static

涉及论文标题：
- LLM as a System Service on Mobile Devices

## Energy-Per-Token（能耗-每token）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Energy-Per-Token 是 FUSE 论文用于量化移动端 LLM 推理能效的核心指标，定义为每个 prefill token 或 decode token 消耗的设备总能量。计算公式：prefill 阶段 energy-per-token = $P_{\text{avg}} \cdot \text{TTFT} / N_p$（$P_{\text{avg}}$ 为 prefill 阶段平均功率，TTFT 为 time-to-first-token，$N_p$ 为 prompt token 数）；decode 阶段 energy-per-token = $P_{\text{avg}} \cdot \text{TPOT}$。该指标将延迟和能耗解耦——移动端低频运行虽减少瞬时功率但延长执行时间，总能量 $E = P \cdot t$ 可能增加或减少，energy-per-token 在同一维度上可比不同频率配置的能效。

论文实验中的关键发现：(1) GPU governor 在 decode 阶段选低频导致 energy-per-token 和 TPOT 同时恶化（低频省电被执行时间增加抵消）；(2) FUSE G1 在同 energy-per-token 下降低 TPOT 25-37%，G2 在同 TPOT 下降低 energy-per-token 7-10%。该指标依赖 Monsoon power monitor 每 0.2ms 细粒度功率采样计算。

涉及论文标题：
- Rethinking DVFS for Mobile LLMs: Unified Energy-Aware Scheduling with CORE

## Time-to-First-Token (TTFT) / Time-Per-Output-Token (TPOT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

TTFT (Time-to-First-Token) 和 TPOT (Time-Per-Output-Token) 是 LLM 自回归推理的两个标准延迟指标。TTFT 衡量 prefill 阶段从接收 prompt 到生成第一个 token 的时间——prefill 一次性处理所有 prompt token（batch processing），TTFT 随 prompt 长度超线性增长（attention $O(N^2)$）。TPOT 衡量 decode 阶段每生成一个 output token 的平均时间——decode 阶段 batch_size=1，TPOT 理论上恒定但受 KV cache 增长和硬件频率波动影响。

FUSE 论文使用两者作为两个独立优化目标——因为 prefill (compute-bound, GPU util ~83%) 和 decode (memory-bound, GPU util ~50-70%) 的计算特征和最优 DVFS 配置差异显著。论文固定 prefill={32,64,128,256,512} + decode=32 tokens 做受控频率搜索，在 ShareGPT trace (avg prefill 232.4 + decode 70.0 tokens) 上做端到端评估。FUSE 对 TTFT 改善 (7-17%) 小于 TPOT (25-37%)——prefill 的 GPU 利用率高使 default governor 已近最优，decode 低利用率使 governor 远离最优。

涉及论文标题：
- Rethinking DVFS for Mobile LLMs: Unified Energy-Aware Scheduling with CORE

## Test-Time Scaling（测试时计算扩展 / Parallel Test-Time Scaling）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Test-Time Scaling（也称 test-time compute scaling 或 inference-time scaling）是一种不修改模型参数、通过在推理时增加计算量来提升 LLM 输出质量的新范式。与传统的 model scaling（增大参数量 → 更好性能但更高资源消耗）不同，test-time scaling 在固定的模型上通过探索多条生成路径来换取更好的输出。核心直觉：对于有可验证结果的任务（如数学证明、编程），多尝试几次（parallel sampling）并选出最佳方案，比单次推理更可能产生正确结果。

两类主要的并行 test-time scaling 方法：
1. **Best-of-N（最优N选1）**：独立生成 N 条完整路径（每条 autoregressive 至 EOS），使用 Outcome Reward Model (ORM) 对每条完整路径打分，返回得分最高的路径
2. **Step-level Beam Search（逐步束搜索）**：在每步生成时保留 top-K 个候选路径（beam width），使用 Process Reward Model (PRM) 对部分序列打分，动态丢弃低质量路径

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Best-of-N Test-Time Scaling 在移动 NPU 上的 Pipeline
def best_of_n_inference_on_npu(model, prompt, N=8, reward_model):
    # Phase 1: Prefill (共享, batch=1)
    kv_cache = model.npu_prefill(prompt)

    # Phase 2: Parallel Decoding (batch=N, N 路径独立)
    # NPU HMX tile: [N→32 padded, 32] × [32, 32]
    # N=1: 1/32=3% tile util → N=8: 8/32=25% tile util
    paths = [[] for _ in range(N)]
    for step in range(max_new_tokens):
        batch_tokens = [p[-1] for p in paths]
        logits = model.npu_decode_batch(kv_cache, batch_tokens)
        # NPU GEMM ~same latency for N≤32 (HMX tile padding)
        for i in range(N):
            paths[i].append(sample(logits[i]))
            if paths[i][-1] == EOS: paths[i].done = True

    # Phase 3: Scoring & Selection (CPU, reward model)
    scores = [reward_model.score(prompt, p) for p in paths]
    return paths[argmax(scores)]

# 数学性质: 若单路径正确概率为 p, Best-of-N 正确概率为 1-(1-p)^N
# p=0.15, N=8 → 1-(0.85)^8 ≈ 0.73 (vs. 0.15 for single path)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **云侧实现**：OpenAI o1、DeepSeek-R1 等 reasoning model 使用 test-time scaling（chain-of-thought + majority voting/Best-of-N）；服务端有丰富的 GPU 并行算力
- **端侧实现（本论文贡献）**：将 test-time scaling 与移动 NPU 的闲置算力结合。LLM decoding 阶段 GEMM→GEMV → HMX tile 利用率仅 ~3%；增加 batch size（多条路径）填充 tile 空行 → 几乎零额外解码延迟
- **关键发现**：NPU batch=1 decoding ~5 tok/s（不如 GPU ~7 tok/s），但 NPU batch=8 ~22 tok/s（GPU 扩展性不如 NPU）→ NPU 在 test-time scaling 场景下有优势
- **Pareto frontier**：Qwen2.5-1.5B + Best-of-N(N=8) ≥ Qwen2.5-3B baseline accuracy；Qwen2.5-3B + Best-of-N ≥ Qwen2.5-7B baseline——小模型+test-time scaling 可匹敌甚至超越大模型
- **Reward model**：论文使用 Skywork-1.5B-PRM 作为 ORM（Best-of-N）和 PRM（Beam Search）

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

## Best-of-N Sampling（最优N选1采样）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Best-of-N 是最简单的并行 test-time scaling 方法：从 LLM 中独立采样 N 条完整生成路径（每条路径 autoregressive 至 EOS 或 max_len），使用 verifier（验证器，通常是 Outcome Reward Model）对每条路径评分，返回得分最高的路径。N 称为 generation budget（生成预算）。核心假设：对于有客观评价标准的任务（如数学题答案可自动验证），多尝试几次总能提高找到正确答案的概率。

与 majority voting（多数投票）的区别：Best-of-N 使用 reward model 而非投票计数来选择最优答案，适用于答案不重复或评分粒度更细的场景。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Best-of-N 概率分析
# 单路径正确概率 p → Best-of-N 正确概率 = 1-(1-p)^N
# N=1: p; N=4: 1-(1-p)^4; N=16: 1-(1-p)^16
# 例如 p=0.20: N=1→20%, N=4→59%, N=8→83%, N=16→97%

# 在移动 NPU 上的运行特性：
# - N 条路径共用 prompt KV cache (prefill 一次)
# - decoding batch=N → HMX tile [N→32 padded, 32]
#   N≤32: GEMM latency ≈ constant (tile padding)
#   非 GEMM 部分 (lm_head CPU, memory) 随 N 增长
# - batch=16 时 CPU lm_head 占比 ≥50% (主要瓶颈)
# - 能耗: 1.5B N=8 < 3B N=1 (更优的 accuracy-energy Pareto)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **适用场景**：有可自动验证结果的任务（数学推理、代码生成、定理证明）
- **Reward model**：Outcome Reward Model (ORM) 对完整结果打分，比 Process Reward Model (PRM) 简单（无需标注中间步骤）
- **N 的选择**：受 latency budget 和 generation diversity 影响。N 过小→质量提升不足，N 过大→收益递减+latency 增加
- **本论文的 Pareto 发现**：在相同的 latency cost 下，1.5B + Best-of-N(N=8) 在 MATH500 上的 accuracy 高于 3B + N=1 baseline

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

## Safe Softmax（数值稳定 Softmax）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Safe Softmax 是对标准 Softmax 的数值稳定实现。标准 Softmax $\text{softmax}(x_i) = e^{x_i} / \sum_j e^{x_j}$ 在 $x_i$ 很大时会导致 $e^{x_i}$ 上溢（FP16 最大值 ~65504）。Safe Softmax 通过减去输入向量的最大值来平移所有输入为非正值：
$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$
其中 $x_i - \max(x) \leq 0$，确保 $\exp$ 输入非正 → 输出在 $(0, 1]$ 范围内 → 无上溢风险。这在 FlashAttention 的 online softmax 中尤为重要——需要跨 tile 维护 running max $m_i$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Safe Softmax in FlashAttention (paper Algorithm 1)
# Online version with running statistics
def flash_attention_tile(Q_i, K_j, V_j, O_old, m_old, l_old):
    S = MatMul(Q_i, K_j^T, FP32 accum)      # [B_q, B_kv] FP16
    m_new = max(m_old, rowmax(S))            # [B_q] FP16, safe update
    S_safe = S - m_new                       # all ≤ 0 (safe softmax property)
    P = LUT_Exp(S_safe)                      # [B_q, B_kv] FP16
    # LUT exploits: input≤0 → sign bit=1 → ignore → only 32768 entries needed
    l_new = exp(m_old - m_new)*l_old + rowsum(P, FP32)
    O_new = diag(exp(m_old - m_new))*O_old + MatMul(P, V_j, FP32)
    return O_new, m_new, l_new
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **本论文的创新利用**：Safe Softmax 的输入 ≤0 性质使 LUT-based exp 的空间需求减半（32768 vs. 65536 FP16 entries, 64 KiB vs. 128 KiB），恰好适合 TCM 存储。vgather 地址 = (input & 0x7FFF) << 1，最大偏移 65536 bytes（vgather 的限制范围内）
- **通用框架**：PyTorch `F.softmax`、TensorFlow `tf.nn.softmax` 等默认使用 safe softmax；FlashAttention 的 online softmax 算法是 safe softmax 的在线版本
- **精度**：FP16 safe softmax 与 FP32 在 LLM 推理中通常无显著差异（论文 Table 5 确认 LUT-based FP16 Attention 与 F32 Attention 精度几乎相同）

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones
