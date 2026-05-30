## GLaM: Efficient Scaling of Language Models with Mixture-of-Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - GLaM 提出基于稀疏激活 Mixture-of-Experts (MoE) 的 decoder-only 语言模型系列。核心实现：(1) 每隔一个 Transformer 层将标准 FFN 替换为 MoE 层（64 个 expert FFN），每 token 通过 top-2 softmax gating 仅激活 2 个 expert，加权组合输出；(2) 非 MoE 层使用 Gated Linear Unit (GLU) + GeGLU 替代标准 ReLU+Linear；(3) 使用 per-layer relative positional bias 替代绝对位置编码；(4) 通过 GSPMD 2D sharding 将 expert 权重 [E, M, H] 沿 E 和 H 维度划分、输入激活 [B, S, M] 沿 B 和 M 维度划分，实现无冗余并行；(5) 使用 GShard 辅助负载均衡损失（系数 0.01）鼓励 expert 均匀使用。最大 GLaM (64B/64E) 拥有 1.2T 总参数，每 token 仅激活 96.6B（8%），推理 FLOPs/Token 约 GPT-3 的 51.4%，训练能耗约 GPT-3 的 1/3。
  - 实验比较：(1) MoE vs Dense 对比：GLaM MoE 系列 vs 同等 nact-params 的 GLaM Dense 系列（0.1B→137B dense），在 29 个 NLP benchmark 上 zero/one/few-shot 性能；(2) vs GPT-3 (175B Dense)：最大 GLaM (64B/64E) vs GPT-3 在相同 benchmark 上的 zero/one/few-shot, FLOPs/token 和训练能耗对比；(3) Scaling 实验：expert 数量 scaling（1→256 experts）、data quality 消融（filtered vs unfiltered web data）；(4) Data Efficiency：不同 training token 量（up to 630B）下 MoE vs Dense 的 learning curve；(5) 参考对比：Gopher (280B), Megatron-NLG (530B), Switch-C (1.5T MoE)。

- 硬件平台是什么，配置是什么。
  - 训练硬件：1,024 块 Cloud TPU-V4 芯片（最大 GLaM 64B/64E 模型）。TPU-v4 单芯片实测系统功耗 326W。数据中心 PUE=1.11（训练期间）。
  - 训练框架：使用 GSPMD (Xu et al. 2021) 进行 2D sharding 模型并行，基于 Lingvo 框架 (Shen et al. 2019) 实现。
  - 网络拓扑：TPU 集群的 2D device mesh 拓扑，expert 沿 device 维度分布，同一 index 的 expert 跨不同 MoE 层放置于同一 device 上。

- 模型是什么。数据集和bench分别是什么。
  - 模型系列：GLaM (0.1B/64E)、GLaM (1.7B/32E, 64E, 128E, 256E)、GLaM (8B/64E)、GLaM (64B/64E) 以及对应 Dense 基线 GLaM (0.1B)、GLaM (1.7B)、GLaM (8B)、GLaM (137B)。关键架构参数：最大模型 L=64 layers, M=8192, H=32768 (MoE expert hidden), nheads=128, dhead=128。
  - 训练数据集：1.6T tokens，混合权重：Filtered Webpages 42% (143B tokens, Pareto 采样)、Books 20% (390B)、Conversations 28% (174B)、Forums 2% (247B)、Wikipedia 6% (3B)、News 2% (650B)。质量过滤使用 feature hash linear classifier 对 curated text vs webpages 分类后 Pareto 采样。
  - 评估 benchmark（29 个）：
    - NLG (8): TriviaQA, Natural Questions (NQS), Web Questions (WebQS), SQuADv2, LAMBADA, DROP, QuAC, CoQA。指标：Exact Match, F1。
    - NLU (21): HellaSwag, StoryCloze, Winograd, WinoGrande, PIQA, ARC-Easy, ARC-Challenge, OpenBookQA, BoolQ, COPA, RTE, WiC, MultiRC, WSC, ReCoRD, CB, ANLI R1/R2/R3, RACE-m, RACE-h。指标：Accuracy（除 MultiRC 用 F1a）。
  - 评估协议：zero-shot（直接评估）、one-shot（随机取 1 训练例作为 demonstration）、few-shot（每个 task 用 up to GPT-3 使用的 shot 数）。NLG 任务使用 beam search width=4。NLU 任务基于条件对数似然 log P(option|context) 长度归一化。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未公开模型权重或训练代码。GSPMD 2D sharding 算法论文开源（Xu et al. 2021, arXiv:2105.04663）。训练基于 Google 内部 Lingvo 框架（开源: https://github.com/tensorflow/lingvo）。GShard MoE 架构参考 Lepikhin et al. 2021（开源: https://github.com/google-research/google-research/tree/master/gshard）。
  - GLaM MoE 算法 pipeline 伪代码：

```python
# === GLaM Decoder-only MoE Transformer 前向传播 ===

# 输入 token_ids: [B, S], 经过 embedding + relative positional bias
x = embedding(token_ids)  # [B, S, M]

for layer in range(L):
    # --- Attention (非 MoE 层，标准 multi-head self-attention) ---
    attn_out = multi_head_self_attention(
        x, relative_positional_bias=per_layer_rel_bias[layer]
    )  # [B, S, M]
    x = x + attn_out  # residual

    if layer % 2 == 1:  # 每隔一层: MoE FFN 层
        # --- MoE Gating ---
        # gating_logits: [B, S, E], softmax over experts
        gating_logits = softmax(x @ W_gate)  # W_gate: [M, E]
        # top-2 gating: 选前 2 大 gate 值的 expert index
        gate_vals, expert_indices = top_k(gating_logits, k=2)  # [B, S, 2]
        # 归一化 top-k gate 值
        gate_vals = gate_vals / sum(gate_vals, dim=-1, keepdim=True)

        # --- Sparse Expert Computation ---
        moe_out = zeros_like(x)  # [B, S, M]
        for e in range(E):  # E=64 experts
            # 找到分配给 expert e 的所有 (batch, seq) 位置
            mask_e = (expert_indices == e).any(dim=-1)  # [B, S]
            if mask_e.sum() == 0: continue
            x_e = x[mask_e]  # [n_e, M]
            # Expert FFN: 两层线性 + GeGLU 激活
            # W_e1: [M, H], W_e2: [H, M]
            h = GeGLU(x_e @ W_e1)  # [n_e, H]
            out_e = h @ W_e2  # [n_e, M]
            # 按 gate 值加权（gate_vals 对应 expert e 的那一列）
            gate_e = gate_vals[mask_e][expert_indices[mask_e] == e]  # [n_e]
            moe_out[mask_e] += gate_e.unsqueeze(-1) * out_e

        x = x + moe_out  # MoE 输出 residual
    else:  # 非 MoE 层: 标准 GLU/GeGLU FFN
        # GLU: component-wise product of two linear transforms
        # gate = GeGLU(x @ W_g), value = x @ W_v
        ff_gate = GeGLU(x @ W_g)  # W_g: [M, H]
        ff_value = x @ W_v          # W_v: [M, H]
        ff_out = (ff_gate * ff_value) @ W_o  # W_o: [H, M]
        x = x + ff_out

# --- 负载均衡辅助损失 (训练时) ---
# 对每个 MoE 层计算 GShard auxiliary loss:
# L_aux = 0.01 * Σ_E (f_e * p_e), 
#   其中 f_e = fraction of tokens dispatched to expert e
#   p_e = average gating probability for expert e
L_total = L_cross_entropy + L_aux
```

```python
# === GSPMD 2D Sharding 张量划分 ===
# Expert 权重: 形状 [E, M, H] → 沿 E 和 H 维度 partition
#   device[{E_partition}, {H_partition}]
# 输入激活: 形状 [B, S, M] → 沿 B 和 M 维度 partition
#   device[{B_partition}, {M_partition}]
# 
# 同一 index expert 跨层放在同一 device:
#   device_i 持有 Layer_0/expert_i, Layer_2/expert_i, ...
# 
# while_loop 包装重复 MoE 层计算图以降低编译时间
```

- GLaM pipeline 关键张量流（inference 单 token）：
  input token → Embedding [1, M=8192] → GSPMD shard 到 device mesh → 64 层 Transformer，其中 32 层为 MoE 层（64 experts 分布在 N/E devices，top-2 softmax gating 选 2 experts，加权 sum）→ 32 层为 GLU/GeGLU FFN 层 → final LM head [M, V=256K] → output logits。

- 训练配置：Optimizer = Adafactor (β1=0, β2=0.99, update clipping=1.0, factored second-moment)。初始 LR=0.01 保持 10K steps 后 inverse sqrt 衰减。Max seq length=1024，每 batch 1M tokens。Dropout=0。float32 weights, bfloat16 activations。NaN/Inf 梯度检测 + 跳过更新 + checkpoint 回退恢复机制。SentencePiece tokenizer (vocab 256K)。
