## MoLE (Mixture of Lookup Experts)

术语是什么？
MoLE（Mixture of Lookup Experts）是一种训练-推理结构解耦的新型 MoE 架构。核心思想：训练时 expert 是标准 FFN，但以 embedding tokens（Embedding 层输出）而非中间特征为输入，且所有 experts 同时激活；推理前将所有 expert 输出预计算为 Lookup Table (LUT)，推理时 expert 计算被 LUT lookup 替代，实现零计算开销。MoLE 由 Jie et al. 在 ICML 2025 提出，代码开源：https://github.com/JieShibo/MoLE。

MoLE 的三个关键设计（对应 MoE 的三项缺陷）：
1. **Embedding as Expert Input**：将 routed expert 的输入从中间特征 h 改为 embedding tokens e = Embedding(input_ids)。e 仅由离散 input_ids 决定，输入空间从连续 R^d 收缩为有限集 |V|，使 LUT 预计算成为可能。代价：expert 无法直接访问上下文信息（由 shared expert 和 attention 层补偿）。
2. **全激活训练**：所有 N 个 routed experts 同时激活并接收梯度（不做 top-K 稀疏选择）。Router 输出全 N 维 SoftMax 权重。因为无稀疏性带来的 collapse 风险，无需 auxiliary loss，仅使用 LM cross-entropy loss。
3. **推理前重参数化**：训练后将每个 expert FFN_j 对 embedding 权重 W_emb ∈ R^{|V|×d} 做单次 forward pass，得到 LUT = {FFN_j(Embedding(i)) for j=1..N, i=1..|V|}。推理时 h' = Σ_j g_j·LUT[input_ids]_j + FFN_shared(h) + h。

从算法pipeline角度拆解术语：

**训练阶段（MoLE Decoder Layer）：**
```
输入: hidden_states (b,s,d), input_ids (b,s)
embedding_states = Embedding(input_ids)        # [b, s, d]

# 1. Self-Attention（标准）
h = RMSNorm(hidden_states)
h = Attention(h) + hidden_states

# 2. Shared Expert（接受中间特征，标准 SwiGLU FFN）
residual = h
h = RMSNorm(h)
shared_out = FFN_shared(h)                     # FLOPs: 4dD_s

# 3. Routed Experts（接受 embedding tokens, 全激活）
g = SoftMax(Router(h))                         # [b, s, N]
e = RMSNorm(embedding_states)
routed_out = Σ_{j=1}^N g_j * FFN_j(e)         # FLOPs: 4dND_r

# 4. 合并输出
h = residual + shared_out + routed_out
```
训练 FLOPs = 4d(D_s + ND_r)，包含所有 expert 计算。

**重参数化（训练后、推理前，一次性）：**
```
W_emb = Embedding.weight                        # [|V|, d]
for j in 1..N:
    V_j = FFN_j(W_emb)                          # [|V|, d]
LUT = {V_j}_{j=1..N}                            # size: N × |V| × d
```

**推理阶段（LUT lookup 替代 expert 计算）：**
```
# 1-2. Attention + Shared Expert：同训练
# 3. Routed "Experts" (零计算)
g = SoftMax(Router(h))                          # [b, s, N]
v = LUT[input_ids]                              # [b, s, N, d] — O(1) lookup
routed_out = Σ_j g_j * v[:,:,j,:]               # 仅加权求和, 零 FFN FLOPs
```
推理 FLOPs = 4dD_s（同 dense model）。Per-token 加载参数量：仅 dN（LUT lookup results）。

术语一般如何实现？如何使用？
- 开源：https://github.com/JieShibo/MoLE（ICML 2025），含 modeling_dense.py / modeling_moe.py / modeling_mole.py / modeling_mole_rep.py
- HuggingFace checkpoints: JieShibo/MoLE-{160M,410M}-{4E,16E}
- 训练配置：Pythia 架构、bf16、Adam(β1=0.9,β2=0.95)、100B Pile tokens、GPT-NeoX tokenizer (|V|=50k)、cosine LR decay
- 适用场景：VRAM 受限的推理部署，LUT 可 offload 到 CPU/disk，per-token 通信量可忽略（~KB 级 vs MoE 的 ~MB 级）
- 局限：(1) LUT 存储开销大（dN|V|），可通过 NF4/NF3 量化压缩至 20-25%；(2) expert 无法直接访问上下文信息；(3) 仅适用于有固定 vocabulary 的语言模型

涉及论文标题：
- Mixture of Lookup Experts
