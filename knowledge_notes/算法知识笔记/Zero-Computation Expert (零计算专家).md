## Zero-Computation Expert (零计算专家)

术语是什么？
Zero-Computation Expert 指在推理阶段不执行任何矩阵乘法或激活函数计算的 expert。已有两种不同的实现路径：

**路径一（MoE++）：异构零计算专家。** 三类零参数的专家类型：(1) Zero Expert：输出恒为零 E_zero(x)=0，使 Top-2 退化为 Top-1；(2) Copy Expert：输出等于输入 E_copy(x)=x，相当于残差 shortcut；(3) Constant Expert：输出为 α1·x + α2·v，其中 [α1,α2]=Softmax(W_c·x)。三种专家参数极少（zero/copy 零参数，constant 仅 O(D) 参数），与标准 FFN 专家混合部署。

**路径二（MoLE）：重参数化零计算专家。** 训练时 expert 是标准 FFN，以 embedding tokens 为输入，所有 experts 同时激活。推理前将所有 expert 输出预计算为 Lookup Table (LUT)：LUT_l = {{v_j^i}_{j=1..N}}_{i=1..|V|}，其中 v_j^i = FFN_j(Embedding(i))。推理时 expert 计算被替换为 LUT lookup：仅按 input_ids 检索预计算的 v_j^i，然后通过 router 加权求和 h' = Σ_j g_j·v_j^i + FFN_shared(h) + h。Routed experts 的推理 FLOPs 从 4dND_r 降至 0（仅 lookup + weighted sum）。

**MoLE vs MoE++ 核心区别：** MoE++ 的 zero-computation expert 是设计时就确定的特殊类型（zero/copy/constant），训练和推理结构一致；MoLE 的 expert 在训练时是正常 FFN，通过 training-inference decoupling 和 reparameterization 在推理前转换为 LUT。MoLE 的 expert 总参数量远大于 MoE++（LUT size = dN|V|），但 per-token 加载量仅 dN。

从算法pipeline角度拆解术语：
**MoE++ 路径：**
```
selected_experts, probs = top_k_router(logits, k=2, capacities=C)
y = 0
for idx, p in zip(selected_experts, probs):
    if type[idx] == FFN:
        out = FFN(x)
    elif type[idx] == ZERO:
        out = torch.zeros_like(x)
    elif type[idx] == COPY:
        out = x  # identity
    elif type[idx] == CONST:
        alpha = softmax(W_c @ x)
        out = alpha[0] * x + alpha[1] * self.v
    y += p * out
```
计算复杂度：O(τ·N_FFN·T/(τ·N_FFN + N_ZC))。

**MoLE 路径（训练→重参数化→推理）：**
```
# 训练: Expert 接受 embedding tokens 输入，全激活
e = Embedding(input_ids)           # [b, s, d]
for j in 1..N:
    routed_output += g_j * FFN_j(e)  # g_j = SoftMax(Router(h))

# 重参数化 (推理前一次性):
for j in 1..N:
    V_j = FFN_j(W_emb)             # [|V|, d], W_emb = embedding 权重
LUT = stack([V_1, ..., V_N])       # [|V|, N, d]

# 推理: Expert 计算替换为 LUT lookup
v = LUT[input_ids]                 # [b, s, N, d] — 零 FLOPs
routed_output = Σ_j g_j * v[:,:,j,:]
```
MoLE 训练时 FFN_j 接受 embedding tokens e = Embedding(input_ids) 而非中间特征 h。因为 e 仅由离散 input_ids 决定，输入空间从连续 R^d 收缩为有限集 |V|（vocab size），使得 LUT 预计算成为可能。

术语一般如何实现？如何使用？
- MoE++ 代码：https://github.com/SkyworkAI/MoE-plus-plus（Apache 2.0，ICLR 2025），在 Megatron 中定义 FFN/ZERO/COPY/CONST 四种专家类型
- MoLE 代码：https://github.com/JieShibo/MoLE（ICML 2025），训练使用 modeling_mole.py（embedding as expert input + 全激活），推理使用 modeling_mole_rep.py（LUT lookup 替代 expert 计算）
- MoLE 的 LUT 可进一步压缩：NF4 量化将 LUT 从 3.5GB 降至 0.9GB，NF3 降至 0.7GB，性能几乎无损（Table 8）
- MoLE 关键 trade-off：LUT 存储开销大（dN|V|），但 per-token 传输量极小（dN），适合大容量存储设备 offloading

涉及论文标题：
- MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts
- Mixture of Lookup Experts
