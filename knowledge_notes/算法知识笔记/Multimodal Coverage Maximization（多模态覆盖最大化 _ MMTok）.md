## Multimodal Coverage Maximization（多模态覆盖最大化 / MMTok）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multimodal Coverage Maximization 是 MMTok 提出的 training-free 视觉 token 选择方法。它将 token 选择建模为最大覆盖问题（Maximum Coverage Problem）：给定源 token 集合（n 个 vision tokens）和目标 token 集合（m 个 text tokens + n 个 vision tokens），从中选择 k 个源 token（k ≪ n）以最大化覆盖目标 token 的信息量。覆盖函数定义为子模函数 f(S; M) = (1/m) Σᵢ max_{j∈S} M_{i,j}，即对每个目标 token i，选择 S 中与之最相似的源 token j，相似度取 max，对所有目标 token 取平均。与现有 unimodal token pruning 方法（FastV 用 vision attention, SparseVLM 用 text attention, VisionZip 用 CLS attention, DivPrune 用 diversity）不同，MMTok 同时利用两个模态：text-vision（T-V）coverage 让选出的 vision token 覆盖文本查询语义；vision-vision（V-V）coverage 让选出的 vision token 覆盖全部图像信息。两者通过 softmax 温度校准后加权融合（α=0.5），贪心算法 O(kn) 获得 (1-1/e) 近似最优解。training-free，无需微调，实际开销极低（2880 tokens → 160 tokens 仅 6.4ms on A6000, 13.93 GFLOPs）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Multimodal Coverage Maximization (MMTok) Pipeline
# 输入: 图像 I, 文本查询 Q, 目标 token 数 k
# 超参数: τ_t=0.02, τ_v=0.2, α=0.5

# Step 1: Vision Encoder → vision tokens
V_raw = ViT(I)         # ∈ R^(n×d), 投影前
V_proj = MLP(V_raw)    # ∈ R^(n×d'), 投影后, 与 LLM 对齐

# Step 2: Text Embedding
T = Tokenizer(Q)       # m 个 text tokens
T_emb = Embed(T)       # ∈ R^(m×d')

# Step 3: L2 归一化
V_proj = L2_norm(V_proj, dim=-1)
V_raw  = L2_norm(V_raw, dim=-1)
T_emb  = L2_norm(T_emb, dim=-1)

# Step 4: 计算相似度矩阵
M_tv = T_emb @ V_proj.T    # ∈ R^(m×n), text-vision
M_vv = V_raw @ V_raw.T     # ∈ R^(n×n), vision-vision

# Step 5: Softmax 温度校准
M_tv' = softmax(M_tv / τ_t, dim=-1)   # 每行归一化
M_vv' = softmax(M_vv / τ_v, dim=-1)

# Step 6: 贪心覆盖选择
S = []
for i in range(k):
    best_s, best_gain = None, -inf
    for s in range(n) where s not in S:
        # 覆盖增量 = T-V 覆盖增益 + α × V-V 覆盖增益
        gain_tv = mean(max(M_tv'[t, S ∪ {s}]) for t in 1..m)
        gain_vv = mean(max(M_vv'[v, S ∪ {s}]) for v in 1..n)
        gain = gain_tv + α * gain_vv - current_f
        if gain > best_gain:
            best_s, best_gain = s, gain
    S.append(best_s)

# Step 7: LLM 推理
V_selected = V_proj[S]    # 仅保留选中的 k 个 vision token
input_llm = concat([V_selected, T_emb])
output = LLM(input_llm)
```

复杂度分析：
- M_tv 构建: O(mnd')
- M_vv 构建: O(n²d)
- 贪心选择: O(kn)，通过 incremental max 优化
- 总体开销: 2880 tokens 选 160 仅 6.4ms (A6000), 13.93 GFLOPs

与 unimodal baselines 的对比：
| 方法 | 模态 | 准则 | 理论保证 |
|------|------|------|----------|
| FastV | Vision-only | Attention ranking | 无 |
| SparseVLM | Text-only | Text→Vision attention | 无 |
| VisionZip | Vision-only | CLS attention ranking | 无 |
| DivPrune | Vision-only | Diversity maximization | 无 |
| **MMTok** | **Multimodal (T+V)** | **Coverage maximization** | **(1-1/e) 近似** |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MMTok 的开源实现：https://github.com/Ironieser/mmtok。使用方式：(1) 安装依赖后加载 VLM 模型（LLaVA-1.5, LLaVA-NeXT, Qwen2.5-VL 等）；(2) 在 vision encoder 输出后、LLM 输入前插入 MMTok token selection 模块；(3) 设置目标 token 数 k 和超参数 (τ_t=0.02, τ_v=0.2, α=0.5)，论文表明这些参数不敏感，可使用默认值；(4) 可选 MMTok++ 优化：排除 CLIP-ViT padding patches 并修复 overflow bug。实现依赖 PyTorch 内置操作（matmul, softmax, max），无需自定义 CUDA kernel。适用于所有采用 vision encoder + LLM 架构的 VLMs。对于 multi-turn conversation，V-V coverage 允许复用一次选出的 vision tokens 回答多个问题。

涉及论文标题：
- MMTok__Multimodal_Coverage_Maximization_for_Efficient_Inference_of_VLMs
