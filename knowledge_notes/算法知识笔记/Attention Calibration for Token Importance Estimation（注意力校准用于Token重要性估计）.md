## Attention Calibration for Token Importance Estimation（注意力校准用于Token重要性估计）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention Calibration 是 VFlowOpt 提出的视觉 token 重要性估计增强技术。核心问题：使用 ViT 所有 token 的 attention 均值估计重要性时，冗余 token（如背景区域）会对同类冗余 token 分配不恰当的高 attention（attention bias），导致背景区域重要性被高估。Attention Calibration 通过两步纠正此偏差：(1) 计算全局 attention 阈值 τ = t · (1/N) Σ_i Σ_j A_{ij}，其中 t 为敏感度超参数，A_{ij} 为 ViT 层内 token i 对 token j 的 attention weight；(2) 筛选"相对重要"token 集合 K = {j | Σ_i A_{ij} > τ}，即接收总 attention 超过阈值的 token；(3) 仅用 K 中 token 的 attention 计算重要性 I_i = Σ_{k∈K} A_{ki} + α · softmax(H(V_i))。效果：排除冗余 token 的噪声 attention 信号，使重要性估计更可靠。消融实验表明移除 calibration 导致 MMStar 从 57.8→56.2、SQA 从 92.3→91.8（retain 25% tokens, LLaVA-OneVision-7B）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Attention Calibration in VFlowOpt
# Input: ViT attention matrix A ∈ R^{N×N}, sensitivity t, entropy weight α
# Output: importance scores I ∈ R^N

# Step 1: Compute global importance threshold
global_mean_attn = mean(sum(A, dim=1))           # 所有 token 接收的平均 attention
τ = t * global_mean_attn                           # 阈值

# Step 2: Identify relatively important tokens
attn_received = sum(A, dim=0)                      # ∈ R^N, 每个 token j 接收的总 attention
K = {j | attn_received[j] > τ}                     # 被高 attention 关注的 token 集合

# Step 3: Compute importance using ONLY calibrated attention + entropy
for i in 1..N:
    attn_from_K = sum(A[k, i] for k in K)         # K 中 token 对 token i 的 attention 之和
    H = compute_entropy(image_patch_i)              # 256 灰度级熵
    I_i = attn_from_K + α * softmax(H)             # 融合得分

# Contrast with uncalibrated (baseline):
# I_i_baseline = mean(sum(A[:, i])) + ...          # 包含冗余 token 的噪声 attention
```
Annotations: t 控制阈值高度——t 越大则 K 越小（越严格，仅保留最强受关注 token），t 越小则 K 越大（越宽松，接近无校准）。α 控制熵项贡献，由 Bayesian Optimization 自动搜索。与 VisionZip 的退化策略（无 [CLS] 时退化为 mean attention）相比，VFlowOpt 的 calibration 在任何 ViT 架构下均有效。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现不依赖 [CLS] token 或特定 ViT 架构——仅需从 ViT 最后一层的 attention matrix 中读取 A_{ij} 值（已在 ViT forward pass 中计算完成，Standard PyTorch hook 可捕获）。attention matrix 按 head 取平均后使用。对于使用 Flash Attention 的 ViT，需 output_attentions=True 以获取完整 attention matrix（增加 ~O(N²) 内存开销）。VFlowOpt 开源实现：https://github.com/sihany077/VFlowOpt，基于 LMMs-Eval + LLaVA-OneVision 框架。使用场景：任何依赖 attention 评估 token 重要性的训练无关剪枝方法均可用 calibration 改进——替换原有的 `mean(attn[:, j])` 为 `mean(attn[K, j])`。

涉及论文标题：
- VFlowOpt__A_Token_Pruning_Framework_for_LMMs_with_Visual_Information_Flow-Guided_Optimization
