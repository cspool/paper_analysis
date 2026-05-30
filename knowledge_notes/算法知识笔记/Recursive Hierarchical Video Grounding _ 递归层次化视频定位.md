## Recursive Hierarchical Video Grounding / 递归层次化视频定位

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Recursive Hierarchical Video Grounding 是一种受人类搜索策略启发的长视频时序定位方法，由 ReVisionLLM (CVPR 2025) 提出。核心思想：给定小时级长视频，模型首先在顶层用压缩的稀疏特征（sparse features）扫描全视频，识别大致的感兴趣区域（如5分钟段）；然后在中间层聚焦预测区域进一步细化；最终在底层用完整时间分辨率的密集特征（dense features）精确定位事件的秒级起止时间。每层 LLM 输出 "From s to e" 或 "Not Present."，上一层的预测边界作为下一层的输入上下文，逐层缩小搜索空间。形式化：对于 L 层层次结构，输入视频特征 I^(ℓ)，第 ℓ 层预测 τ^(ℓ)，条件于先前层次预测 τ^(<ℓ)。该递归结构使模型既能高效扫描小时级视频（使用稀疏特征，段级压缩比可达 250:1），又能精确定位秒级边界（使用密集特征，250帧全部保留）。与传统的 coarse-to-fine 方法（如 CONE 两阶段：候选生成→排序）的区别在于多层级递归和 LLM 内部置信度驱动的排序。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Recursive Hierarchical Video Grounding ===
F = CLIP_ViT_encode_CLS_only(V)  # (T, 768), 仅取CLS token
C = sliding_window(F, L_w=125s, stride=25s)  # MAD: |C| ≈ 100 segments
D = DenseFeatures(C)   # 底层: (|C|, 250, 4096)
S = SparseFeatures(C, Q)  # 上层: (|C|, 768), 250:1压缩

# Hierarchy ℓ=3 (顶层): 粗粒度扫描
τ_3 = LLM([S_top100, "when can we see <event> happening?"])
# → "From 5000s to 5300s" (分钟级精度)

# Hierarchy ℓ=2 (中层): 聚焦τ_3附近约33个段
τ_2 = LLM([S_focused33, "when can we see <event> happening?"])

# Hierarchy ℓ=1 (底层): 精确边界定位
τ_1 = LLM([D_selected, "when can we see <event> happening?"])
# → "From 5123s to 5126.5s" (秒级精度)

# 置信度排序: R_i = 1 / mean(entropy(LLM_output_probs))
τ_final = topk_by_confidence(τ_1_predictions, k=1)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
推理时视频按 sliding window 分段，每段 uniform 采样 250 帧。底层 hierarchy 使用 LoRA_A（Stage 1 训练），上层 hierarchies 使用 LoRA_B（Stage 2 训练）。对 MAD 数据集默认 3 层次，MAD segment L_w=125s/stride=25s，VidChapters-7M segment L_w=500s/stride=100s。消融：0层 R1@.1=0.0, 1层 R1@.1=8.4%, 3层 R1@.1=15.0%。代码: https://github.com/Tanveer81/ReVisionLLM。

涉及论文标题：
- ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos
