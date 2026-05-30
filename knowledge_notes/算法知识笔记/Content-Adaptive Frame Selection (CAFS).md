## Content-Adaptive Frame Selection (CAFS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Content-Adaptive Frame Selection (CAFS) 是 DIG 提出的基于视频语义内容的自适应代表性帧选择方法，用于替代 uniform sampling 或 FPS-based sampling 作为候选帧生成策略。CAFS 利用 DINOv2 自监督视觉特征捕捉视频中的高层语义变化（如场景切换、物体出现/消失），通过检测语义边界自适应地选择每个稳定段落的代表帧（r-frames）。关键创新：(1) 基于内容密度而非固定间隔选择帧——信息密集段产生更多 r-frames，静态冗余段产生更少；(2) 使用 topographic prominence >0.1 过滤噪声峰值，排除微小帧间波动；(3) 选择段落中点帧（而非峰值帧本身）作为代表——因峰值帧位于语义边界（混合两个场景），中点帧最能代表稳定语义内容。量化指标：LoC (Localized Coverage) 评估 r-frame 局部代表性，GIC (Global Coverage) 评估 r-frame 全局覆盖性。CAFS 的非线性信息缩放特性（Figure 10）：r-frame 数量不随视频时长线性增长，证明视频语义信息密度分布不均。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# CAFS 完整流程 (DIG Algorithm 1)
# 输入: 2 fps 采样 M 帧 {f_{I_i}}_{i=1}^M
# 输出: r-frame indices R_idx

# Step 1: DINOv2 特征提取
for i in 1..M:
    V_i = DINOv2(f_{I_i})              # 768-d (ViT-B)
# Step 2: 逐帧语义距离
for i in 1..M-1:
    d_i = 1 - cosine_sim(V_i, V_{i+1})  # scalar ∈ [0,2]
# Step 3: 峰值检测 (local maxima)
P = {i | d_{i-1} < d_i and d_i > d_{i+1}}
# Step 4: Topographic Prominence 过滤
P_valid = {j ∈ P | prominence(d_j) > 0.1}
# prominence: d_j - max(l_min, r_min)
#   l_min = 向左搜索到更高峰 min distance
#   r_min = 向右搜索到更高峰 min distance
# Step 5: 选相邻峰值中点
R_idx = {(I_p1 + I_p2)/2 | p1,p2 consecutive in P_valid}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CAFS 在 DIG 中作为 localized query pipeline 的第一步执行，使用冻结的 DINOv2 ViT-B 作为特征提取器。计算开销较低（MLVU: 25.9 min, LVB: 20.8 min on 8×A100）——无需大模型推理。与 uniform sampling 对比消融（Figure 7）：在 DIG 中用 uniform frames 替代 CAFS r-frames 后所有 benchmark 均下降且差距随帧数增加而扩大。r-frame 统计特性（Appendix E.2）：0-10 min 视频平均 47.9 r-frames，10-20 min 视频平均 226.4 r-frames（~4.22s/帧），实现高压缩比（~99% 的帧被压缩为代表帧）。

涉及论文标题：
- Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding
