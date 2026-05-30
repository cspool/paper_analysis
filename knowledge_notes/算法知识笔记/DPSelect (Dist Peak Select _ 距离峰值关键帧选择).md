## DPSelect (Dist Peak Select / 距离峰值关键帧选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DPSelect (Dist Peak Select) 是 RETAKE 提出的 training-free 关键帧选择方法，用于在 VideoLLM 推理前减少视频帧间的时间冗余（temporal redundancy）。核心思想模仿人类视觉系统通过峰值刺激感知运动的机制：计算相邻帧的 token 平均余弦距离（token-averaged cosine distance），用 max pooling 识别距离的局部极大值帧作为 pivot frames（关键结构帧），再按距离值 top-k 补充剩余关键帧，最终将视频帧序列压缩到 alpha_dp 比例。与传统的均匀采样（uniform sampling）或简单 top-N 距离选择不同，DPSelect 通过峰值检测保留了视频中"变化最大的瞬间"（如场景切换、动作突发），这些帧被后续 PivotKV 模块标记为不可压缩的 pivot。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DPSelect 在视觉编码器输出后、LLM 输入前执行：
```
# 输入: M (T帧, 每帧N个visual tokens, d维)
# alpha_dp: 压缩比

# Step 1: 计算帧间 token 平均余弦距离
for i in range(T-1):
    d[i] = (1/N) * sum_{j=1..N} (1 - cos(M[i,j], M[i+1,j]))

# Step 2: Max pooling 识别 pivot frames (局部峰值, window=3)
P = {i | d[i] 是 [i-1, i, i+1] 窗口内的最大值}

# Step 3: Top-k 补充关键帧至目标压缩比
k = alpha_dp * T - |P|
K = P ∪ TopK({d[i] for i not in P}, k=k)

# Step 4: 提取压缩特征并标记 pivot mask
M_hat = Flatten(M[K, :, :])
S[j] = 1 if token j 源自 P 中的帧 else 0
```
RETAKE 中 w=3（max pooling 窗口），alpha_dp 按视频自适应设置以控制总 context length <= 32K。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DPSelect 完全基于 PyTorch 实现，无需训练或额外模型参数。输入为视觉编码器（如 QWen2VL 的 ViT）输出的 frame-level features，输出为压缩后的 token 序列和 binary pivot mask。与 PivotKV 解耦设计，DPSelect 可独立使用作为 keyframe selector（实验表明 DPSelect 本身在 256 帧限制下已优于 M2SM、MA-LLM 等 baseline）。代码开源在 https://github.com/SCZwangxiao/video-ReTaKe。DPSelect 的超参数 alpha_dp 与 PivotKV 的 alpha_kv 通过 trade-off 分析联合调优：固定总压缩比 0.25 时，alpha_dp/alpha_kv 在 2~3 之间取得最优性能（即更依赖知识冗余进行压缩）。

涉及论文标题：
- ReTaKe__Reducing_Temporal_and_Knowledge_Redundancy_for_Long_Video_Understanding
