## 校准多模态近似计算（Calibrated Multimodal Approximate Computing，模型级冗余利用）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
模型级多模态冗余利用策略：多模态输入按更新频率（lifespan）分层——language token 整任务不变、vision token 跨多个去噪步不变、action token 每步都变（lifespan 最短），平均 91.7% 的输入每步不变却参与全部计算。直接跳过不变 token 的计算会引入两类错误：(1) attention-shift 误差：SoftMax 依赖全部 token 的全局比较，去掉冗余 token 后 Q/K 维度缩短、归一化分母变小，注意力分布整体漂移；(2) 迭代累积误差：DiT 多步迭代使单步近似误差累积到不可接受。因此采用"校准"：缓存冗余模态的 K 特征补全 SoftMax 分母、缓存 V 特征保持聚合对齐，并周期性重插完整去噪重置误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
计算流（论文 Fig.9–10）：
```
# 按 lifespan 识别不变 token
unchanged = vision_tokens_unchanged ∪ language_tokens_unchanged
K_cached = K[unchanged]            # 缓存冗余模态 K（跨步不变）
V_cached = V[unchanged]            # 缓存冗余模态 V
for each step:
    Q,K = project(updated_tokens)  # 仅更新 token 做投影
    score = Q @ [K; K_cached]^T    # K_cached 校准 SoftMax 输入：分母保持全 token 贡献
    attn = softmax(score / sqrt(d))
    out = attn @ [V; V_cached]     # V_cached 保持数据对齐
    # FFN 同样跳过不变 token 行
# action 模态列稀疏（约 52% 列近零）：
    zero_cols = where(all(|score[:, c]| < eps))
    skip softmax 输出零列与 V 对应行；相应 V 投影行旁路
# 每 20 个跳过迭代插入一次完整去噪，重置累积误差
```
张量视角：设 token 总数为 N、不变 token 为 M（≈91.7%），跳过后每步 GEMM 规模从 N 降至 N−M；attention 矩阵只对更新 token 行 × 全部 token 列计算，未更新行沿用上步结果。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：软件框架 S3 策略 + 硬件 multimodal scheduler（lifespan 序列生成器管理校准数据地址）。使用：消除 91.74% 的冗余 token 计算；消融中贡献总加速 32.60× 的主体（配合数据管理硬件避免 GPU 上 35.4% 的数据操作时延）；vision token 从 64 增至 512 仍保持 115.48Hz（跳过冗余历史视觉帧），language token 增长几乎无影响（指令整任务不变）。

涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence
