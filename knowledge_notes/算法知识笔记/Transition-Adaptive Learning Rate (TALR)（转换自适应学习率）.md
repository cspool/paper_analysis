## Transition-Adaptive Learning Rate (TALR)（转换自适应学习率）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Transition-Adaptive Learning Rate（TALR，转换自适应学习率）是 Lee et al. 在 TR 调度中提出的自适应学习率机制，替换传统 QAT 中手动调度的 LR。核心公式：`U^t = max(0, U^{t-1} + η(R^t - K^t))`，其中 R^t 是目标 TR，K^t 是当前 running TR，η 控制更新幅度（设为初始 TALR 值）。这是一个负反馈控制回路：当 K^t < R^t（实际 transition 不足），U^t 增大，促使更多潜权重越过 transition point；当 K^t > R^t（transition 过多），U^t 减小，抑制 transition。与传统 LR 的关键区别：TALR 不是手动预设的 schedule，而是实时反馈控制——当潜权重向 transition point 聚集、transitions 自然容易发生时，K^t 自然升高，TALR 自动降低以抑制振荡。这使 TALR 隐式地考虑了潜权重的分布状态，实现自适应优化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TALR 的三种更新方案（论文 Sec. S3.2）：
```
方案1 — 加法更新 (Eq.11，论文默认):
U^t = max(0, U^{t-1} + η(R^t - K^t))
特点: 类似 SGD 权重更新，稳定可控，对 step scheduler 也鲁棒

方案2 — 乘法更新 (Eq.S1):
U^t = U^{t-1} · (R^t / K^t)
特点: 快速适应但敏感于 outlier，step scheduler 时不稳定（Fig. S5）

方案3 — 动量乘法更新 (Eq.S2):
U^t = m'·U^{t-1} + (1-m')·U^{t-1}·(R^t/K^t)
特点: 稳定性与速度的折中，m'=m=0.99，性能接近加法更新
```
TALR 在训练中的典型轨迹（Fig. 3b/S3b）：
- 初期：K^t << R^t → U^t 快速增大以鼓励 exploration
- 中期：U^t 随 R^t 的 cosine decay 逐渐下降
- 后期（~50K iter of CIFAR）：U^t 急剧下降趋近零——因为潜权重已聚集在 transition point 附近（平均距离很小，Fig. 3d），即使极小步长也触发大量 transition，TALR 自动"骤降"抑制振荡
对比 SGD 的 step-decay LR（Fig. S4b）：LR 在两个 decay 台阶之间固定不变，而 TALR 在每个台阶内部仍在单调递减，体现其自适应性。SGDT 在使用 step scheduler 时精度仅降 0.3-0.6%（ReActNet-18），而 SGD 降 0.7-9.1%（Table S3），验证了 TALR 对 scheduler 类型的鲁棒性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) TALR 初始值 = 同类优化器的初始 LR（SGD→1e-1, Adam→1e-3, AdamW→3e-4）；(2) 仅用于量化潜权重的更新，非量化参数仍用传统 LR；(3) TALR 更新发生在 optimizer.step() 中参数更新之前——先根据 K^t 和 R^t 计算 U^t，再用 U^t·g^t 更新 w；(4) TALR 是所有量化层共享的一个标量值（per-model，非 per-layer）；(5) 计算开销极低：仅 element-wise 比较（Eq.5）+ 标量 EMA（Eq.10）+ 标量更新（Eq.11），训练总时间仅增 2%；(6) TALR 与各种 optimizer（SGD, Adam, AdamW, NAdam, Adamax, RMSProp, Adagrad）和 scheduler（cosine/step/linear）兼容，均带来一致精度提升。注意：TALR 的设计不唯一——只要满足 (a) K^t<R^t 时 U^t 增大、(b) K^t>R^t 时 U^t 减小、(c) K^t=R^t 时 U^t 不变这三个准则即可。

涉及论文标题：
- Scheduling Weight Transitions for Quantization-Aware Training

---
