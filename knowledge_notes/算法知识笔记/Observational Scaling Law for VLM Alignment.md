## Observational Scaling Law for VLM Alignment

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Observational scaling law for VLM alignment 是 Mordal 发现的经验规律：固定 VLM pretrained 模型参数，alignment 性能（下游任务 error）与对齐训练数据量存在 log-linear 关系——log(Error) ∝ log(data_ratio)。与标准 scaling laws (Kaplan et al. 2020, Hoffmann et al. 2022) 的区别：标准 law 研究 pretraining 时 compute/model/data scale 与 loss 的关系；observational scaling law (Ruan et al. 2024 NeurIPS Spotlight, Lin et al. 2024) 关注固定模型参数下从 observational data 预测不同 training scale 的最终性能。Mordal 将此概念首次应用于 VLM alignment 场景。

从算法pipeline角度拆解术语，给出具体例子。
Scaling Prediction (Algorithm 1)：
```
for each candidate c in remaining:
    P = []; r = 0.125  // start from 1/8 data
    while True:
        train_from_checkpoint(c, data_ratio=r)
        Err = evaluate(c, D_task)
        P.append((log(r), log(Err)))
        if len(P) > p (e.g., 3):
            f_c = LinearRegression(P)  // log(Err) = α·log(r) + β
            if fitting_loss(f_c) < δ (e.g., 5e-5): break
        r = r / u  // reduce data (u=2)
    predicted_err = exp(f_c(log(1)))  // predict at r=1 (full data)
select argmin(predicted_err)
```
关键发现（Figure 9）：log-linear 仅在**一定训练样本量后**出现（consistent with Ruan et al.）；不同 VLM 候选有**不同斜率**——解释不同组合的收敛速度差异。从大→小 ratio 递减以利用已有 checkpoint 节省计算。

术语一般如何实现？如何使用？
Mordal 中仅用于 intra-cluster evaluation（inter-cluster 阶段 speculative prediction 不可靠）。默认 p=3, δ=5e-5。其他相关工作：Ruan et al. (2024) 通过 PCA 从 80+ LLM 提取 latent capability 度量（PC-1 解释 ~80% 方差），证明 capability 与 compute 呈 log-linear (R²>0.9)，可预测 GPT-4 等未公开模型的性能。Mordal 的差异：关注单 VLM 在 alignment 过程中的 data scaling，而非跨模型 compute scaling。限制：仅在 7B 级模型验证；log-linear 关系需要一定数据阈值后才可观测。

涉及论文标题：
- Mordal: Automated Pretrained Model Selection for Vision Language Models

---
