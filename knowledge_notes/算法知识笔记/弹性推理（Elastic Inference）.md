## 弹性推理（Elastic Inference）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 弹性推理是脉冲神经网络（SNN）独有的一种时序特性：由于 SNN 按离散时间步逐次积分-激发，输出（分类概率/检测框）随时间步推进逐步涌现、逐步收敛，因此对于"显著输入"（salient inputs），正确输出可比完整推理（跑满全部时间步）更早出现；给定足够推理时间，最终预测会收敛到与完整执行一致的结果。它对应生物神经系统的"显著刺激触发更快神经响应"现象。数学上，若神经元输出随累积输入单调收敛（如 ST-BIF 与量化 ReLU 等价），则早停输出是完整输出在时间上的前缀近似。ELSA 论文（ISCA 2026）把该特性视为 SNN 加速器此前未开发的关键机会：Fig.1 显示自主驾驶场景下首正确响应（FCR）可比稳定态输出早 82%。
- 别名/相关：early response、first-correct-response（FCR）、progressive inference、early exit（早停/提前退出是其在系统层的实现手段）。Web 证据显示同类工作包括 SIREN（熵基早退）、Elastic Spiking Transformers（NESTformer，运行时自适应弹性）、NEURAL（单时间步 + 早退的弹性神经形态架构）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在 SNN 推理 pipeline 中，弹性推理 = 逐时间步评估 + 每步后的置信度检查。执行例子（ELSA 论文，ResNet50，T=32）：
```
for t = 1..T:
    V_t = V_{t-1} + Σ_i x_{i,t}·w_i          # 积分（spike × weight 加法）
    y_t = Θ(V_t, V_thr, S_t)                 # 激发（ST-BIF 三元 {-1,0,1}）
    V_t = V_t - y_t·V_thr;  S_t = S_{t-1}+y_t  # 膜更新 + tracer
    p_max = max(class_probs(t))              # 分类头置信度
    if p_max ≥ threshold: break              # 提前终止，输出当前预测
```
- 关键点：早停的粒度可以是整网（层间必须同步的 TBT/LBL 架构只能整网早停），也可以是 spine/token 粒度（ELSA 的细粒度流水使每个 spine/token 可独立早停）。ELSA 测量：置信度阈值 0.55 时平均延迟减 21.9%、精度损失 <0.2%；COCO2017 检测用 objectness score，sweet point 0.2 时 match 率 94.9%、延迟减 45.4%。
- Annotations：T 是最大时间步（ELSA 全部 benchmark 用 32）；V_thr 激发阈值；S_t 为 spike tracer（见独立术语）；threshold 是提前终止置信度阈值（分类取最大类概率、检测取 objectness）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 算法侧：模型需在训练/转换时保证"时间累积输出收敛"，即用 ST-BIF 类神经元（与量化 ReLU 精确等价）构建 SNN，使每多跑一个时间步只是让量化输出更精细；推理时在分类头/检测头附加置信度函数（最大类概率 / objectness score）与阈值比较，超阈即停。ELSA 的 artifact（GitHub Intelligent-Computing-Research-Group/ELSA，ELSA_Algorithm 目录，PyTorch 2.4.1）直接输出逐时间步的 SNN 精度曲线与 FCR 延迟。系统/硬件侧：早停收益取决于执行模式——LBL 架构因层间全同步无法提前，TBT 只能整网早停，只有 spine/token 级细粒度流水（ELSA）能把早停粒度压到单个 spine/token，从而获得 Fig.20 中 2.0×（ViT-S）/2.4×（ResNet50）的同精度延迟缩短。Web 证据（SIREN）显示同类实现常用熵而非最大概率作为置信度，并加 patience 参数防止抖动误停。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing
