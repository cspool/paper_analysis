## Weight Averaging for Domain Generalization (SWA / SWAD / DiWA / EoA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight Averaging for DG 是一类通过平均多组模型权重提升 OOD 泛化的方法。主要变体：(1) SWA (Izmailov et al., UAI 2018)——平均训练轨迹上的多时刻权重；(2) SWAD (Cha et al., NeurIPS 2021)——在验证损失最优区间内做权重平均，DomainBed 最强单模型之一（Avg 66.9%）；(3) DiWA (Rame et al., NeurIPS 2022)——平均多个独立训练模型权重，需 60 个模型；(4) EoA (Arpit et al., NeurIPS 2022)——结合集成和权重平均。共同原理：权重平均平滑损失景观的尖锐区域，定位平坦极小值中的连通盆地中心。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SWAD 流程：训练中每 300 步验证 → 当 val_loss 达最优开始收集 checkpoints → 当 val_loss 超过最佳值 1.2× 停止 → 平均收集的所有权重。局限：需存储多份完整模型副本，最终模型仍是全精度。QT-DoG 通过量化噪声无需权重平均即可找到平坦极小值，模型还小 4.6×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 在线平均（SWA）—— w_swa = (n*w_swa + w_current)/(n+1)；(2) 离线平均（SWAD/DiWA）—— Σ w_i/N；(3) 加权平均（EoA）——基于验证性能分配权重。有效原因：平坦盆地中不同时刻/seed 模型权重在盆地不同位置，平均更接近中心最低点。

涉及论文标题：
- QT-DoG Quantization-Aware Training for Domain Generalization
