## Spine/Token-wise Pipeline（细粒度流水与 Output Scheduler）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Spine/token-wise pipeline 是 ELSA 的核心执行模式：以 spine（CNN 中 Z^{1×1×C} 的输入向量）或 token（Transformer 中 Z^{1×D}）为流水粒度，每个完成的 spine/token 立即前传下一层，无需等本层全部 N 个 spine/token 完成（对比 TBT 的 layer-wise barrier）。它把首响应延迟从 O(L×N) 降到 O(L)，从而真正兑现弹性推理。配套的 ELSA Output Scheduler 决定"何时可前传"：对 CNN 用数据依赖判定（Algorithm 1），对 Transformer 用"同一 token 内依赖 + ssoftmax 需全部 QK token 就绪"的停顿规则。
- 为支持该流水，ELSA 采用层级数据管理：核内——部分和（膜电位）驻留膜 buffer（时间步间持久）；核间——输出 spike 打包成 flit 存 FIFO Queue（作为核间 pipeline register），实现非阻塞、保序传输。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- CNN spine 前传判定的伪代码（ELSA Algorithm 1 的核心，Output Scheduler 运行）：
```
输入: 核高 H_k、核宽 W_k、步长 S、填充 P、输入特征高宽 H_I/W_I、spine 位置 (i,j)
i ← i + P; j ← j + P
if i < j and (j-i+1) > H_k and i,j ≡ 0 (mod S):   # 数据依赖 spine 已到齐
    L ← L ∪ {(i/S, (j-W_k+1)/S)}                   # 右→左处理
if i > j and i > 0 and i,j ≡ 0 (mod S):            # 底→上处理
    L ← L ∪ {((i-H_k+1)/S, (j-W_k+1)/S)}
if i = P and j = P+W_I-1:                          # 最后 spine 到达→处理 padding
    L ← L ∪ {...边界输出 spine...}
```
- 例（ELSA Fig.13a）：层 3 的 spine S_1 依赖层 2 的 S_1,S_2,S_4,S_5；层 2 一完成这 4 个 spine，层 3 的 S_1 立即开算，不等层 2 全部 spine 完成。时间线显示层 3 首算比 layer-wise 方案早得多。Transformer 侧（Fig.13b）：token 内依赖使 spike 逐 token 处理，ssoftmax 需全部 QK token → 该处停顿。
- Annotations：H_k/W_k/S/P 是卷积参数；mod S 保证输出 spine 网格对齐；padding 的 spine 计算延迟到最后输入有效 spine 到达；FIFO 作为 pipeline register 使核间异步非阻塞。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：ELSA Output Scheduler（每核路由器内）按 Algorithm 1 调度；spine/token 完成即由控制模块触发下一层。效果（Fig.26/Fig.22-B）：相对无流水基线，spine/token 流水在 ResNet18/34/50/ViT-S 提速 2.2×/2.3×/2.8×/2.5×（Fig.26），相对 TBT 层流水（Fig.20）在同精度下 ViT-S 早 2.0×、ResNet50 快 2.4×；Fig.22-B 消融显示该技术单独贡献 ResNet50 6.7×、ViT-S 15.2× 速度（深度网络收益更大，因硬件利用率提升）。Tab.XI：ELSA 是唯一计算/通信粒度都为 spine/token、时间推进在 PE 级的加速器（Loihi/SpiNNaker/PAICORE 为 spike 级 + 层同步）。Fig.21 网络拥塞分析：注入率>0.04 时 NoC 拥塞、周期骤增，但早停 cycle 缩减仍稳定 >19%。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing
