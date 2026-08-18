## LBL / TBT 执行模式与时间步（Layer-by-Layer / Time-step-by-Time-step / Time-step）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SNN 推理有三个固有维度：时间步 T（synaptic transmission 发生、神经元积分激发一次的单位区间）、层 L、层内 spine/token 数 N。既有 SNN 加速器按遍历这三个维度的顺序分为两类执行模式：LBL（layer-by-layer）——先跑完某层全部 T×N 计算再进下一层，输出只在整网完成后出现，与弹性推理天然不兼容（代表：C-DNN、SpinalFlow、SASAP、Prosperity、Phi）；TBT（time-step-by-time-step）——每个时间步评估所有层，输出逐时间步涌现，可支持弹性推理（代表：TrueNorth、Darwin、MorphIC、PAICORE）。TBT 的问题在于其层间流水是粗粒度层级的：必须等一层全部 N 个 spine/token 缓冲并同步后才前传，完成的个体不能立即转发，因此首响应延迟仍被层内同步拖到 O(L×N)。
- ELSA 论文把时间步定义为"synaptic transmissions occur and neurons integrate inputs and generate spikes once"的离散区间；把 spine（CNN 的 Z^{1×1×C}）与 token（Transformer 的 Z^{1×D}）定义为流水粒度单位（Fig.4）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 两种执行模式的遍历顺序（T=时间步、L=层、N=spine/token）：
```
LBL:  for layer l in 1..L:            # 层外循环
        for t in 1..T:                 # 层内先跑满所有时间步
          for n in 1..N: compute(l, t, n)
      输出只在整网完成后可用（无弹性）
TBT:  for t in 1..T:                  # 时间步外循环
        for layer l in 1..L:
          for n in 1..N: compute(l, t, n)
          barrier: 等本层 N 个 spine 全部完成才进下一层   # 粗粒度层同步
      输出逐时间步涌现，但首响应被层内 barrier 延迟
ELSA: for t in 1..T:
        for l in 1..L:
          for n in 1..N:
            compute(l, t, n)  # spine/token 完成后立即前传下一层，无 barrier
```
- 例（ELSA Fig.5）：L=74、N=197 的 Spikeformer 中，TBT 的层内 barrier 使每个时间步都要等 197 个 token 集齐；ELSA 的 spine/token 级流水把"整层同步"替换为"每 token 完成后立即流入下一层"，首响应从 O(L×N) 降为 O(L)。Tab.XI 对比：Loihi/SpiNNaker/PAICORE 是 spike 级计算+层同步，ELSA 是 spine/token 级计算+PE 级时间推进。
- Annotations：barrier 指层内全 spine 完成才前传的同步点；ELSA 在每 PE 内独立推进时间步（PE-level time advance），是粒度最细的。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- LBL 实现：加速器跑满每层所有时间步，膜电位可在层间丢弃（C-DNN 因 LBL 而避免存膜，能效高但无弹性）。TBT 实现：每核存全部权重+膜+tracer（SRAM-only，如 TrueNorth 4096 核 + 1kHz 全局 tick、PAICORE 1024 核），层间用全局同步。ELSA 实现：6×6 神经核 + 2D-mesh NoC，每核 4 PE×128 ST-BIF 电路，spine/token 完成即由 Output Scheduler 调度前传，FIFO Queue 作核间 pipeline register。指标影响：Tab.IV 中 ELSA 以 spine/token 调度取得 4135.4 GOPS / 25.55 TOPS/W，相对 PAICORE 的 1.65× 速度增益即来自流水粒度（Fig.16/图 22-B 消融显示 spine/token 流水在 ResNet50 提速 6.7×、ViT-S 提速 15.2×）。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing
