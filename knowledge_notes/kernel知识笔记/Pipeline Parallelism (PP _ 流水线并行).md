## Pipeline Parallelism (PP / 流水线并行)

术语是什么？
Pipeline Parallelism 是将模型按层（layer）切分为多个 stage，每个 stage 放在不同设备/节点上的模型并行技术。与 TP 在宽度维度切分不同，PP 在深度维度切分模型。前向时前一 stage 完成计算后将中间 hidden states 通过 p2p 通信发给下一 stage；反向时后一 stage 完成 backward 后将梯度发回。常用调度策略为 1F1B（one-forward-one-backward），在稳定阶段每个设备交替执行一个 forward 和一个 backward，通过 micro-batch pipeline 填充 bubble。

从kernel调度角度拆解术语：
PP 的 1F1B 调度（P stages, M micro-batches）：
```
Timeline (device 2 of 4-stage pipeline, M=8):
Warmup:  F0 F1 F2 F3
Steady:  F4 B0 F5 B1 F6 B2 F7 B3
Cooldown:         B4 B5 B6 B7
```
其中 F=forward, B=backward。Bubble 比例 = (P-1)/(P-1+M)，micro-batch 数 M 越大 bubble 越小。

在 PPMoE 中，PP 与 EP+TP 无缝集成——MoE 层的输入/输出格式和通信模式与非 MoE FFN 一致，dense 模型的 TP+PP 框架可直接通过替换部分 FFN 为 MoE 层转化为 PPMoE。

术语一般如何实现？如何使用？
Megatron-LM 的 `pipeline_model_parallel_size` 配置。PP 的通信为 p2p send/recv（节点间 InfiniBand 传输 b*s*h 数据），通信量远小于 TP 的 all-reduce 频率。PP 的缺点是 bubble overhead——bubble ratio = (P-1)/M，小模型更明显（PPMoE 小规模实验中 PP=4, M 较小时 bubble 显著）。适用场景：模型超出单节点内存时，PP 是必需的扩展方式。

涉及论文标题：
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism
