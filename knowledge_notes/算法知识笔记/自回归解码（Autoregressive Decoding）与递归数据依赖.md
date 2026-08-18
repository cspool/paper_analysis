## 自回归解码（Autoregressive Decoding）与递归数据依赖

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
自回归解码是因果语言模型（LLM）的生成方式：逐 token 生成，每个新 token 的预测以之前所有已生成 token 为条件（P(t_1..t_n)=Π P(t_i|t_<i)），因此每次迭代都等待上一次输出作为当前输入，形成递归数据依赖。这带来两种典型执行形态：prefill（并行处理整段输入序列，计算密集）与 decode（逐 token 迭代，访存/延迟敏感）。对部署系统（尤其 wafer-scale）的直接影响：解码的"每迭代依赖前输出"使 pipeline 化的层间数据流形成闭环——最后一个流水段（输出）必须把结果送回第一个流水段（输入），若两者物理距离远（如 ZigZag 映射下近直径路径），每次 decode 迭代都要承担长距离通信，成为端到端延迟的主要成分（BusyBarn 的 Fig.1 黄箭头）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
decode 迭代的计算-通信依赖（pipeline 视角）：
```
for step in 1..T:                        # T 个新 token
    h = embed(x_{step-1})                # 上一 token 输出作为输入（递归依赖）
    for layer_group g in 1..G:           # PP：die 组间串行
        h = Attn(g, h); h = FFN(g, h)    # 组内 TP/SP/CP 并行计算
        send(h, next_group)              # 组间传输激活（D2D 链路）
    x_step = sample(head(h))             # 最后一个 die 组产出 token
    send(x_step, group 1)                # ★ 回环：最后组→第一组（距离决定延迟）
```
自回归递归依赖使"最后一个 die 组与第一个 die 组"之间的通信距离成为关键指标：ZigZag 映射下该距离近 mesh 直径（Fig.5a/5c 虚线箭头），BusyBarn 的 Hamiltonian Loop 映射（Fig.5b/5d）把 die 组排成环使最后↔第一相邻，每次 decode 迭代的回环通信降到一跳邻居距离——这是 inter-die 映射直接由解码算法特性驱动的设计。评估中序列长度 512/2048/8192 覆盖 prefill 与 decode 两种形态（两者计算与通信模式差异显著）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：所有自回归 LLM 推理框架/硬件均按此逐 token 迭代执行（vLLM 等 serving 框架用 continuous batching 同时推进多条请求的 decode，见"Prefill/Decode"与"Continuous Batching"条目）；并行系统用 PP 把层切到多设备并承担回环通信。使用：BusyBarn 以 decode 的递归依赖为映射优化目标（Hamiltonian Loop）而非仅考虑 DNN 前馈数据流；其数据流执行允许 TP 部分和在单个 tile 完成后立即 reduce-scatter（vs bulk-synchronous 等整矩阵乘完），进一步隐藏通信（ablation 显示通信优化对端到端延迟的贡献大于单独映射改进）。局限：自回归本质限制批量并行度（decode 每步只推进一个 token），是 KV cache 与批处理优化（PagedAttention 等）的动机。

涉及论文标题：
- Mapping and Communication Optimizations with Fault Tolerance for Wafer-Scale LLM Inference
