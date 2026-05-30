## Failure Cost Model for MoE Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Failure Cost Model（故障开销模型）是 Tarragon 提出的量化单 worker 故障对 MoE 推理影响的数学模型。模型定义了两个核心指标：(1) **Inference Stall Time (T_stall)**：pipeline 无法产生新 token 的时长，等于 worker 重启时间 + 重放所有 prefill 层 + 重放已产生 decoding 层的时间；(2) **Re-execution Cost (G)**：以 GPU-time（执行时间 × GPU 数量）衡量的浪费计算量。模型以 decoded-token index i 和 frontier layer ℓ 为参数，区分三种场景：monolithic worker failure、decoupled AW failure、decoupled EW failure。核心发现：decoding 阶段 fault 的开销远超 prefill（64 tokens 解码后 ~19× 高于 128-token prefill），因此 decoding 是优化的主目标。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
模型公式（以 decoupled AW failure 为例）：

```
T_stall(ℓ, i) ≈ T_w + L · t_pre + [(i-1)L + ℓ] · t_dec

其中:
  T_w: worker 重启时间 (包含进程启动 + CUDA context init + 权重加载 + 通信栈初始化)
  t_pre: 单层 prefill 平均执行时间
  t_dec: 单层 decoding（单 token）平均执行时间
  L: transformer 层数
  i: 当前正在生成的 token index（1-indexed）
  ℓ: 故障发生时正在执行的 layer（1 ≤ ℓ ≤ L）

G(ℓ, i) ≈ M · [L · g_pre + ((i-1)L + ℓ) · g_dec]

其中:
  M: worker 总数
  g_pre: 单 worker 处理单层 prefill 的 GPU-time
  g_dec: 单 worker 处理单层 decoding 的 GPU-time
```

实测参数（Mixtral-8×7B, MegaScale-Infer 配置, 16 GPUs, 8 AWs + 8 EWs, GCP H200）：
- T_w = 18.5s, t_pre = 2.18ms, t_dec = 0.85ms
- g_pre = 0.006, g_dec = 0.0022

对于 EW failure（stateless）：
```
T_stall ≈ T_w + t_dec       // 仅 worker 重启 + 单层 expert 重算
G ≈ g_dec                   // 仅单 EW 的单层 expert 计算
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 用途：(a) 指导恢复策略设计——既然 decoding 开销远大于 prefill，Tarragon 重点优化 decoding 阶段 recovery（KV cache checkpointing）；(b) 指导 resource provisioning——EW 故障的 G 代价小可用 shadow expert 快速恢复，AW 故障需更重的 checkpointing 机制；(c) 推广到其他 MoE 系统——任一 worker 故障开销可由此模型参数化推广。
- 模型简化假设：忽略 warm cache 效应、通信/计算 overlap、straggler 影响；假设 worker 完全负载均衡。
- 与 Tarragon 设计的对应：D1（worker 级故障域）解决 M· 乘数；D2（self-healing）解决 T_w 等待；D3（KV cache checkpointing）解决 (i-1)L 重放开销。

涉及论文标题：
- Making MoE-based LLM Inference Resilient with Tarragon

---
