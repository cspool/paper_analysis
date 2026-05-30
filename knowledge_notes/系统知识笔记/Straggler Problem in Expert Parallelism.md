## Straggler Problem in Expert Parallelism

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Straggler Problem 是 MoE 训练中因 expert load 不均衡导致的 GPU 利用率下降：all-to-all 通信前后的同步要求所有 GPU 等待最重负载 GPU（straggler）完成 FFN 计算。每个 GPU 的 FFN 时间约与其收到的 token 数成正比。Expert load 分布高度动态且偏斜（training 初期严重），straggler 效应发生在**每个 micro-batch**。端到端训练吞吐由 max_gpu_load 决定而非 avg_gpu_load。

从系统架构角度拆解术语：
以 EP=4 为例：expert 0 为 hot expert（100 tokens），expert 1-3 各 50 tokens → expert 0 GPU FFN 时间 ~100·T_ffn，其他 ~50·T_ffn → 3 GPU idle ~50·T_ffn 等待 straggler → all-to-all combine 延迟。

术语一般如何实现？如何使用？
- 三种解决路径：(a) Algorithmic（load-balancing loss, token dropping）——可能损害精度；(b) Expert scheduling（FlexMoE/SmartMoE）——粒度粗、动态性差；(c) Token scheduling（FineMoE/LPLB）——细粒度 per-micro-batch 决策。
- FineMoE 将 straggler LP 化为 min max_gpu_load，s<1 时完全消除。

涉及论文标题：
- FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

---
