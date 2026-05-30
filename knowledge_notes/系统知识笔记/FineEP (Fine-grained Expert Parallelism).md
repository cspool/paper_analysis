## FineEP (Fine-grained Expert Parallelism)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FineEP 是 FineMoE 提出的新型 Expert Parallelism 策略，通过合并 EP groups + shuffle expert placement + LP-based token scheduling 三步实现 per-micro-batch 细粒度负载均衡。参数 d（1 < d ≤ DP_degree/EP_degree）控制 FineEP group 大小：d 越大调度空间越大但通信组越大。FineEP 将传统 EP 的"token-GPU mapping 由 gate 固定"转变为"token 可在多个 GPU replica 间选择"。

从系统架构角度拆解术语：
FineEP 三步转换（图 3a→3b→3c）：
- 图 3a（vanilla EP）: expert 0 在 GPU {0,2}，无调度空间。
- 图 3b（合并 EP groups）: 利用 EDP 使 token 可在 EDP group 内多 replica 间选择，但仅实现 EDP group 内均衡。
- 图 3c（shuffle placement + token scheduling）: shuffle 使 expert 0 EDP={0,2}, expert 1 EDP={0,1}（交叉），LP token scheduling 实现全局 GPU 均衡。

术语一般如何实现？如何使用？
- 基于 Megatron-LM：Placement Manager（GPU 0）生成 placement → broadcast → Token Dispatcher（C++ LP + routing）。
- 与 FlexMoE 核心区别：FlexMoE 调度 unit=expert replica（粗粒度），FineEP 调度 unit=token（细粒度）。
- DeepSeek-V3 的 LPLB 采用类似 LP-based token scheduling 但缺少 FineEP 的 graph-theoretic placement theory。

涉及论文标题：
- FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

---
