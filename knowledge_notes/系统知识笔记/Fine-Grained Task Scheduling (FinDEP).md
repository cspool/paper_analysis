## Fine-Grained Task Scheduling (FinDEP)

术语解释
FinDEP 是一种针对 DEP 的细粒度任务调度框架，通过两级 tensor 切分（r1 micro-batch + r2 fine-grained token）最大化 AG、EG、A2E、E2A 四类任务的重叠，提升 MoE 推理吞吐。

术语是什么？
FinDEP 在 PP-Pipe 的 micro-batch 级流水线基础上增加了：(1) r2 fine-grained pipeline：将每个 micro-batch 内的 token 进一步切分为 r2 个 segment，使 A2E/E2A 通信可与 expert GEMM 计算在更细粒度上重叠；(2) Shared Expert 感知调度：支持两种任务顺序——AASS（All Attention then All Shared）和 ASAS（Alternating Attention-Shared），根据 Shared Expert 开销自适应选择；(3) 数学优化框架：将调度参数形式化为优化问题，利用单调性（Theorem 1-3）和凸性（Theorem 4）约束搜索空间，Algorithm 1 在 O(√M) 时间内求解。

从系统架构角度拆解：
FinDEP 将一个 MoE layer 的执行展开为 r1 × r2 个可以流水线的子任务：
- AG 端：r1 个 pipeline stage，每个 stage 包含 Attention(ma) + Shared Expert(ma) + r2 次 A2E(me)
- EG 端：r1 × r2 个 fine-grained stage，每个包含 expert GEMM(E/eg × me) + E2A(me)
- 调度约束：τ_s 和 τ_a2e 可并行（无数据依赖）；τ_e 和 τ_e2a 可部分重叠；所有任务需满足资源互斥约束（同一设备不同时执行两个操作）
吞吐提升的关键在于：r2 增大使通信和计算在更小粒度上交替，隐藏更多通信延迟；但 r2 过度增大会因 kernel launch 开销反噬性能（α 项累积），需通过优化求解平衡点。

术语一般如何实现？如何使用？
FinDEP 通过 Algorithm 1 实现：(1) 对每个允许的 (ma, r1) 组合（满足 GPU memory 约束），分别评估 ASAS 和 AASS 两种顺序；(2) 对每种顺序，求解关于 1/r2 的凸优化问题（min Eq.17），获得最优 r2 和对应的 me；(3) 选择吞吐最高的配置。算法复杂度 O(C·√M)，求解时间 <1s，使 FinDEP 可在线自适应变长提示词。在 32×H20 DeepSeek-V2 S=4096 上 FinDEP 比最优配置的 PP-Pipe 快 1.24×。

涉及论文标题：
- Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism
