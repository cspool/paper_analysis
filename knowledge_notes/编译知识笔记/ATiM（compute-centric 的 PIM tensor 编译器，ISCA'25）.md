## ATiM（compute-centric 的 PIM tensor 编译器，ISCA'25）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ATiM（"Autotuning Tensor Programs for Processing-in-DRAM"，SNU-CODElab，ISCA'25；arXiv:2412.19630；开源 github.com/SNU-CODElab/atim，Zenodo 10.5281/zenodo.15379924）是面向 UPMEM DRAM-PIM 的搜索式 tensor 编译器，基于 Apache TVM：联合搜索 host 侧数据分布与 DPU kernel 循环调度（复用 TVM schedule 原语 + PIM-aware 优化：DMA 边界检查消除、loop-bound tightening、invariant branch hoisting），在真实 UPMEM 上相对手调库最高 6.18×、GPT-J 层最高 8.21×。DCC（ISCA'26）对其关键批评：三步串行流程（先用 TVM 找定计算 schedule 模板集 → 再为每个模板生成数据重排 → 最后在固定模板集内搜索）是 compute-centric——先最小化计算时间、后补数据重排，把两个相互依赖的问题当独立问题顺序求解；且只支持 UPMEM（CPU-PIM），无法覆盖 GPU-PIM 异构 ML。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
DCC 用 TVM 式方案（T，公平复刻 ATiM）做动机实验：对 Reduction/GEMV 在 AttAcc 上按 TVM cost model 选最优计算模板，再对每个模板生成优化数据重排，取端到端最优者。结果：T 的平均端到端时间比手工最优（B）差 1.28×，数据重排占 kernel 时间 64.68%——即"看起来计算最优的 schedule 需要昂贵重排"；手工方案计算时间反而差 1.16×，但重排成本低 1.55×，端到端更优。结论：计算变换与数据重排必须联合 co-optimize，这正是 DCC data-first 调度的立论基础。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：TVM 的 autotuning 框架 + tensor IR lowering 扩展到 UPMEM（DPU 的 24 tasklets、IRAM/WRAM/MRAM、DMA）；演化搜索算法对扩展后的搜索空间做剪枝与早停。使用：给 UPMEM 上的 CNN/Transformer 层自动生成 host+DPU 程序。局限（DCC 指出）：固定模板集限制了搜索空间，漏掉"稍差计算换取便宜重排"的配置；CPU-PIM 场景对 ML 通常不如 GPU-only。DCC 的对应改进：先枚举数据分区再映射计算、联合预测器选优（见 Data-Centric Schedule Generation 与 Coupled Performance Predictor 条目）。

涉及论文标题：
- DCC: Data-Centric Compilation of Machine Learning Kernels for Processing-In-Memory Architectures
