## Alpha-Beta (α-β) Performance Model

术语解释
α-β（alpha-beta）性能模型是分布式系统中一种简单且有效的线性执行时间预测模型，用于建模 GPU kernel（GEMM/Attention）和通信（NCCL send/recv）的执行时间随输入规模的变化关系。

术语是什么？
模型形式为 t(x) = α + β·x，其中 α 为固定启动开销（intercept，包括 kernel launch、memory management 等），β 为单位工作量的执行时间系数（slope），x 为 workload 规模（如 GEMM 的 FLOPs 数或通信的 bytes 数）。α 捕捉不随输入变化的开销，β 捕捉随输入线性增长的开销。此模型在 FSMoE、PipeMoE、MegaScale-Infer 等多个 MoE 系统中被验证有效，R² 通常 > 0.99。

从系统架构角度拆解：
在 FinDEP 中，α-β 模型用于三个关键组件的性能预测：
- GEMM: t_gm(x) = α_gm + β_gm·x（x 为 FLOPs，实测 α_gm=0.17ms, β_gm=8.59×10^{-11}）
- Attention: t_attn(y) = α_attn + β_attn·y（y 为 attention FLOPs，实测 α_attn=0.15ms, β_attn=1.54×10^{-11}）
- Communication: t_c(z) = α_c + β_c·z（z 为 bytes，实测 (ag=1,eg=7): α_a2e=0.10ms, β_a2e=9.61×10^{-7}）
通过这些线性模型的参数代入到分层的 timing constraints 中，FinDEP 能够快速预测任意 (ma, r1, me, r2, order) 配置的端到端推理时间，从而在 <1s 内搜索近似最优配置。

术语一般如何实现？如何使用？
采集方式：在目标硬件上运行 micro-benchmark——对 GEMM 测试 MLA 中所有矩阵大小配置，对 Attention 测试不同 seq_len 和 batch_size 的 attention 计算，对 Communication 测试不同 (ag, eg) 组合下的 NCCL send/recv 延迟。30 次运行（10 warm-up + 20 统计），全过程 <2 分钟。使用线性回归拟合 α 和 β，验证 R² > 0.99。在线阶段直接代入公式预测，无需重新测量。

涉及论文标题：
- Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism

---
