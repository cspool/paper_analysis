## KernelBench（LLM 写 GPU kernel 的基准：250 题三级难度 / fast_p 指标）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- KernelBench 是 Stanford Scaling Intelligence Lab 的基准（Ouyang 等，arXiv:2502.10517，ICML 2025 海报，https://github.com/ScalingIntelligence/KernelBench），评估 LLM 把 PyTorch 参考实现翻译成自定义优化 GPU kernel（CUDA/Triton/CUTLASS/ThunderKittens 等）的能力。含 250 个来自真实深度学习负载的 PyTorch workload，按原语操作数分三级：Level 1（100 题，单原语：卷积、矩阵乘、loss、激活、layer norm）、Level 2（100 题，可融合算子序列如 conv+bias+ReLU）、Level 3（50 题，完整模型块如 AlexNet、MiniGPT）。核心指标 fast_p = 生成 kernel 既功能正确又对 PyTorch 基线加速超过阈值 p 的比例（fast_0=正确率、fast_1=正确且更快、fast_2=正确且 ≥2×）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 作为 KernelEvolve 的公开泛化性验证基准运转：输入 = KernelBench 250 题（每题的 PyTorch 参考 + 输入 shape）→ KernelEvolve 的图搜索+知识库流水线为每题生成 Triton kernel（原本面向 CUDA 的 benchmark，KernelEvolve 产出 Triton）→ 在目标硬件编译执行、与 torch.compile 的 PyTorch 参考做数值对比（torch.allclose 容差）与 speedup 测量 → 论文报告 250/250 = 100% pass（Level 1/2/3 全覆盖）。这验证了从单算子（L1）、融合模式（L2）到完整模型块（L3）的端到端正确生成能力。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：`git clone https://github.com/ScalingIntelligence/KernelBench` 后按 README 配置 L40S 等 NVIDIA 硬件，对每个 problem 提供 torch 参考实现与输入，调用 LLM 生成 kernel 后运行 eval 脚本计算 pass@k/fast_p。原论文发现前沿推理模型（o1、DeepSeek-R1、Claude 3.5 Sonnet）开箱仅 <20% 匹配 PyTorch 性能，但带执行/profile 反馈的迭代精修大幅提升（DeepSeek-R1 L1 从 ~12% 到 43%、L2 到 ~72%），这正是 KernelEvolve 图搜索+执行反馈设计的动机。KernelBench-v2（binary-husky）扩展为 Torch→Triton 问题集。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta
