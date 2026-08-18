## TorchInductor（编译器驱动融合与 Triton 代码生成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TorchInductor 是 torch.compile 的默认代码生成后端：把捕获的 FX 图（经 AOTAutograd 的完整前+反向图）做算子融合（含 pattern-based 的 select/pointwise/reduction 融合与编译器驱动 fusion）后，用 Triton 生成 GPU kernel（CPU 后端生成 C++/OpenMP）。MTIA 300（ISCA'26）以 TorchInductor 为代码生成与融合后端：支持 Triton 代码生成到 MTIA、手写 pattern-based fusion 与编译器驱动 fusion 并存，并让 collectives（HCCL 调用）与 compute 算子一起编译进同一张图。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
MTIA 300 中 TorchInductor 的运转流程：TorchDynamo/AOTAutograd 产出前+反向图 → Inductor 应用 MTIA 优化算子分解 → pattern-based 融合（手写规则匹配 GEMM+elementwise+reduction 模板）与编译器驱动融合 → Triton 代码生成 PE kernel（C++ 亦可）→ 与 HCCL 通信编成单图 → 图调度器（ILP 启发式）降低每迭代峰值内存 + activation rematerialization → 工作包下发 CPU-C。MTIA 还利用编码 agent（KernelEvolve [21]、Agentic Operator Generation [13]）自动化 kernel 生成，应对 DLRM 训练远多于推理的算子/形状面。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：TorchInductor 开源（PyTorch 仓库），MTIA 作为其新后端目标（Triton → MTIA dialect → LLVM-IR，见"Triton"条目）；训练优化（ILP 图调度、rematerialization）为 MTIA 编译器在 Inductor 之上的扩展。使用场景：DLRM 训练（150B 参数、TorchRec + TorchInductor 全图编译，与 H100 同栈对比）+ LLM 推理（DeepSeek-R1 编译）。信息缺口：论文未给出 Inductor 融合 pass 的具体清单与 kernel 数。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
