## Accelerator Unit (AU) in CPU Pipeline

术语是什么？

Accelerator Unit (AU) 是现代通用CPU在执行流水线中集成的专用功能单元，用于加速特定矩阵运算。区别于CPU外的专用加速器（如Intel DSA独立于CPU core），AU嵌入在每物理核的流水线中，共享指令流、微架构资源和数据访问路径。代表包括：Intel AMX (矩阵乘)、ARM SME (Scalable Matrix Extension)、RISC-V M-extension。AU遵循SIMD范式，但instruction working set更小、i-cache miss更低（frontend bound从传统ALU的5%降至1%）。

从硬件架构角度拆解术语：

AU在CPU微架构中的独特性（AUM论文分析）：
1. **三层次资源特征差异**：(i) Usage Pattern：不同application/operator的AU使用率不同（AMX cycle ratio: prefill 14.4% vs decode 1.5%）；(ii) Frequency Interference：AU高功耗触发compulsory frequency reduction；(iii) Resource Bound：AU backend bound远高于frontend bind（prefill 92% backend, decode DRAM bound 59.9%）
2. **非SMT共享**：AU资源不跨hyperthread共享，同一物理核的两个SMT线程竞争AU，从根本上限制SMT-based sharing
3. **代际演进**：AU算力和数据类型持续增强（SPR BF16 206.4 TFLOPS→GNR FP16+FP8 344 TFLOPS），memory bandwidth同步提升（DDR5 233.8→MCR 600 GB/s）

术语一般如何实现？如何使用？

工业界AU使用现状：
- **Exclusive模式**（AWS/Azure/Inspur）：整个AU-enabled CPU独占给LLM serving→避免management complexity但40-50% AU核心闲置
- **Shared模式**（AUM提出）：AU CPU与通用workload共享→需处理三维AUV
- AU适用于CPU-only serving小模型（Phi-3 3.8B, Llama2 7B, Qwen3-A3B 30B MoE）或CPU-GPU hybrid部署

涉及论文标题：
- AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving

