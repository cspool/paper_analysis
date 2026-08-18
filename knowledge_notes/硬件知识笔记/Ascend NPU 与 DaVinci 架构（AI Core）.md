## Ascend NPU 与 DaVinci 架构（AI Core）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
华为昇腾（Ascend）AI 加速芯片采用的 DaVinci（达芬奇）架构（Hot Chips 2019 提出）：以 AI Core 为基本计算单元，每个 AI Core 内集成三类专用单元——Cube（AIC，矩阵乘加，INT8/FP16/BF16）、Vector（AIV，向量/SIMD 运算：Add/Exp/LayerNorm/Softmax 等）、Scalar（标量控制，循环/分支/寻址/指令分发）；AI Core 通过 Memory Transfer Engine（MTE1/2/3 + FixPipe）与全局内存（HBM/GM）和各级片上缓冲（UB/L1/L0A/L0B/L0C/BT/FP）交互。Ascend 910B2 包含 24 个 AI Core（24 Cube + 48 Vector，vector-to-cube 2:1），每 core 的 AIV 数据必须先经 MTE 载入 Unified Buffer（约 192KB）才能被向量单元操作。ENEC 论文强调的架构特征：①SIMD 向量执行、无条件分支、无 scatter/gather、无高效变长内存操作；②每 AI core 是单一重线程、无 CUDA 式轻量线程间同步，靠任务队列（EnQue/DeQue）驱动的流水数据流重叠搬运与计算。这些特征使传统无损压缩算法（ANS/Huffman/LZ77）与 Ascend"根本性不兼容"。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
ENEC 在 910B2 上的数据通路（压缩 kernel）：
```
HBM(权重) --MTE2--> UB(16384元素块) --AIV向量指令--> UB(打包结果)
                                                    |MTE3
                                     HBM(压缩流 + bit mask)
```
Annotations：AIV 只操作 UB 内数据（32 字节对齐）；压缩是向量逐元素运算，不能直接用 Cube（矩阵单元）——ENEC 因此让 AIV 跑压缩/解压、AIC 跑模型矩阵运算，构成协作推理流水线（逐层解压与当前层 forward 重叠）。MTE 在搬运时可做 in-flight 格式/类型转换。910B2 HBM 64GB 装不下 Qwen3-32B（61-65GB）+ 开销，触发 CPU offload 与权重传输瓶颈（占 prefill/decode 78-85% 执行时间）——这是 ENEC 无损压缩的硬件动机。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：DaVinci 架构芯片（910B2 等 Atlas 系列），驱动 + CANN 软件栈（AscendCL → AscendC/TBE 算子 → AI Core 指令）；编程用 AscendC（tensor/queue/pipe 抽象）或高层 torch_npu。使用：云端训练/推理（CloudMatrix384、Atlas 训练/推理系列）；ENEC 在 910B2 上实现无损压缩算子（csrc/，C++17 + CANN 8.2.RC1.alpha002），配合 HuggingFace Transformers 做端到端推理。性能特点：算力强（矩阵/张量核心）、HBM 容量与 NPU-CPU 带宽有限——正是无损压缩能填补的差距。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs
