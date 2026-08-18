## PCIe（PCI Express，CPU-GPU 互联总线 / 专家参数传输瓶颈）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PCIe（Peripheral Component Interconnect Express）是 CPU 与 GPU 之间的标准板级高速串行互联总线，负责 host 内存与设备显存之间的数据搬运（H2D 上传权重、D2H 回传结果）。在 MoE 边缘部署场景中，PCIe 是 expert offloading 的主通道：GPU 显存放不下全部专家时，CPU 内存中的专家权重经 PCIe 加载到 GPU，或用 CPU 直接计算后结果经 PCIe 回传。关键特征：带宽远低于 GPU 片内/NVLink 互联且随代际翻倍（PCIe 3.0 x16 单向约 16 GB/s、4.0 约 32 GB/s、5.0 约 64 GB/s、6.0 约 128 GB/s）；延迟与带宽是 MoE 解码的关键路径。SMoE 论文的核心量化（Fig.4，A6000）：专家加载的 PCIe 时间比 GPU 专家- token 计算慢 10–100×，low-score 专家加载占 TPOT 42%（表 I），因此减少 PCIe 传输量（专家替换+命中率提升）是降 TPOT 的主杠杆。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
PCIe 在 SMoE 硬件平台（A6000 48GB + PCIe 4.0 x16 + Intel Xeon Gold 6444Y）上的数据路径：
```
┌────────────┐        PCIe 4.0 x16 (~32 GB/s)       ┌──────────────┐
│  CPU DRAM  │ ── H2D: 预取层 i+1 top-score 专家 ──▶ │  A6000 GPU   │
│  (150GB,    │                                      │  (48GB HBM,  │
│  全部专家)   │ ◀── D2H: CPU 计算的专家结果 ──────── │  共享专家常驻) │
└────────────┘                                      └──────────────┘
```
执行流程：①GPU 上共享专家+缓存专家预测下一层 top-score 专家 → ②CPU 侧 expert-cache router 决策替换（减少需传输量）→ ③PCIe 异步预取（与当前层 GPU 计算重叠）→ ④未命中专家按 CPU-assisted 调度分派：高分数走 H2D 加载、低分数留 CPU 计算（结果 D2H 回传）→ ⑤GPU 合并。带宽示例：单 expert（如 Qwen2-57B-A14B 的 14B 激活中每专家 ~0.1-0.2B 参数，FP16 约 0.2-0.4GB）在 32GB/s 下传输约 6-12ms，远超 GPU 计算（μs-ms 级）——所以 SMoE 用替换把传输量降下来。论文提到 UMA（Apple M 系列统一内存）虽无离散 PCIe，但系统内存不足时退化为 SSD swap 仍呈 PCIe 式瓶颈，替换策略同样适用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要素：DMA 传输要求源/目标内存物理连续且 pin 在 RAM（cudaHostAlloc/cudaMallocHost 分配 pinned memory）；异步传输用 cudaMemcpyAsync + CUDA stream 与计算重叠。SMoE 运行时（https://github.com/goingshr/SMoE）中 PCIe 由两路操作构成：预取下一层专家（overlap 隐藏）与当层立即加载；CPU-assisted 调度把 C_load（单专家 PCIe 时间，用过去 p 次实测）作为主成本建模，min max(n_load×C_load, n_CPU×C_CPU)。评估平台：S1 3080Ti 12GB + PCIe 3.0 + E5-2683 v3、S2 4060Ti 16GB + PCIe 3.0、S3 A6000 48GB + PCIe 4.0 + Xeon Gold 6444Y。相关替代/补充互联：CXL（cache-coherent 内存共享）、NVLink（GPU-GPU）；STEP 补充视角（ISCA'26，PCIe 4.0 多卡 + 64GB/s switch 平台）：STEP 评估平台为 4× NVIDIA A100 80GB（PCIe 4.0、经 64 GB/s switch 互联）+ AMD EPYC 7542 32-core CPU + 512GB 主存，GPU-GPU 与 GPU-CPU 通信全部走 PCIe（刻意不用 NVLink peer-GPU 共享以与 baseline 公平）。profiling（Qwen3-30B-A3B、A100、INT8）显示专家取数（PCIe H2D）占执行时间约 88%，是 PCIe 带宽瓶颈的量化证据；STEP 用三招对抗：①空间剪枝减 k_l 从而减 PCIe 传输量；②窗口投票选举临时共享专家常驻 GPU、把每步动态加载从 k 降到 k−c；③独立 CUDA stream 异步 H2D 预取 + CUDA event 与专家计算重叠隐藏传输。硬件敏感性（Fig.19）：V100/A100/H20 互联能力递增时传输延迟下降、STEP 的"隐藏传输"优势减弱，但靠减少重复加载与热专家复用仍 ≥1.3×；NVLink/peer-HBM 场景下建议把 peer GPU HBM 作二级缓存（EP 扩展）。

CXL Type-3 / NDP 是把"cold"专家放近存侧计算以避免 PCIe 参数搬运的硬件演进方向。

涉及论文标题：
- SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference
