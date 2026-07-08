# idea库

## Accelerating MoE with Dynamic In-Switch

- baseline方法是什么？
  - **Baseline 通信实现**：MoE 的 Expert Parallelism (EP) 中，Dispatch（将 token 发送到对应的 expert GPU）和 Combine（将 expert 输出聚合回 token 源 GPU）通过点对点通信库（如 DeepEP）实现。每个 token 的相同数据在 Dispatch 时从源 GPU 经 NVLink 多次传输到 Switch 再到多个目标 GPU；Combine 时多个 GPU 的中间结果各自独立经 Switch 传回源 GPU 然后软件归约。这种冗余通信占 MoE 层执行时间的 50-80%（DeepSeek-V3 上为 70.4%），其中近 50% 流量为冗余（topk≥8 时）。
  - **全栈执行例子**：
    - **算法pipeline**：MoE FFN 层，token 经 gate network 选择 topk experts，各 expert 执行 GEMM-1 + GEMM-2，topk 输出加权求和。
    - **系统框架**：DeepEP 通信库或 NCCL AllGather/Reduce-Scatter；FasterMoE/Tutel/COMET 提供计算-通信重叠。PyTorch + CUDA。
    - **编译框架**：论文未明确说明。标准 CUDA 编译工具链，kernel 通过 nvcc 编译。
    - **kernel调度**：Dispatch kernel 逐 token 封装目的地址、经 NVLink 发送；Combine kernel 逐 expert 读取输出、经 NVLink 回传、软件归约。COMET 等通过 SM 分区实现 GEMM 与通信的部分重叠，但 Dispatch 和 Combine 仍为独立 kernel，各自隔离执行。
    - **硬件架构**：NVIDIA GH200 NVL32——32 GPU 经 9 NVSwitch（NVLink 4.0, 900 GB/s 双向）全连接。NVSwitch 仅做数据转发，无动态 in-switch 计算能力（NVLS 仅支持静态 AllGather/Reduce-Scatter 的 in-switch multicast/reduction，无法处理 MoE 的动态非规整通信）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法**：DySHARP 提出 integral dynamic in-switch computing 方案，包含两大技术：
    1. **Dynamic Multimem Addressing**：扩展 NVLS 的 multimem 寻址，新增 `dymultimem.st`（Dispatch 多播）和 `dymultimem.ld_reduce`（Combine 归约）指令。包携带单个 multimem 地址 + 轻量 target expert list，GPU 本地管理内存（AL Table 做 algebraic→layout index 映射），Switch 按 target list 多播/归约。消除 Dispatch 冗余（源 GPU→Switch 仅发一次）和 Combine 冗余（Switch 内归约后仅回传最终结果），减少近 50% 通信流量。
    2. **Token-Centric Kernel Fusion**：通过 Token Tracker（TS/TID/OR Table）追踪 token/tile 粒度依赖，Token-Centric Scheduler（megakernel + persistent TBs）将 Dispatch-GEMM1-GEMM2-Combine 全流水线化。Dispatch（GPU→Switch 为主）和 Combine（Switch→GPU 为主）并发执行，互补的非对称通信模式合并，将流量减少转化为实际加速。**两种技术缺一不可**：仅 dynamic multimem addressing 无法将流量减少转为加速（因双向非对称瓶颈），仅 kernel fusion 无流量减少也无法超过 SOTA baseline。
  - **全栈执行例子**（对比 baseline）：
    - **算法pipeline**：算法层面不变（MoE FFN + topk gating），但 Combine 的加权求和通过在 GEMM-2 epilogue 中提前乘 gating weight 实现（因 dymultimem.ld_reduce 不支持加权归约）。
    - **系统框架**：扩展 CUDA Runtime API——新增 `cuDyMulticastCreate` 指定 block size/stage/vector length，`cuDyMulticastBindAddr` 绑定 multimem 和 virtual 地址空间。程序员使用 dymultimem 指令替代传统 send/recv，Dispatcher 不再需软件追踪 remote memory state。
    - **编译框架**：论文未明确说明。dymultimem 指令作为 PTX/HOASM 扩展嵌入 CUDA kernel，标准 nvcc 流程。
    - **kernel调度**：Token-centric scheduler 将 SM 划分为四组（Dispatch/GEMM-1/GEMM-2/Combine），以 readiness-gated 方式调度：TS Table 检测 Dispatch 到达 tsize tokens→立即触发对应 GEMM-1 TB row → GEMM-1 完成触发 GEMM-2 TB row → GEMM-2 完成通知 source GPU 更新 OR Table → nReady==topk 时立即执行 Combine。Dispatch 和 Combine 在时间上并发，合并 GPU→Switch（Dispatch 主导）和 Switch→GPU（Combine 主导）的互补带宽。
    - **硬件架构**：
      - **Source GPU**：SM LSU 新增 MultimemQ（32 entry），先预取 target list，再组装 dymultimem 请求包（flit0: 48b 地址+stage+target count，后续 target extension flit）。
      - **Switch**：Route 模块按 target 计算 OutPort，复制并裁剪请求包（每端口仅保留该端口 target）。Reduction Logic 追踪归约计数（减至零返回结果），数据路径仅增 1 cycle。
      - **Destination GPU**：Hub Hardware Memory Manager——Dispatch 时 AL Table 动态分配 layout block（AIdx→LIdx 注册），Combine 时查表获取 LIdx 做 multimem→virtual 翻译。AL TLB（512 entry CAM+SRAM）加速查表。AL Table 大小 4×nToken bytes（1M token 仅 4MB/layer）。
      - **Token Tracker**：TS Table（1024 entry on-chip）追踪 Dispatch→GEMM-1 到达量（DAcc 字段）和 GEMM TB 完成量；TID Table（DRAM）记录 token tile 中的 token ID；OR Table（1024 entry on-chip）追踪每个 token 的 Combine 就绪状态（nReady 计数）。
