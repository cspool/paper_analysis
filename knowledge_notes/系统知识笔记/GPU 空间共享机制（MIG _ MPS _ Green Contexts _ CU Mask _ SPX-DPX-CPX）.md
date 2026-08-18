## GPU 空间共享机制（MIG / MPS / Green Contexts / CU Mask / SPX-DPX-CPX）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 一组把单颗 GPU 的计算资源在多个租户/进程间做空间划分的机制：NVIDIA MIG（Multi-Instance GPU）——硬件级空间分区，把 GPU 切成多个实例（各含独立 SM 分区/内存/L2 slice），强隔离、性能可预测；NVIDIA MPS（Multi-Process Service）——进程级并发共享，多个进程的 kernel 可同时在 SM 上执行（提高利用率但隔离较弱）；CUDA Green Contexts——driver API 提供的绿色上下文，可在 MPS 内实现流/上下文级空间隔离（2026 新增，vLLM 生态已有 MuxWise 用 GreenContext 做 prefill-decode 空间复用）；AMD 侧等价物：SPX/DPX/CPX（MI300X/MI355X，跨 XCD 的 MIG 等价隔离）与 ROCm CU masking（流到 Compute Unit 的 MPS 式细粒度分配）。PowerWeave 的 spatial partitioning 模型与这些机制兼容：MIG 下 Governor 无需修改即可在实例内工作；MPS 下沿用 LithOS 的 TPC assignment 或用 Green Contexts 做隔离。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 系统架构中的角色：空间共享机制提供"多模型/多租户同卡共存"的资源底座，PowerWeave 在其上叠加"每空间域独立频率"。运转流程例子：三租户 GPU（18/19/37 TPC）——MPS/Green Contexts 下各租户 kernel 在其 TPC 集上执行 → PowerWeave 把每个 TPC 集视为独立频率域 → 各域按自己 SLO 调频（一个租户拉满频率不抬高其他租户频率）→ 消除"高负载租户触发全局热节流拖累低负载租户"的干扰。MIG 场景：硬件分区天然对应独立域，Governor 在 MIG 实例内无需改动直接监控实例级 RPS/尾延迟。论文还用 MIG-based partitioning 与 isolated/same-GPU compute partitioning 对比仿真空间 DVFS 的共享资源争用（平均 +3%、最坏 <+7% TTFT/TPOT，SLO 按此保守放大）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：MIG 经 nvidia-smi/mig 配置（硬件分区，参考 https://docs.nvidia.com/datacenter/tesla/mig-user-guide/）；MPS 经 CUDA_MPS 控制（https://docs.nvidia.com/deploy/mps/）；Green Contexts 经 CUDA driver API（https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA_GREEN_CONTEXTS.html，MuxWise 论文报告一组 green context 仅 4MB 开销）；AMD 侧见 https://instinct.docs.amd.com/（SPX/DPX/CPX）与 ROCm CU mask（https://rocm.docs.amd.com/projects/HIP/）。使用场景：多租户 LLM serving、disaggregated prefill/decode 同卡共存、agentic 流水线多模型同卡——这些正是空间 DVFS 的收益场景（每域独立频率、每域独立节流、每域独立 SLO）。

涉及论文标题：
- PowerWeave: Unlocking Energy-Efficient ML on GPUs with OS-Level Spatial Power Management
