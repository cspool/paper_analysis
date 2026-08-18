## NVLink（NVIDIA 高速 GPU 互连）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NVLink 是 NVIDIA 专有的高带宽、低延迟 GPU 间互连技术，从 Pascal 时代首代（160 GB/s）演进到第四代（每 GPU 双向 900 GB/s，如 H100/B200），并在 Hopper 起集成 NVSwitch 与 NVLink SHARP（NVLS，in-switch 多播/归约计算）。物理上 NVLink 是板级/封装级芯片间互连（GPU-to-GPU 直连、NVIDIA GPU-CPU NVLink-C2C 扩展），不属于标准网络协议（区别于 InfiniBand/Ethernet 的 scale-out 网络）。本论文（Rearchitecting the Datacenter Lifecycle for AI）把 NVLink 放在数据中心 build 阶段的组网决策中（表 VII）：对比 all-Ethernet、all-InfiniBand、all-NVLink、hierarchical（NVLink intra-server + InfiniBand intra-rack + Ethernet inter-rack）四种设计，评估成本/带宽/延迟/能耗权衡。NVLink 是 tensor parallelism（TP）的通信底座——TP 的 all-reduce 在节点内走 NVLink（论文实测 TP1/TP4/TP8 的跨代 GPU 效率，V100→H200）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
NVLink 在芯片/系统互连组织中的运转流程（以 8×H100 DGX 为例）：
```
8 个 GPU 各自 NVLink 端口 → NVSwitch（全连接 crossbar，4 个 NVSwitch 提供 448GB/s 全对全）
TP-8 执行 Llama3-70B: QKV/FFN 权重按列切分 → 每 GPU 本地部分和
  → all-reduce 经 NVLink/NVSwitch 聚合（multimem 指令走 NVLS 在 switch 内归约）
  → 激活在 GPU 间高速交换，无 NIC/网络协议栈开销
论文视角: NVLink 是 scale-up 网络——成本高（表 VII：Cost Higher），
  但 TP 场景必须；生命周期决策只在"模型大到需跨机架"时才需 InfiniBand/Ethernet
  结论: hierarchical（NVLink intra + IB intra-rack + Eth inter-rack）降 TCO 6%
```
论文还指出 all-NVLink 全局组网并非现实部署（拓扑/扩展性/可用性受限），但作为"理想高带宽低延迟"上界参考，用于推演哪些架构选择在生命周期内保持高效。NVLink 的带宽/延迟参数进入 roofline 模型的互连维度（interconnect bandwidth and latency），影响多 GPU 推理的 TTFT/TBT 预测。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：NVLink 以定制 SerDes 走 PCB/背板（NVLink 4.0 每链路 200GB/s 双向），NVSwitch 芯片做全对全交换（DGX H100 双 8-GPU 域），NVLS 的 multimem 指令（multimem.st 多播、multimem.ld_reduce 归约）让集合通信在 switch 内完成。库层面 NCCL 自动检测 NVLink 拓扑并选择最优 collective 路径。论文在 AI Lifecycle Compass（https://github.com/Azure/AI-Lifecycle-Compass）中以 YAML 配置硬件 roadmap（含 NVLink 规格：带宽/延迟/每 server 成本），性能模块用其做 roofline 互连维度建模，build 模块做组网拓扑 TCO 比较（表 VII 的成本数据参考 [61],[90]）。相关语境：MixNet 等论文把 NVLink 归类为 intra-server scale-up 网络，与 Ethernet/InfiniBand 的 scale-out 网络区分；NVLink 带宽显著高于 NIC（如 900GB/s vs 400GbE），因此"
ShadowUpdate 补充视角（ISCA'26，MGPUSim 中的互联建模）：ShadowUpdate 论文把多 GPU 系统的互连建模为两级：GPU-GPU 高带宽互联 600GB/s（NVLink 级，承载页迁移的 copy engine 传输与 completion 信号）、CPU-GPU 互联 32GB/s（PCIe 级，承载 page fault 的 ATS 请求与 host IOMMU 往返）。页迁移机制设计因此与带宽直接相关：invalidation/completion 广播用 GPU-GPU 互联（论文称相对 re-fault 成本开销可忽略），而 baseline 的每次 re-fault 都要走 32GB/s CPU-GPU 路径 + host 集中式走查——ShadowUpdate 把翻译留在 GPU 内、用 600GB/s 路径完成映射传播，是"互连带宽决定迁移/翻译机制取舍"的量化实例。

RoCC 补充视角（ISCA'26，ROP 复用下的 NVLink 互连利用）：RoCC 论文把 NVLink 作为 CC 网络假设（V100 baseline 300GBps full-mesh、H100 900GBps、B200 1.8TBps，CPU-GPU 用 PCIe Gen4 x16 ≈150 cycle），并把 CC 数据通路改造成"ROP↔NVLink"而非"SM↔NVLink"：跨 GPU 的 doorbell 包（含归约结果 payload）由 ROP 经 MPU 的 memory issue 逻辑直接发到目标 GPU 的 ROP，网络带宽成为 CC 的唯一瓶颈（roofline 分析：AllReduce 操作强度 ≈0.1 FLOPs/Byte，网络 bound）。互连延迟敏感性：2× 慢/快互连仅使性能 -6.5%/+2.5%；随 GPU 规模扩到 32-256，RoCC 稳定 13%-21% 加速，说明 ROP 卸载在更大 NVLink 集群上仍有效。

- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
RESONATOR 补充视角（ISCA'26，NVLink 上的 encoder TP 通信）：RESONATOR 的 8×A100（NVLink 互联，A100 12 条 NVLink、600GB/s 双向）上，encoder 的 TP 通信（all-reduce）等待间隙的 SM 被 Intra-GPU Sharing 引擎回收给 co-located decode（TPOT 收益来源）；TP 度选择的成本模型正比于"通信量随 TP 度增长（~线性）vs 计算量随序列长增长（~二次）"的对比——低分辨率时通信主导、1 GPU 最优，高分辨率时 compute 主导、4-TP 最优，该权衡在 NVLink 带宽固定的前提下随分辨率变化，驱动 PRISM 动态选 TP；logical sharding 使 TP 切换无需跨 NVLink 搬权重（数据面零传输）。
涉及论文标题：
- Reducing Page Faults via Invalidation-based Mapping Propagation in Multi-GPU Systems
- Rearchitecting the Datacenter Lifecycle for AI
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication
