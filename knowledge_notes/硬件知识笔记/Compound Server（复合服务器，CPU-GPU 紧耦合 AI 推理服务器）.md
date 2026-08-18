## Compound Server（复合服务器，CPU-GPU 紧耦合 AI 推理服务器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Compound Server 指把 AI 加速器（GPU 等）与 AI 优化的 CPU 紧密集成在同一平台/机箱内的服务器形态，如 NVIDIA Grace Hopper（H100 GPU + Grace CPU 经 NVLink-C2C 集成）以及 8 GPU + CPU host 的通用 GPU 服务器。论文（Power Sloshing）用它承载大规模 AI 推理 fleet：CPU 负责预处理（特征提取、输入归一化）、后处理（结果聚合、格式化）与请求调度，GPU 负责模型计算；单服务器含多个 GPU（典型 4-8 个）共享一个服务器级功率预算。与 GH200 NVL32（32-GPU 机架级系统）的区分：后者是"超级节点/机架参考系统"，前者是单服务器内的 CPU+GPU 异构计算单元组织。
- 关键特性（本文观测）：① 组件功率需求随服务/模型差异极大——CPU 密集服务（预处理重）CPU 触顶而 GPU 闲置，GPU 密集服务（长上下文 LLM，CPU 只做调度）反之；② 同服务器多 GPU 功率分布高度不均衡且随时间波动（colocate 多服务、diurnal 模式、流量尖峰）；③ 组件功率占比相对稳定（C1：GPU 65%、GPU 内存 10%、CPU 22%、CPU 内存 3%），但绝对值因模型而异。这些特性使"静态独立功率上限"（每组件固定 TDP）严重失配，是 power sloshing 动机的物理基础。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件组成与数据路径：GPU（H100，含 HBM/GPU memory）经 NVLink-C2C 与 Grace CPU（Arm 核，含 DDR/LPDDR CPU memory）紧耦合；8-GPU 平台中 GPU 间经 NVSwitch/NVLink 互联。一次推理请求流程：请求到达 → CPU 核做特征预处理/调度 → 经 NVLink-C2C 把输入送到 GPU → GPU 执行 embedding/MLP 等计算（消耗 GPU 功率大头）→ 结果回 CPU 做后处理/聚合 → 返回。CPU 与 GPU 共享服务器供电/散热预算：固件为各组件分配功率上限（TDP），DVFS 在各自上限内调速。论文通过仪表化测量各组件功率（GPU/GPU memory/CPU/CPU memory）以刻画功率分解与频率-功率关系（Fig.9：0.5×TDP_M 下 GPU 频率降至 55% f_GM、其余组件功率继续上升）。
- 硬件-软件协同启示（本文 §VII）：现有功率管理接口粒度粗、响应慢（频率缩放 100µs-数 ms、功率上限更长），需要更细粒度/更快的功率接口；负载均衡器与硬件电源控制器之间缺乏集成；建议硬件级电源管理内置化。这些为下一代 AI 平台（Blackwell 级多 GPU 复合服务器）的 co-design 指南。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：商用平台为 NVIDIA Grace Hopper（NVLink-C2C 连接 CPU-GPU，统一内存模型，https://docs.nvidia.com/grace/）、DGX 系列（8 GPU+CPU host）、AMD Instinct + EPYC 组合等。使用：数据中心按模型/服务混部部署，多服务 colocate 到同一服务器；运维通过 nvidia-smi / amd-smi 设功率上限，通过 NVML 读利用率/功率/频率。评估本论文方法需具备可配置频率/功率上限的 NVIDIA 平台（Grace Hopper 或其 8-GPU 服务器），负载为生产推荐/排序模型（不可公开）。论文未开源，硬件为商用平台，无模拟器。

涉及论文标题：
- Power Sloshing in Compound Servers for Large-Scale AI Inference Workloads
