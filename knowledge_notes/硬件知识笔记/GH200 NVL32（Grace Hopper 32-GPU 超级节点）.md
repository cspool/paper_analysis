## GH200 NVL32（Grace Hopper 32-GPU 超级节点）

术语解释
NVIDIA 的 32-GPU 机架级参考系统：32 个 GH200 Grace Hopper superchip 经 NVSwitch 全互联，作为本论文模拟与评估的目标平台。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GH200 superchip = Hopper 世代 GPU（H100 级）+ 72 核 Grace CPU（Arm Neoverse V2，最高 480GB LPDDR5X）经 NVLink-C2C 片间互联封装；每 GPU 18 个 NVLink 4.0 端口（双向 900 GB/s，单跳约 250ns）。NVL32 把 32 个 superchip（16 个双 GPU 计算 tray）经 9 个 Gen3 NVSwitch（64 端口/块；物理部署为 9 个 switch tray × 2 芯片）连成 576 端口的全连接 fat-tree，提供 NVLink 可寻址统一内存（约 19.5TB）与约 14.4 TB/s bisection 带宽。定位：单节点内聚更多 GPU 以支撑 EP/TP 等大模型并行，减少跨节点慢速链路。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
本论文以 H200 规格配置每 GPU、以 NVLink 4.0 真实器件参数（900 GB/s 双向、250ns 单跳、1µs 往返、16B flit）+ NVSwitch 参数（每输入端口 16×256 深 VC、64KB reduction buffer）在 Accel-Sim + BookSim2 中逐周期建模 NVL32。扩展评估：64-GPU 节点 = 18 个 64 端口 NVSwitch（每端口直连一 GPU，NVL32 的双倍互联）；对比平台 DGX-H100（8 GPU）；多节点评估 4/8×DGX-H100、2/4×NVL32 + InfiniBand（Quantum Switch）+ 16-way pipeline parallelism。训练内 NVL32 节点：attention 层数据并行 + MoE 层专家并行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
商用交付：AWS EC2 / NVIDIA DGX Cloud 整机架交付；软件栈用 CUDA 统一地址空间 + NCCL/NVLS（本论文的 NVLS/NVSwitch 建模即针对该平台）。研究意义：单节点 GPU 数持续增加（NVL32→GB200 NVL72→256-GPU SuperPod），计算能力增长快于通信能力，MoE 通信瓶颈加剧（本论文立论之一：Blackwell/Rubin 世代该比率进一步恶化）。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch Computing on Multi-GPUs
