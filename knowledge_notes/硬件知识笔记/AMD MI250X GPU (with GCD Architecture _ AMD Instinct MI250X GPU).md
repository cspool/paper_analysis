## AMD MI250X GPU (with GCD Architecture / AMD Instinct MI250X GPU)

术语是什么？

AMD MI250X 是 AMD 面向 HPC 和 AI 工作负载的数据中心 GPU，基于 CDNA2 架构，部署于 Frontier 超级计算机。其关键架构特征是每块物理 GPU 包含两个 Graphics Compute Die (GCD)，每个 GCD 被视为一个独立的有效 GPU。两个 GCD 通过 Infinity Fabric 互联（峰值带宽 200 GB/s），共享同一块 PCIe 插槽。每 GCD 在 FP16 下的峰值吞吐为 191.5 TFLOPs。

从硬件架构角度拆解：

MI250X 的内存与互联层次：
- GCD 0 ←→ GCD 1：Infinity Fabric 200 GB/s（同物理 GPU 内）
- GCD 0 ←→ 同节点其他 MI250X 的 GCD：Infinity Fabric 50-100 GB/s
- 跨节点 GCD ←→ GCD：Slingshot 25 GB/s（通过 NIC）

X-MoE 利用这一层次结构设计 RBD：最大化 GCD-GCD 快速链路的利用率，最小化跨 Slingshot 的通信。

与 NVIDIA 平台的对比：
- NVIDIA A100：单 die，NVLink 600 GB/s intra-node, InfiniBand 200 GB/s inter-node（带宽比 3:1，较均衡）
- AMD MI250X：双 GCD，Infinity Fabric 200 GB/s intra-node, Slingshot 25 GB/s inter-node（带宽比 8:1，极不对称）

这种不对称使得 X-MoE 的 RBD 和 DP-First 等拓扑感知策略在 MI250X 平台上收益更显著。

术语一般如何实现？

在 ROCm 5.7.1 软件栈下运行，使用 PyTorch 2.2.0 + DeepSpeed 0.15.5。每 GCD 独立寻址、独立执行 kernel，两个 GCD 间需要显式的数据传输。X-MoE 将每个 GCD 视为一个独立 GPU 进行 EP/DP/TP 并行配置。

涉及论文标题：
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms
