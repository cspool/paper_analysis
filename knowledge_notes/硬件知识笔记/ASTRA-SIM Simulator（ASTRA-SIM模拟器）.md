## ASTRA-SIM Simulator（ASTRA-SIM模拟器）

术语是什么？
ASTRA-SIM 是一个面向分布式深度学习训练的软硬件协同设计模拟器。它支持定义 DNN 模型的计算工作负载和通信模式，并在可配置的网络拓扑上模拟训练/推理的性能。ASTRA-SIM 提供两种后端：(1) Analytical backend——使用线性成本模型（启动延迟 + 传输时间）快速评估通信性能；(2) GARNET backend——cycle-accurate 网络模拟，逐 flit/packet 模拟路由、链路分配和拥塞。

从硬件架构角度拆解术语：
ASTRA-SIM 的模拟流程：
1. **工作负载定义**：用户定义 DNN 模型的 layer 图，每层的计算量（FLOPs）、通信量（数据大小）和并行策略（DP/TP/EP/PP）。ASTRA-SIM 将通信映射为对应的 collective（All-to-All、All-Reduce、All-Gather 等）。
2. **拓扑配置**：定义网络拓扑类型（torus/mesh/fat-tree 等）、维度数、每维节点数、链路带宽、延迟和路由策略。
3. **模拟执行**：Analytical backend 根据通信算法和拓扑特征，用线性成本模型直接计算通信时间（总 hop 数 × 单跳数据量 / 带宽 + 启动延迟）。GARNET backend 产生 flit/packet 级网络事件，模拟链路仲裁、VC 分配和拥塞。
4. **输出**：各 collective 的完成时间、链路利用率、通信-计算重叠时间、端到端训练/推理时间分解。

术语一般如何实现？如何使用？
在本文中，使用 ASTRA-SIM 的 analytical backend 进行无拥塞算法的快速模拟（合成实验、可扩展性、真实工作负载），使用 GARNET backend 进行有拥塞场景的 cycle-accurate 模拟（与 Google DOR/WFR 对比实验）。论文开源 artifact 提供完整的 ASTRA-SIM 配置和运行脚本（Zenodo: https://doi.org/10.5281/zenodo.16735313, GitHub: https://github.com/redbird-arch/micro2025-torus-ft-all2all-artifact）。使用时：创建 conda 环境 → 编译 ASTRA-SIM（`./build/astra_analytical/build.sh -c` 或 `./build/astra_garnet/build.sh -c`）→ 运行示例脚本（`bash run-all.sh`）。真实机器实验使用 PyTorch Distributed + Ascend 910B NPU 在 2 节点 16 设备上模拟 4×4 torus。

涉及论文标题：
- Optimizing All-to-All Collective Communication with FaultTolerance on Torus Networks
