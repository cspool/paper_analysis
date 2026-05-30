## Optimizing All-to-All Collective Communication with FaultTolerance on Torus Networks

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：All-to-All 集合通信在 N-D torus 网络上的算法与调度优化。在无故障场景下提出 **HalfRing 算法**（利用双向链路实现最短路径单跳 store-and-forward 数据传输）和 **DimRotation 调度**（将数据分块并轮转每块的维度通信顺序，实现无气泡的全带宽利用）。在故障场景下提出 **FoldedRing 算法**（在故障链路上构建折叠环以维持容错通信）和 **MATE/MATEe 调度**（利用其他维度的链路加速故障环上的数据传输）。
  - 实验比较：
    - 无故障：Ring+Pipeline（baseline）vs HalfRing+Pipeline、Ring+DimRotation、HalfRing+DimRotation
    - 有故障：Ring+Pipeline（fault-free baseline）vs FoldedRing+Pipeline、FoldedRing+DimRotation、MATE、MATEe
    - 与 Google TPUv4 的 DOR（Dimension-Order Routing）和 WFR（Wild-First Routing）对比
    - 指标：性能加速比、All-to-All 带宽、维度利用率、可扩展性、端到端训练/推理时间分解、非均匀 All-to-All 性能、多故障弹性

- 后端平台是什么，配置是什么。
  - 模拟平台：ASTRA-SIM 模拟器（analytical backend + GARNET cycle-accurate backend）
  - 拓扑：2D/3D/4D torus（合成实验），4×4×4 TPUv4 pod（单pod），8×4×4 TPUv4 pod（双pod），TPUv3 8×8，TPUv4 8×8×8（实际工作负载）
  - 链路带宽：32 GB/s（合成实验），56 GB/s（TPUv4），82 GB/s（TPUv3）
  - 网络延迟：100 ns
  - 真实机器：16×Ascend 910B4 NPU（2节点，每节点8设备），节点内高带宽链路，节点间 200Gb/NPU RoCE

- 评估性能的软件/脚本是什么。修改了什么。
  - 软件：ASTRA-SIM 模拟器（https://github.com/astra-sim/astra-sim），GARNET 网络模拟器
  - 修改内容：在 ASTRA-SIM 中实现了 HalfRing、FoldedRing 算法及 DimRotation、MATE/MATEe 调度的 collective communication 策略；在 GARNET backend 中实现了 DOR（Dimension-Order Routing）和 WFR（Wild-First Routing）作为对比 baseline，并加入 dateline 死锁避免机制
  - 真实机器：PyTorch Distributed 模块（torch.distributed），在 Ascend NPU 上模拟 4×4 torus 拓扑（禁用节点内互联，限制通信到特定设备对）

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：是。Zenodo DOI: https://doi.org/10.5281/zenodo.16735313；GitHub: https://github.com/redbird-arch/micro2025-torus-ft-all2all-artifact
  - 使用方式：
    1. **分析后端实验**：`cd analytical_backend/ && conda env create -f astra-sim-analytical.yml && conda activate astra-sim-analytical && ./build/astra_analytical/build.sh -c`，然后运行 `cd examples/scripts/ && bash run-all.sh`
    2. **GARNET 后端实验**：`cd garnet_backend/ && conda env create -f astra-sim-garnet.yml && conda activate astra-sim-garnet && bash setup_protobuf.sh && ./build/astra_garnet/build.sh -c`，然后运行 `bash run-all.sh`
    3. **真实机器实验**：`cd real_machine/ && bash Run_All_to_All.sh`
  - 评估原理：ASTRA-SIM 接收计算工作负载描述（DLRM/MoE 模型的 layer 定义与并行策略）和网络拓扑配置，在 analytical backend 下使用线性成本模型（启动时间 α + 传输时间 S/B）直接计算通信时间；在 GARNET backend 下进行 cycle-accurate 网络模拟，逐 flit/packet 模拟路由、链路分配和拥塞。最终输出 All-to-All 完成时间、带宽、维度利用率等指标。输入为系统配置（拓扑、带宽、延迟）和通信数据量，输出为 PDF 图表（Fig 11-19）。
