## CXL (Compute Express Link) for MoE

术语解释
CXL（Compute Express Link）是一种开放的高速互连标准，在MoE推理中用于连接GPU与近存计算（NDP）单元，实现低延迟的激活值传输和expert计算结果回传。

术语是什么？
CXL基于PCIe物理层，提供三种协议：
- CXL.io：基于PCIe的I/O协议（设备发现、配置等）
- CXL.cache：允许设备访问主机CPU缓存
- CXL.mem：允许主机CPU访问设备内存

MoNDE使用CXL连接GPU与NDP控制器：
- Activation Movement模式中，activation通过CXL传输到NDP核
- 相比将expert权重从CPU内存传输到GPU（通过PCIe），CXL路径传输更少的字节
- 利用CXL的低延迟特性支持实时expert计算

从硬件架构角度拆解术语。
在MoNDE中，CXL作为GPU与NDP之间的数据传输通道：
```
GPU (HBM) ←→ [PCIe/CXL Switch] ←→ CXL NDP Controller
                                        ↓
                                   LPDDR SDRAM
                                   (cold expert weights)
                                        ↓
                                   NDP Core
                                   (in-memory compute)
```

CXL相比传统PCIe的优势：
- 缓存一致性（CXL.cache协议）
- 更低的延迟和更高的带宽效率
- 支持内存池化和资源共享

术语一般如何实现？如何使用？
- 需要支持CXL的CPU（Intel Sapphire Rapids、AMD Genoa等）
- CXL交换机（如Astera Labs Leo）用于多设备互联
- 在MoE场景中，CXL NDP设备是新兴形态，尚未大规模商用
- 当前更多处于研究原型和标准制定阶段

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems

**补充（来自 Context-Aware MoE on CXL-NDP）**：该论文将 CXL-NDP 作为 MoE expert 的 offloading tier，关键设计是：(1) prefill-guided once-per-sequence expert migration → decoding 期间 zero additional migration via CXL/PCIe；(2) CXL 通道仅传输 activation (~8KB/token)，而非 expert weight (~170MB/expert)；(3) NDP 设备使用 64×(4×4) systolic arrays @ 1 GHz + 512 GB DDR + 512 GB/s internal BW，通过 CXL 连接 GPU 实现 GPU-NDP 并行 pipeline。对比 baseline MoNDE 的 context-agnostic placement + on-demand expert migration，本论文通过 prefill 统计驱动的动态 hot/cold 分类和单次迁移策略，将 CXL 带宽从 parameter movement 的瓶颈转化为 activation movement 的高效通道。

---
