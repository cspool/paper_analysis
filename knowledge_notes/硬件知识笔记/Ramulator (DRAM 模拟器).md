## Ramulator (DRAM 模拟器)

术语解释
Ramulator 是 CMU SAFARI 组开发的开源快速 DRAM 模拟器，支持多种 DRAM 标准（DDR3/4/5, LPDDR3/4/5, GDDR5/6, HBM, WIO 等）的 cycle-accurate 时序模拟。在本论文中，Ramulator 被用作 NDP 系统模拟器的基础，模拟 DDR-based CXL-NDP 设备的性能和延迟。

术语是什么？
Ramulator 的核心能力：
- 基于状态的 DRAM 时序模型：将 DRAM 命令调度建模为状态机，精确模拟 precharge/activate/read/write/refresh 等操作的时序
- 多模式运行：Memory trace driven、CPU trace driven、gem5 全系统集成
- 模块化设计（Ramulator 2.0）：分层状态机 + lambda 函数 DRAM 命令实现 + 可插拔 memory controller 组件

本论文对 Ramulator 的扩展：
- 在 Ramulator 上构建 NDP system simulator，增加 DDR-based NDP compute unit (64×(4×4) systolic arrays @ 1 GHz) 的建模
- 增加 CXL 互联模型：GPU↔NDP 通过 PCIe Gen4 ×16 的 activation/weight 传输延迟
- 增加 multi-precision NDP execution 模拟：不同 bitwidth (1/2/3/4-bit) 下的 NDP 计算延迟
- 增加 expert placement 调度模拟：prefill statistics → single migration → decoding with fixed placement 的全时序模拟

从硬件架构角度拆解术语：
Ramulator 在 NDP 模拟中的性能建模流程：

```
输入配置:
  - DRAM 标准: DDR5
  - NDP 参数: 64×(4×4) systolic arrays, 1 GHz, 512 GB/s BW, 512 GB capacity
  - GPU 参数: H100 SMs=132, TFLOP/s=989.4, HBM=80GB
  - 互联: PCIe Gen4 ×16 BW

模拟流程:
  Prefill → [Ramulator: DRAM access for expert weights] → 
    Router → [GPU model: GEMM latency] →
    Placement decision → [PCIe model: weight migration latency] →
  Decoding:
    per step per layer:
      GPU: [GPU model: attention + hot expert FFN latency]
      NDP: [Ramulator: DDR access latency for weights + 
            systolic array compute latency (bitwidth-dependent)]
      PCIe: [PCIe model: activation transfer latency]
      → max(GPU_latency, PCIe_up + NDP_latency + PCIe_down)
  
  输出: end-to-end latency, decoding throughput, NDP-side latency breakdown
```

术语一般如何实现？如何使用？
- 开源: https://github.com/CMU-SAFARI/ramulator (原始), https://github.com/CMU-SAFARI/ramulator2 (v2.0)
- C++ 实现，MIT 许可证
- 使用模式：配置文件指定 DRAM 标准/时序 + workload trace 输入 → 输出 cycle-accurate 延迟/带宽统计
- NDP 扩展：在 Ramulator 的 memory controller 模型中增加 compute unit 延迟模型，计算访存+计算的总延迟
- 局限性：Ramulator 是 DRAM 模拟器，不含 GPU 模型 → 需配合 GPU 性能模型（如本文的手工 H100 GEMM latency model）构建完整系统模拟
- 相关工具：Ramulator-PIM (https://github.com/CMU-SAFARI/ramulator-pim) 集成 ZSim 处理器模型实现通用 PIM 架构模拟

涉及论文标题：
- Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems
