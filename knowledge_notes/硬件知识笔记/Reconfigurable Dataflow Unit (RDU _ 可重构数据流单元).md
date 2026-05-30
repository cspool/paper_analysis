## Reconfigurable Dataflow Unit (RDU / 可重构数据流单元)

术语是什么？
Reconfigurable Dataflow Unit (RDU) 是 SambaNova 的商业数据流加速器架构系列，以 streaming dataflow 为核心理念，区别于传统 GPU 的 SIMT 和 TPU 的 systolic array 架构。SN40L RDU 是第四代 RDU，采用 TSMC 5nm 工艺，每 socket 含两个 Reconfigurable Dataflow Dies (RDD) 加 HBM（2.5D CoWoS 封装）。核心组件：1040 个 PCU（Pattern Compute Units）、1040 个 PMU（Pattern Memory Units）、AGCUs（Address Generation and Coalescing Units），通过 RDN（Reconfigurable Dataflow Network，2D mesh）互连。RDU 的关键特性是"可重构"——PCU 可配置为 systolic array（GEMM）或 pipelined SIMD core（element-wise），PMU 的 scratchpad 和地址生成由软件可编程，RDN 的路由表由 Place-and-Route 配置。RDU 支持三级存储体系（片上 SRAM + HBM + DDR），区别于仅含 HBM 的传统 GPU/TPU。SN40L RDU 提供 638 BF16 TFLOPS 峰值算力。

从硬件架构角度拆解：
RDU 的 tile 级架构由 PCU、PMU、AGCUs、RDN switches 组成，排列在 2D grid 上。单个 kernel 执行的数据流：
```
HBM/DDR → AGCUs → RDN(vector fabric) → PCU(systolic/SIMD) → RDN → PMU(scratchpad/stage buffer)
    ↕ (credit-based flow control, sequence ID based reordering)
RDN(scalar fabric): AGCUs/PMUs间传输metadata/address
RDN(control fabric): 分布式coarse-grain flow control tokens (e.g., counter done events)
```
PCU 之间通过 chaining 实现 pipeline parallelism；PCU 间通过 data/tensor parallelism（partition/fork/join）实现并行；PMU 作为 stage buffer 解耦 producer/consumer；AGCUs 桥接片上 RDN 和片外 HBM/DDR。

术语一般如何实现？如何使用？
RDU 通过自研编译器从 PyTorch 代码生成 dataflow kernel 配置。用户无需编写硬件级代码，编译器自动完成 PCU 模式选择、PMU 地址生成器编程、RDN 路由配置、AGCUs 数据传输调度。RDU 面向 enterprise AI 推理和训练，特别适合 Composition of Experts 和大规模 LLM 部署。

涉及论文标题：
- SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts
