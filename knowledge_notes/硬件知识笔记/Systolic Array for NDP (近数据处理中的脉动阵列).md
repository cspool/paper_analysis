## Systolic Array for NDP (近数据处理中的脉动阵列)

术语解释
Systolic Array（脉动阵列）是一种高度规则的并行计算结构，由二维网格的处理单元（PE）组成，数据以"脉动"方式在 PE 间同步流动。在 CXL-NDP 设备中，Systolic Array 作为核心计算引擎执行矩阵乘法操作，以有限的功耗和面积预算提供可观的吞吐量。

术语是什么？
脉动阵列的核心原理：
- 每个 PE 执行一个 MAC（乘加）操作，输入数据从相邻 PE 流入、计算后流向下一级
- 数据流动像心跳脉冲一样规律（故名"systolic"）
- 典型结构：N×N PE 网格 → 每个 cycle 完成 N 个 MAC → 对矩阵乘法有极高的数据复用效率

GPU-NDP 系统中的 NDP 脉动阵列配置（本论文）：
- 64 × (4×4) systolic arrays → 64 个独立的 4×4 PE 子阵列
- 每个 4×4 array: 16 个 PE/array × 64 arrays = 1024 MACs/cycle
- Clock: 1 GHz → 峰值 1 TOPS (INT8 equivalent)
- 不同 bitwidth 有效吞吐：4-bit ~4× INT8（位宽减半，MAC 数量翻 4 倍）；1-bit → XNOR + popcount 操作（极高吞吐）

与 Google TPUv1 的对比：TPUv1 使用单个 256×256 systolic array (65536 MACs)，本论文 NDP 使用多个小型 4×4 阵列（64×16=1024 MACs），以适应 DDR 控制器的面积和功耗限制。

从硬件架构角度拆解术语：
NDP 脉动阵列的矩阵乘法执行流程（4×4 array, 计算 Y = A × B）：

```
Cycle 0:  [a00→PE00]  [a01→PE01]  ...
Cycle 1:  [a00→PE00·b00] [a01→PE01] [a10→PE00]
Cycle 2:  [PE00: a00·b00 + a01·b10] [PE01: a00·b01] ...
...
# 数据沿行方向流动（权重）和列方向流动（激活）
# 每个 PE 累加部分积，最终形成输出矩阵

4×4 array 输出: 4×4 partial result matrix per cycle
64 parallel arrays: 64 × 4×4 = 1024 partial results per cycle
```

```
Bitwidth 对 systolic array 吞吐的影响:
FP16 (16-bit weight × 16-bit activation):
  - 每次 MAC 需要 16b×16b 乘法器 → 面积大、功耗高
  - 64×16 = 1024 MACs/cycle → 1 TMAC/s (FP16 equivalent)

INT4 (4-bit weight × FP16 activation):
  - 将 4 个 4-bit weights 打包为 16-bit → 一次乘法完成 4 个 MAC
  - Effective throughput: 4× INT8 → 4 TMAC/s
  - 面积节省: 4-bit 乘法器远小于 16-bit

INT1 (binary weight):
  - 权重 = ±1 → 乘法退化为条件符号翻转 (XNOR + popcount)
  - Throughput: ~16× vs FP16 equivalent（极度高效的 binary 操作）
```

术语一般如何实现？如何使用？
- 硬件实现：PE = 乘法器 + 累加器 + 寄存器，通过 controller 协调数据流入流出
- 编程模型：将矩阵乘法映射到 systolic array 的 weight stationary / output stationary / input stationary 数据流
- NDP 使用场景：执行 quantized expert FFN 的矩阵乘法 (activation × expert weight)，利用高内部 DDR 带宽 (512 GB/s) 减少数据搬运
- 面积/功耗权衡：64×(4×4) 小型阵列 → 总面积小、适合嵌入 DDR 控制器旁 → 计算密度约 1 TOPS → 远小于 H100 (989 TFLOP/s) 但足以处理 cold experts (量化后)
- 关键约束：systolic array 计算吞吐远低于 GPU → 需配合量化降低 NDP 计算压力

涉及论文标题：
- Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems
