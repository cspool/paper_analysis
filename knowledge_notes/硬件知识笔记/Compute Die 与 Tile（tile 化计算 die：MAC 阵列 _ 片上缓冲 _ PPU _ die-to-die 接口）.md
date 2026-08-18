## Compute Die 与 Tile（tile 化计算 die：MAC 阵列 / 片上缓冲 / PPU / die-to-die 接口）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SHyLA 每个计算 die 由 20 个同构 tile（各约 20mm²）+ 全局输入 buffer + 后处理单元（PPU，非线性运算）+ chiplet 级控制器 + die-to-die 接口组成。每个 tile 集成：MAC datapath（FP16 MAC Width 32、MAC Lane 32、每 tile 4 个 MAC Tree、0.2pJ/MAC，层级类似 [78]）、NoC router、本地缓冲（weight buffer/tile 1.2MB、output buffer/tile 100KB）与本地控制器；全局输入 buffer 4MB。内存带宽/容量由 CACTI-3DD 推导（DRAM 552GB/s、PCM 1792GB/s per die，2GB/64GB）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：chiplet 级控制器收粗粒度指令 → 分发给各 tile 本地控制器 → tile 从所属 memory plane 经本地缓冲取数据（weight buffer 存 Weight 块、global input buffer 存 IA 行）→ MAC 阵列按 output-stationary/input-stationary 执行分块 GEMM/GEMV（双缓冲与加载重叠）→ 输出写 output buffer → PPU 处理非线性 → die-to-die 接口做片间 all-reduce。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在 GPGPU-Sim 中建模计算 die 与内存控制器的 request 级交互；CUDA 管理片上缓冲并实现 plane-aware tile 映射（双缓冲）。论文未提供 tile 的 RTL/开源实现（联网未找到），MAC 层级参考 [78] 的共享层级。面积归一化（die 面积 400mm² 作为制造约束"货币"）是跨 DSA baseline 比较的关键。

涉及论文标题：
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
