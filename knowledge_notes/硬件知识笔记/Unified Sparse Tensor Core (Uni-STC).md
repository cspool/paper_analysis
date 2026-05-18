## Unified Sparse Tensor Core (Uni-STC)

术语是什么？

Uni-STC (Unified Sparse Tensor Core) 是一种替代 GPU SM 中原有 dense tensor core 的统一稀疏张量核架构，通过 BBC 格式、TMS/DPG/SDPU 三阶段流水线和 UWMMA 指令序列协同设计，统一覆盖 SpMV、SpMSpV、SpMM、SpGEMM 四类常见稀疏 kernel。与 NVIDIA 2:4 structured sparse TC 和已有学术 STC 设计（DS-STC、RM-STC）不同，Uni-STC 不对稀疏模式施加约束，而是通过"gather tasks"策略替代传统的"gather data"模式，将固定形状任务拆为灵活的 dot-product 片段以提高 MAC utilisation。在 SuiteSparse 全量矩阵上，Uni-STC 相对 DS-STC 和 RM-STC 的几何平均 speedup 为 3.35x 和 2.21x，energy reduction 为 1.97x 和 1.27x。

从硬件架构角度拆解术语：

Uni-STC 作为 GPU SM 内的独立 coprocessor 工作，其内部硬件流水线分为三阶段：
1. **TMS (Task Merge & Splitting)**：从 Meta Buffer 读取 top-level bitmap，将 16×16×16 T1 任务沿 M/N/K 三维对称拆分为 4×4×4 T3 task，支持 M-merge/K-merge 和 task ordering（在 outer-product 与 row-major 间动态选择以提升 A/B tile 复用）
2. **DPG (Dot-product Generation unit) ×8**：并行读取 bottom-level bitmap，对 A/B tile bitmap 做 overlay 生成 4-bit sparse dot-product pattern，生成 8-bit T4 task code 写入 Dot-product queue；采用 Z-shaped 任务分配顺序；支持 dynamic activation——TMS 根据 queue head 的 intermediate product prefix sum 决定需开启 DPG 数量，多余 DPG 及关联 network 被 power-gate
3. **SDPU (Segmented Dot-Product Unit)**：弹出合并后的 T4 task，执行 1×1×4 segmented dot-product，累加到 1KB accumulator buffer；merge-forward 结构在写 C 前预合并最多四个 partial products，减少 C 网络和写回流量

硬件集成要点：SM 需更新 instruction decoder 解析 UWMMA opcode，扩展 warp scheduler 分发指令。数据交互通过 register file 和 operand collector 完成。两层 Benes/MUX network 主要搬运控制信息（T4 code）而非全量数值数据。面积评估：在 A100 类 826 mm² die 上 432 个 Uni-STC 单元额外面积约 0.0425 mm²（~2.12%），critical path 满足 1.5 GHz。

术语一般如何实现？如何使用？

Uni-STC 通过 UWMMA 指令序列由软件驱动：`stc.load` 同步收集 BBC 格式的 metadata 和 values 到硬件 buffer → `stc.task` 异步触发 TMS/DPG 生成 task queue → `stc.numeric` 检查 READY/BUSY 状态驱动 SDPU 执行，完成后结果写回 register file。软件预处理需将稀疏矩阵从 CSR/CSC 离线转换为 BBC 格式（64-core CPU <1000ms 或 A100 GPU <100ms），可在 GNN training、linear solver 等迭代应用中摊销。评估使用 Accel-Sim 扩展的 STC simulator，面积/能耗用 Yosys+FreePDK45+CACTI 7 建模。

涉及论文标题：
- Uni-STC: Unified Sparse Tensor Core

