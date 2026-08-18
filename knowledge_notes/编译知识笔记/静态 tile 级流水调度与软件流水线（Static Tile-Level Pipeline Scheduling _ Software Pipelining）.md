## 静态 tile 级流水调度与软件流水线（Static Tile-Level Pipeline Scheduling / Software Pipelining）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 论文 baseline 概念：当前 AI 加速器（CUDA/TensorRT/XLA 编译器与手调 kernel 如 FlashAttention-3）普遍采用的编译期编排流水模板——编译器把算子 tiling/fusion 后按固定阶段排布，用显式同步屏障（GPU 的 PTX bar.sync、NPU 的 compiler-placed fence）协调 tensor/vector/DMA 异构单元，发射确定性指令流、硬件按固定时序执行。静态 tile 级流水调度是其形式化：FA3 融合注意力被分成双阶段（M0^i+S^i 在前、M1^i 在后，图 2b）或三阶段（每个 tiled 算子单独成级，图 2d）的固定模板，通过 bar.sync 等固定同步点锁定执行顺序。
- 论文指出四大局限：① 编译期跨单元依赖协调与重排探索对应最小 makespan 与 bin packing 两个 NP-hard 问题，只能靠架构相关启发式 + 手工调优；② 抽象粒度失配——CUDA stream/图级运行时粒度太粗（遮蔽 tile 边界与数据依赖），warp 指令级太细（语义丢失）；③ 无法适应运行时变动——静态调度假设固定延迟与带宽，真实系统的 DMA backpressure、cache/SPM bank 冲突、热节流、OS 效应导致单元失同步，静态模板无法重定长/重排；④ 历史先例——超标量 CPU 靠动态调度获高 ILP，静态 VLIW/IA-64 因不能泛化到工作负载与微架构变动而失败。
- 软件流水线（software pipelining / modulo scheduling）是本领域的经典技术：编译期用可预测延迟与硬件谓词重叠循环迭代（如 CPU 的 modulo scheduling）；论文指出 AI 加速器不满足其前提——非确定性执行延迟（DMA backpressure、SMEM bank 冲突、热节流）、缺乏重型张量/向量单元的硬件谓词、异构多单元协调超出经典 modulo scheduling 范围，因此它最多平滑静态阶段结构内的流水气泡，不能改变阶段化执行模板（FA3 仍靠固定 barrier 锁定）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 编译框架中的运转：编译器在编译期做 fusion → tiling → 阶段划分 → 重排 → 插入同步屏障 → 发射指令流。例子（FA3 CUDA 伪代码，论文 Fig.5 左）：tma_load_q/tma_load_k_transpose → warpgroup_fence_producer → wgmma::mma_sync（M0）→ wgmma::wait + softmax_warpgroup（S）→ 循环内 tma_load_k_next + wgmma::mma_async（预取下一 K tile）、tma_load_v + wgmma::mma_sync（M1）→ rescale_warpgroup + warpgroup_commit_batch + update_carousel_index——8 处显式同步把 S^i 与 M0^{i+1}（数据独立、资源不冲突）序列化，迭代间存在隐式屏障。
- 效果对比：Naive（无重排）→ Static（软件流水 + 重排）在 Epoch 上 1.03–1.69×；但 Dynamic（硬件动态调度）再高 1.14–1.63×，且 Accumulated Overlap Score 动态达静态的 1.10–5.17×。静态流水的最优阶段边界需穷举多参数编译期探索或启发式换取次优，动态调度用运行时语义机会性利用安全并行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：编译器（XLA/TensorRT/CUDA nvcc、TVM/Triton 的编译期调度）产出固定阶段模板 + 同步点；GPU 上 bar.sync/命名屏障、NPU 上 fence。论文实验中的 TISA+Static 配置即"同编译器优化 + 编译期静态软件流水 + fence 手工依赖管理"，作为强静态 tile 级流水调度基线。
- 使用场景与限制：适用于延迟可预测、单元同构、可硬件谓词的平台（经典 CPU）；在 AI 加速器上因非确定性延迟与异构单元而锁死利用率。这也是论文转向"把有界语义感知的调度移到运行时"（类比超标量 vs VLIW）的直接动机。
- 相关：E0+E2=E1+E3（论文图 2）说明双阶段/三阶段静态模板的等效延迟节省，而动态调度（图 2c/e）无需固定阶段定义即可获得同等甚至更紧的跨迭代重叠。

涉及论文标题：
- Dynamic Scheduling for AI Accelerators via TISA
