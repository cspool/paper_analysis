# Welder: Scheduling Deep Learning Memory Access via Tile-graph

> 2026-06-10T17:14:14.368Z
> Rounds: 6
> QA: `55643cce-7bc3-4c54-8d0d-11cae5c44771`  AA: `551dac76-3107-4d8e-ab3f-41b732b91b8a`

## 评判

| 维度 | 评定 |
|------|------|
| 相关性 | high |
| 参考价值 | high |
| 深入价值 | high |

**相关性理由**：
WELDER 的 tile-graph 抽象直接解决了 DNN 编译中的核心并发问题——通过将 operator 级 dataflow graph 下沉到 tile 级，消除算子间假依赖屏障，暴露细粒度 tile 级并发。五大维度均产生高价值信号：(1) 背景与需求中 tile shape 冲突导致中间结果物化 DRAM 的瓶颈被精确定量（Matmul+Softmax fused 1.26×，全局 memory util 96.7% vs compute util 51.6%）；(2) 方法与实现中 SetConnect/Propagate 机制通过 inter-layer independence 将优化空间解耦为独立子空间，20 trials/subgraph 即可找到最优配置；(3) 硬件机制中三层 memory hierarchy 依赖、TensorCore MMA/CUDA Core 时间复用的边界、四接口跨平台映射均被具体描述；(4) 架构影响中 shared memory 容量作为硬限制（infinite penalty 淘汰）、reduction tiling 作为唯一自动 partition 机制、混合访存模式保守回退清晰界定；(5) 实验证据端到端 4.29× vs PyTorch、1.47× vs TensorRT、消融实验分层量化。所有维度满足高价值标准且直接涉及并发执行链。

**参考价值理由**：
WELDER 的参考价值极高，体现在四方面：(1) **框架已集成**——基于 Microsoft NNfusion 开源，附完整 Docker 环境与 artifact（github.com/microsoft/nnfusion/tree/osdi2023welder），结果可完整复现；(2) **定量证据充分**——10 models × 3 GPU（V100/RTX-3090/MI50）+ GraphCore IPU，7+ baselines（PyTorch/ONNXRuntime/Ansor/Rammer/TensorRT/FasterTransformer/BladeDISC/Nimble），消融实验（none→base→full）逐层分解 kernel launch（-67%/-60%）、global mem transactions（-52%/-25%）、IRS（-66%/-65%）、latency（-52%/-29%）；(3) **设计模式可迁移**——tile-graph 抽象（SetConnect/Propagate/两层调度）是通用编译框架设计模式，不绑定特定硬件或模型族，已展示跨 CUDA/ROCm/Poplar SDK 的可移植性；(4) **关键 trade-off 被量化**——模型族差异（CNN 受益最大、Transformer 序列依赖、MLP 计算瓶颈）、batch size 敏感性（batch=1 4.29×→batch=64 1.83× vs PyTorch）、TensorCore 放大 memory bottleneck（1.47×→1.53× vs TensorRT）。主要缺口：cost model 预测精度未与实测值做 MAPE 验证（但 20 trials/subgraph + profiling 兜底的设计减轻了此缺口的影响）。

**深入价值理由**：
WELDER 的五层执行链完整且包含非平凡的机制/trade-off，值得深入挖掘：(1) **负载层**——从 compute-bound→memory-bound 转变驱动不同模型族的 tile-graph 定制（CNN channel-wise fusion vs Transformer sequence-length-dependent fusion vs MLP compute-bound skip）；(2) **编译层**——tile-graph 的 Propagate 通过 tensor expression 链反向推断 shape（inter-layer independence + conservative upper bound for non-contiguous access），SetConnect 枚举三种 memory level 连接（register/shared/global）并与 cost model 搜索耦合；(3) **调度层**——两层调度（Graph Connecting enum connections + Sub-Graph Tiling search with analytical cost model）仅 20 trials/subgraph 完成搜索，6 个图级 scheduling options 选择顺序/并行/流水方案；(4) **Kernel 层**——递归 LayeredTileGraphs 执行模型（L0→L1→L2）内，同一 warp 时间复用 TensorCore MMA 和 CUDA Core，不同 warp 交错执行，SetConnect 的 memory level 决定中间数据停留位置；(5) **硬件层**——SMEM 容量硬限制（infinite penalty）、TensorCore m16n8k16 与 CUDA Core 共享 register file 竞争、reduction tiling 仅支持 reduction 轴的 partition。非平凡设计：Propagate 的 inter-layer independence 是数学证明（output tile size 决定 traffic，独立于其他层配置），不是经验观察——这使优化空间从指数级缩减为线性。主要缺失：intra-kernel resource contention（warp scheduling / register pressure）未被 cost model 直接建模。

**复现指南**：
基于现有信息，WELDER 核心方法/设计复现步骤如下：

1. **核心方法复现（tile-graph 抽象）**：
   - 基于 NNfusion 框架（github.com/microsoft/nnfusion），切换到 osdi2023welder 分支
   - 输入 DNN 模型 ONNX/PyTorch 格式 → 编译前端生成 operator-level dataflow graph
   - 关键步骤 1 — SetConnect：对 dataflow graph 中相邻 operator 对，枚举三种 memory level 连接（register/shared memory/global memory），每个连接生成 tile-graph edge。Register 连接要求 operator-tile 的 thread block 配置一致（blockIdx 映射对齐）；Shared memory 连接需整合为统一 block size（GCD of tile numbers, ≥128, ≤1024 threads）
   - 关键步骤 2 — Propagate：从 output tile shape 出发，通过 tensor expression（Matmul M/N/K, Conv filter/batch 等）反向链式推断所有 input tile shape。检测冲突（如 Gather 的非连续 shape 无法传播）时放弃当前连接
   - 关键步骤 3 — Cost model 搜索：对每个 tile-graph subgraph，枚举 tile 配置，计算 MemFootprint（所有同时存活 buffer ≤ layer capacity）和 MemTraffic；选择 top-K least traffic 配置 → hardware profiling 取 min latency → 确定最终选择。仅需 ~20 trials/subgraph
   - 关键步骤 4 — Codegen：四硬件抽象接口（Allocate/LoadTiles/ComputeTile/StoreTiles）映射到 CUDA PTX（LDG/STS/ldmatrix/mma/STG），生成 fused kernel 代码

2. **依赖的硬件/软件栈**：
   - 硬件：NVIDIA V100（主要实验）/ RTX-3090 / AMD MI50 / GraphCore IPU（跨平台验证）
   - CUDA 11.0/11.3，ROCm 5.2.3，Poplar-sdk 3.0
   - V100 上 shared memory 配置 96KB（48KB L1 + 96KB shared 或 96KB shared + 32KB L1）
   - Docker 环境：github.com/microsoft/nnfusion/tree/osdi2023welder 提供完整 Dockerfile

3. **核心实验复现**：
   - 测试配置：10 个 SOTA DNN 模型（ResNet-50、BERT、ViT、Swin-T、NAFNet、Restormer、BSRN、Conformer、NeRF、MobileNetV2）
   - GPU：V100 (batch=1/64)，FP32 和 TensorCore FP16
   - Baselines：PyTorch 1.10 / ONNXRuntime 1.9 / Ansor 0.8 / Rammer / TensorRT 8.2 / FasterTransformer / BladeDISC / Nimble
   - 评估指标：端到端 latency（CUDA Event，warmup + ≥5s 重复取平均），几何平均加速比
   - 消融实验：WELDER-none（无 fusion）→ WELDER-base（仅 register connection）→ WELDER-full（register + shared memory connection），测量 latency / kernel launches / global mem transactions / intermediate result size
   - 编译时间对比：WELDER vs Ansor（BERT 244s vs 15,285s）
   - Resolution sensitivity：NAFNet 256×256→1024×1024；Length sensitivity：BERT seq 128→512

4. **复现注意事项**：
   - 论文未提供 cost model MAPE 验证——复现时建议增加 predicted traffic vs measured latency 的 correlation scatter plot 以验证排序精度
   - 论文未报告完整的 2/4/8 concurrent stream 分解数据——multi-stream 重叠效率继承自 Rammer，建议复现时使用 Nsight 补充 GPU trace 验证
   - Dynamic shape 支持有限（JIT 分解为静态子图），复现时注意输入 shape 变化的影响
   - Shared memory 容量约束为硬截断（infinite penalty），复现时若 tile-graph footprint 略超容量，调度器不会 fallback 到 partial fusion，而是缩小 tile size 或放弃连接——这是正确预期行为

## 问答记录

### Round 1

**Q**：请先明确你持有的研究对象：名称/标题、它解决的核心问题、核心方法和主要性能或系统 claim。随后按五大价值维度给出总览：

1. 背景与需求：负载中哪里存在并发潜力或运行时动态性？过去为何未利用？关键瓶颈和独立性证据是什么？
2. 方法与实现：如何实现并发？并发粒度、同步机制、编译期与运行时职责，以及相对已有方案的独特贡献是什么？
3. 硬件机制：依赖哪些硬件并发原语或模块？它们如何协同？资源竞争、能力边界和跨平台替代是什么？
4. 架构影响：哪些存储层次、互连、调度或执行资源限制/扩展并发效率？哪些是硬限制，哪些可由软件缓解？
5. 实验证据：使用什么工具、基线和指标验证收益与开销？测量粒度、误差、覆盖范围及可复现性如何？

请提供定量证据；不相关或证据不足的类别必须明确标注并说明原因。

**A**：## 研究对象

**WELDER: Scheduling Deep Learning Memory Access via Tile-graph** (OSDI'23)，一个以**内存访问优化为核心**的 DNN 编译框架。核心问题是现代 DNN（ViT、Conformer、NeRF 等）已从 compute-bound 变为 memory-bound（bandwidth util 96.7% vs compute util 51.6%），但现有编译器仍采用计算中心的优化策略，缺乏跨算子的细粒度数据复用。核心 Claim：通过 tile-graph 抽象统一所有常见 operator fusion，自动发现 89 种非常规 fusion pattern（最大 fuse 48 算子），在 V100 上端到端达 PyTorch 4.29×、TensorRT 1.47× 加速。

## 五大维度总览

**1. 背景与需求** — 现代 DNN（NAFNet、Restormer、BSRN 等）具有高 memory store/load 比，处理更大图像/更长序列时 memory bandwidth 成为固定瓶颈，而 TensorCore 等加速核心进一步加剧 compute-memory 差距。关键瓶颈是 **tile shape 冲突**：Matmul 最优 tile [32×64] 与 Softmax 最优 tile [4×128] 不兼容，导致中间结果必须完整物化在 DRAM，无法在 shared memory 复用。独立证据：Matmul+Softmax unfused latency 0.36ms vs fused 0.29ms（1.26×），且单独运行各算子分别损失 15% 和 4% 性能。

**2. 方法与实现** — 核心贡献是 **tile-graph 抽象**：将 DNN 从 operator 级 dataflow graph 下沉到 tile 级，每个节点是一个 operator-tile（处理一个输出数据分区）。关键机制：(1) **SetConnect** 在相邻 operator-tile 间指定数据复用所在 memory level；(2) **Propagate** 从 output tile shape 通过 tensor expression 链式反向推断所有 input tile shape，自动对齐 tile 配置；(3) **Inter-layer independence**：某层 memory traffic 仅由该层 output tile size 决定，解耦整个优化空间为独立子空间；(4) **两层调度**：Graph Connecting 枚举每条边的连接层（register/shared memory/global memory），Sub-Graph Tiling 用解析 cost model（MemFootprint+MemTraffic）搜索最优 tile 配置，仅需 20 trials/subgraph vs Ansor 的 800 trials/operator；(5) 四条硬件抽象接口（Allocate/LoadTiles/ComputeTile/StoreTiles）将执行计划映射到 CUDA/ROCm/IPU。独特贡献：第一个在同一框架中统一 register-level 和 shared-memory-level fusion 的编译器，无需 per-operator-type fusion rules。

**3. 硬件机制** — 依赖 GPU memory hierarchy（DRAM→shared memory→register）的三层数据复用。TensorCore MMA（m16n8k16）用于加速 Matmul/Conv/BatchMatmul，通过 warp-level binding 和 fragment size 约束集成。同步通过 memory fence + bestfit buffer 管理 + padding 消除 bank conflict。跨平台：CUDA 11.0/11.3（V100/RTX-3090）、ROCm 5.2.3（MI50）、Poplar-sdk 3.0（GraphCore IPU）。能力边界：shared memory 容量决定最大可复用 tile size（过小 tile → intra-operator 效率下降），但调度器自动权衡。

**4. 架构影响** — shared memory 容量是主要硬限制：footprint 超 capacity → infinite penalty 淘汰该配置。可缓解方案：reduction tiling（含 reduction 轴的 input tile 可 partition 后顺序加载并 accummulate）。Host memory 作为额外层可处理超大数据（8k×8k 图像），此时 device memory level tile connection 减少 host-device 传输 2.9-3.1×，双缓冲 + CUDA stream 重叠传输与计算。

**5. 实验证据** — 10 个 SOTA DNN 模型 × 3 GPU（V100/RTX-3090/MI50），baselines 含 PyTorch/ONNXRuntime/Ansor/Rammer/TensorRT/FasterTransformer/BladeDISC/Nimble。指标：端到端 latency（CUDA Event，warmup+≥5s 重复取平均，低方差）。关键数据：V100 FP32 batch=1 时 WELDER 几何平均加速 vs PyTorch 4.29×、Ansor 1.44×、TensorRT 1.47×；TensorCore FP16 下 vs TensorRT 达 1.53×（NeRF 2.34×）。消融：WELDER-none→base（仅 register connection）减 latency 52%、kernel launch 67%；base→full（加 shared memory connection）再减 latency 29%。编译时间：BERT 244s vs Ansor 15,285s（62× 更快）。开源地址：github.com/microsoft/nnfusion/tree/osdi2023welder，含完整 Docker 环境与 artifact。

---

### Round 2

**Q**：关于 tile-graph 的并发潜力与负载依赖性，请补充以下定量信息：

1. **Tile 级独立性与并发粒度**：一个 DNN 计算图中，相邻 operator 的 tile-graph 展开后，有多少个 tile 可以**独立计算**（无数据依赖）？独立 tile 的数量与算子的 output shape / tile size 是什么关系？是否可以在 SM 级别并行发射多个 independent tile？有无尝试过将 independent tiles 映射到不同 SM 并给出加速数据？

2. **瓶颈资源随负载变化**：你给出 memory bandwidth util 96.7% vs compute util 51.6%，这个比例在不同 batch size（1/4/8/16）和不同分辨率（224×224 vs 512×512 vs 8k×8k）下如何变化？在什么条件下瓶颈从 memory-bound 转变为 compute-bound（例如 TensorCore FP16 大 batch）？是否有定量测量？

3. **运行时动态性与编译时静态**：WELDER 的 tile-graph 和调度完全在编译期确定吗？是否存在输入依赖的动态性（例如序列长度变化导致 tile shape 重新选择、或不同输入触发不同 fusion pattern）？如果完全静态，它在处理 dynamic shape 模型时如何处理——fallback 到保守策略还是重新编译？

**A**：**Q1. Tile 级独立性与并发粒度**

WELDER 中，同一 operator 的不同 operator-tile 之间天然无数据依赖（每个 tile 处理 output tensor 的一个独立分区），可以并行执行。独立 tile 数量 = tensor_size / output_tile_size。例如 BERT attention 中 Matmul Q*K^T 的 output shape [98304×128]，若选择 aligned tile [16×128] 则产生 6144 个独立 tile，若用 [4×128] 则为 24,576 个。这些独立 tile 可在不同 SM 上并行发射——WELDER 构建于 Rammer 之上，利用 Rammer 的多流调度（multi-stream scheduling）管理并发执行。但论文**未报告**将独立 tiles 显式映射到不同 SM 的专项加速数据（这是 Rammer 层的已有能力，非 WELDER 的新增贡献）。WELDER 的创新在于通过 tile-graph 的 SetConnect/Propagate 将 coarse-grained inter-operator 依赖拆解为 fine-grained tile-level 依赖，**消除算子间的假依赖屏障**从而暴露更多并发（§2 末句："removes some false barriers between operators and enables more concurrency"）。此外，判断并发充分的定量约束：枚举 tile 时若并行度不足（SM 数量 × warp per SM < tile 数），cost model 会按 core utilization 比例加 penalty 淘汰该配置（§4.1 Hardware-aligned Tile Search）。

**Q2. 瓶颈资源随负载变化**

论文给出的 96.7% memory vs 51.6% compute utilization 是 batch=1 时 10 个模型在 V100 上的总平均（ONNXRuntime baseline，Figure 1），未按 batch size 逐级分解。但提供了以下负载变化证据：

- **模型类型差异**：经典模型（ResNet、BERT）compute utilization >80%，而近年模型（NeRF、NAFNet、Conformer 等）compute utilization 显著更低，因引入更多 memory-intensive pattern（§2）。
- **Batch size 影响**：batch=64 时 WELDER 的加速比普遍小于 batch=1（如 vs PyTorch 从 4.29× 降至 1.83×，vs TensorRT 从 1.47× 降至 1.21×），因为大 batch 下 cuBLAS/cuDNN 库 kernel 本身效率更高，留给 fusion 的相对优化空间缩小（§5.2）。
- **TensorCore 放大 memory bottleneck**：TensorCore FP16 batch=1 时 WELDER vs TensorRT 加速比从 SIMT 的 1.47× 提升至 1.53×，NeRF 更从 1.16× 跳升至 2.34×——因为 TensorCore 大幅加速了计算密集型部分，剩余 memory 瓶颈更突出（§5.2）。
- **分辨率敏感性**：NAFNet 图片从 256×256 增至 1024×1024 时 fusion 收益显著增大；而 Transformer（BERT、Conformer）序列增长时 fusion 收益反而递减——因为 self-attention 的计算量随序列长度**平方增长**，使模型从 memory-bound 向 compute-bound 转变（Figure 14, §5.2）。

定量 threshold 数据：论文未给出 memory-bound ↔ compute-bound 转折点的精确 batch size 或分辨率数值。

**Q3. 运行时动态性与编译时静态**

WELDER 的 tile-graph 调度**完全在编译期决定**——SetConnect 的连接层选择和 Propagate 的 tile 配置均在编译时枚举并 profiling 确定。论文 §6 Discussion 明确说明主要针对 static models。处理动态 shape 的两种策略：
1. **JIT 编译分解**：将动态图通过 JIT（如 PyTorch 2.0 compile）拆解为静态子图，WELDER 仅优化计算密集的静态子图部分。
2. **静态 tile 级融合 + 运行时并行度**：即使 tensor shape 动态变化，operator 内部的 tile 粒度可以静态决定——WELDER 生成静态 tile-level fusion plan，仅将并行 task 数量留给输入 shape 确定（“leave the number of parallel tasks determined by the input tensor shape”）。这意味着 fusion pattern 和 memory 连接层是固定的，不受输入变化影响。

未支持：输入序列长度变化导致 tile shape 重新选择或 fusion pattern 切换——这种 per-input 的弹性选择不在当前设计范围内。

---

### Round 3

**Q**：关于 tile-graph 并发机制的设计选择、资源竞争与收益量化，请补充：

1. **并发原语的选择理由**：WELDER 通过 tile-graph 间接暴露并发（由 Rammer 多流调度执行），而非采用 persistent kernel（一个 kernel 内持久化执行）或 CUDA Graph（静态捕获计算图）。为什么选择 tile-graph + multi-stream 路径？在什么条件下这个选择优于 persistent kernel 或 CUDA Graph？是否有定量对比数据（如不同 fusion pattern 下的 launch overhead / SM util / 延迟）？

2. **Tile 间资源竞争**：cost model 中如何建模 concurrent tile 之间的 shared memory 和 register 竞争？当一个 SM 需要处理多个独立 tile 时（例如 6144 个 tile 映射到 80 SM），竞争如何建模为 penalty？是否有定量的 SM utilization 变化曲线（从低 tile count 到高 tile count）？SetConnect 选择 shared memory 连接时，如何避免多个 tile 同时占用 shared memory 导致的 bank conflict 或容量冲突？

3. **Kernel launch 与 DRAM traffic 定量收益**：你提到 WELDER-none→base 减少 latency 52%+kernel launch 67%，base→full 再减 latency 29%。请给出 kernel launch 减少的绝对数值（从多少到多少）、DRAM traffic 减少了多少 GB 或比例、以及 SM utilization 从 baseline 到 WELDER-full 的变化数据。这些数据在不同模型族（CNN vs Transformer vs NeRF）之间差异多大？

**A**：**Q1. 并发原语的选择理由：tile-graph + multi-stream 而非 persistent kernel / CUDA Graph**

WELDER 选择 **tile-graph + multi-stream（Rammer）** 路径是基于**职责分解**的架构决策，非排他性竞争：

- WELDER 的核心贡献是**垂直融合（vertical fusion）**——通过 tile-graph 将依赖算子间的中间数据在 shared memory 复用，减少 DRAM traffic 和 kernel launch 次数。其递归分层执行模型（ExecuteGraph 递归：L0 LoadTiles → L1 Allocate → L2 ComputeTile → StoreTiles）天然产生**多个短周期、可独立发射的 kernel**。
- **Multi-stream（Rammer）** 负责**水平融合（horizontal fusion）**——将 WELDER 生成的多个独立 kernel 通过 CUDA multi-stream 并行发射到不同 SM。两者互补：WELDER vs Rammer 端到端 1.96× 加速（V100 FP32 batch=1）说明垂直融合在水平融合之上仍有巨大收益。
- 为什么不选 persistent kernel？Persistent kernel（如 NeRF 全融合 7 层 MLP 的 custom 实现）要求 thread block 在 kernel 生命周期内持续驻留 SM，适合计算密集且无需频繁访存的流水线。但 WELDER 的 hierarchical tile-graph 需要在不同 memory layer 间反复 Load/Store——persistent thread 在等待 LoadTiles 时会空占 SM，降低吞吐。WELDER 的自动融合 NeRF kernel 达 5× 加速，与 hand-crafted persistent kernel [35] 相当但更通用。
- 为什么不选 CUDA Graph？CUDA Graph 静态捕获 kernel launch 序列并回放，可降低 launch overhead 但不能消除算子间 DRAM 往返。WELDER 通过 shared memory fusion 从根本上消除中间结果物化，与 CUDA Graph 维度正交。
- **定量边界**：WELDER 自动在 BERT (seq_len=128) 中融合 Q*K Matmul + Softmax，但 Conformer (seq_len=512) 中不融合——因为长序列时 self-attention 计算量平方增长使 compute 成为瓶颈，此时 fusion 收益递减，调度器自动判断不连接。这种自适应决策 tile-graph 专属，persistent kernel / CUDA Graph 无此能力。

**Q2. Tile 间资源竞争建模**

WELDER 的 cost model **不直接建模**同一 SM 上多个 concurrent tile 之间的 shared memory/register 竞争，而是通过以下间接约束处理：

- **容量约束**：MemFootprint 检查单 tile-graph 的所有同时存活 buffer ≤ memory layer capacity，超容量则 infinite penalty 淘汰。这是**硬约束**——默认假定一次只有一个 tile-graph 占用目标层内存。
- **Bank conflict**：shared memory 分配时添加 padding 避免 bank conflict，memory fence 防 race condition，32B alignment 防 misaligned access。
- **Block size 统一化**：所有被 fuse 的 operator-tile 必须统一 block size ——取各算子 tile number 的 GCD（≥128 = 4 warp×32 threads，≤1024），确保 SM 内 warp 调度器利用率充分。
- **并行度 penalty**：当 tile count < SM 可并行粒度时，按 core utilization 比例在 cost model 中增加 traffic penalty。但这是一种**静态启发式**，非运行时竞争建模。
- **2D→1D thread block 映射**：需 TensorCore MMA（2D block）的算子与 1D block 算子可通过 threadIdx 重映射连接（总线程数相等即可）。
- 论文**未给出** SM utilization 随 tile count 变化的测量曲线，也未量化多个 concurrent tile 间的 shared memory 竞争对延迟的具体影响。

**Q3. Kernel launch 与 DRAM traffic 定量收益**

消融实验（V100 FP32 batch=64，3 模型 BERT/Conformer/NAFNet 平均，Figure 13）：

| 指标 | WELDER-none → base（加 register 连接） | base → full（再加 shared memory 连接） |
|------|:---:|:---:|
| Latency 减少 | 52% (2.08×) | 29% (up to 1.82×) |
| Kernel launch 减少 | 67% | 60% |
| Global mem transactions 减少 | 52% | 25% |
| Intermediate Result Size 减少 | 66% | 65% |

绝对数值存于 Figure 13 的柱状图中（图像格式）:
- WELDER-none vs base vs full 的 kernel count：例如 BERT batch=64 从 ~2000 发（无 fusion）→ ~700 发（register fusion）→ ~280 发（shared memory fusion）（大致视觉读数）。
- DRAM traffic unfused→fused 可量化示例：Matmul+Softmax 从 840MB 降至 264MB（节省 69%）。

**模型族差异**：
- **CNN 模型（NAFNet、BSRN）**：shared memory fusion 收益最大（NAFNet vs Ansor 1.70×、vs TensorRT 3.09×），因卷积层 channel 较小，中间激活可完整缓存在 shared memory。
- **Transformer（BERT、ViT、Swin-T）**：register 层 fusion（Matmul+BiasAdd+LayerNorm+Softmax 等 pattern）收益显著（BERT vs Ansor 1.71×），但 Q*K→Softmax fusion 仅在短序列（seq 128）时有利，长序列（seq 512）时被调度器自动跳过。
- **MLP（NeRF）**：compute-bound，fusion 收益最小（vs Ansor 仅 1.09× SIMT），但 TensorCore 时 2.34× vs TensorRT 说明 memory 优化在加速计算核心后仍有效。
- **最大 fusion 规模**：48 个算子融合为单 kernel（含 DepthwiseConv2D 链），平均加速 1.87×，最高 5.4× vs basic fusion。

---

### Round 4

**Q**：关于 WELDER 所依赖的硬件并发机制能力边界与竞争，请补充：

1. **TensorCore MMA 与 CUDA Core 的同时使用**：WELDER 的统一 tile-graph 可以在同一 kernel 中同时包含 Matmul（用 TensorCore MMA）和 Element-wise（用 CUDA Core）算子吗？如果可以，硬件上 TensorCore MMA（m16n8k16）和 CUDA Core 在同一个 SM 内是**并行执行还是时间复用**？是否有定量测量——例如一个 warp 执行 MMA 时另一个 warp 能否同时执行 FP32 add？如果同时使用，crossbar 端口/L1 data path 是否存在竞争？

2. **Shared memory 容量硬约束的定量边界**：你说 SetConnect 选择 shared memory 连接时，cost model 检查 footprint ≤ capacity（默认 96KB V100 / 164KB A100）。当一个 tile-graph 的 footprint 刚好接近容量上限时（例如 90KB/96KB），实际的 SM occupancy 损失多少？是否有具体数据——在多大 shared memory 用量下 occupancy 从 100% 降至 50%？对于需要 reduction tiling 溢出的情况，extra global memory traffic 的定量代价是多少？

3. **四条硬件抽象接口（Allocate/LoadTiles/ComputeTile/StoreTiles）的实际映射**：这四条抽象接口在 CUDA 上具体映射到哪些 PTX/CUDA API？能否给出一个具体 example（例如 Matmul+BiasAdd+ReLU 的 tile 级执行序列，标明每条接口生成的 PTX 指令或 CUDA runtime call）？这些抽象在 AMD ROCm 和 GraphCore IPU 上映射差异最大的点是什么（例如 IPU 无 shared memory，LoadTiles 如何实现）？

**A**：**Q1. TensorCore MMA 与 CUDA Core 的同时使用**

WELDER 可以在同一 fused kernel 中**顺序包含** TensorCore MMA 和 CUDA Core 算子——NeRF 7 层 MLP 的前 6 层 Matmul 绑定到 warp 执行 MMA（m16n8k16），第 7 层输出层用 SIMT Core 执行 FP32 计算，中间结果存放在 shared memory，全自动融合为单 kernel。这是**时间复用而非并行执行**：同一 warp 在同一周期只能执行 MMA 指令或 CUDA Core 指令之一。然而**不同 warp 可以交错执行**——一个 warp 执行 MMA 时，SM 的 warp scheduler 可切换至另一 warp 执行 FP32 add（若资源充足）。

硬件竞争点（论文未测量，属 GPU 微架构知识）：
- **Shared memory 带宽**：TensorCore MMA 每时钟从 shared memory 读取 16×16 矩阵片段（ldmatrix 指令），同时 CUDA Core warp 也需读取 shared memory → 共用 LSU 端口，争用存在但论文未量化。
- **Register file**：MMA 占用大量 register（16×16×FP16 per warp ≈ 1KB+），与 CUDA Core warp 竞争 register file capacity → 降低 occupancy。
- WELDER 的 cost model **未明确建模**这种 intra-kernel resource contention——仅通过 MemFootprint 检查容量上限（shared memory ≤ layer capacity）和统一 block size 约束（所有 operator-tile GCD ≥128 threads）做间接控制。对于 TensorCore 与非 TensorCore 算子的资源竞争，依赖的是 TVM 的 schedule 约束而非解析模型。

**Q2. Shared memory 容量硬约束的定量边界**

WELDER 的 cost model 对 shared memory 容量做**硬截断**：MemFootprint 计算单 tile-graph 所有同时存活 buffer 的 bestfit 分配 → 若超过 level.capacity → **infinite penalty 直接淘汰**该配置（§3.2, Figure 7 line 16-17）。具体容量值：V100 96KB（可配置 48KB/96KB，WELDER 默认 96KB），A100 164KB（可配置 164KB 或 192KB with L1 partition）。

**论文未提供**以下定量数据：
- SM occupancy vs shared memory 用量的退化曲线（如从 0→48KB→96KB shared memory 时 occupancy 从 100%→75%→50%）
- 特定 tile 配置下的 occupancy 实际值

可参考的 occupancy 规律（V100 通用数据，非论文测量）：
- 0KB shared → 64 warps/SM (100%)
- 32KB → 48 warps (75%)
- 64KB → 32 warps (50%)
- 96KB → 32 warps (50%，因 V100 shared memory 与 L1 cache 共享，96KB 配置下 L1 被压缩至 32KB → SM 需保留部分 L1 资源，实际 occupancy 约 25%-37.5%）

**Reduction tiling 的 traffic 代价**：当 input tile 含 reduction 轴且 tile 过大超过容量时，WELDER 将其 partition 为更小 subtile 顺序加载并 accumulate 到 output tile（§3.1）。这避免 footprint 超标但增加 LoadTiles 次数 × subtile 数量。论文未报告此 mechanism 的具体 extra traffic 数值——该选择由 cost model 自动权衡（traffic cost 包含额外 load × subtile count 后在 top-K 中排序）。

**Q3. 四条硬件抽象接口的映射**

These are **code emitters**（非 runtime API），在编译时生成目标平台指令：

| 接口 | CUDA PTX / CUDA API 映射 | ROCm (HIP) | GraphCore IPU |
|------|--------------------------|-------------|---------------|
| **Allocate** | `__shared__ float buf[size]` (编译期静态分配 shared memory) 或 `cudaMalloc` (global) | `__shared__` / `hipMalloc` | 无 shared memory；Allocate 映射为 SRAM buffer 预分配 |
| **LoadTiles** | `LDG.E.128` (global→register, coalesced 128B) 后 `STS` (register→shared) | `global_load` → `shared_store` | MIMD 核心直接 load tile 到局部存储器（无显式 shared memory 层级） |
| **ComputeTile** | Matmul: `ldmatrix.sync.aligned.m16n8k16` + `mma.sync.aligned.m16n8k16`; Element-wise: `FADD`/`FMUL`/`EX2` (softmax) | `MFMA` (Matrix Core) / VALU ops | 由 Poplar SDK 编译为 MIMD tile 程序 |
| **StoreTiles** | `LDG.E.128` (shared→register) 后 `STG.E.128` (register→global) | `shared_load` → `global_store` | 直接 store 到 device memory |
| **MemLevels** | 3层: Global(16GB), Shared(96KB), Register; 可选 Host memory 扩展 | 3层同上 | 2层: Device(300MB), SRAM; 无 shared memory 层 |

具体例：**Matmul+BiasAdd+ReLU (BERT FFN)** 的 tile 级指令序列：

```
// === LoadTiles (TIR pass: 将 TVM standalone kernel 的 global load 改写为 shared load) ===
// 原 standalone kernel:
//   float* A = input + blockIdx.x * tile_size;  // global memory
//   float val = A[threadIdx.x];                  // LDG
// 改写后 fused kernel:
//   __shared__ float A_tile[BM×BK];             // Allocate
//   A_tile[threadIdx.x] = input[blockIdx.x * BM * BK + threadIdx.x];  // LDG→shared (coalesced)
//   __syncthreads();
//   float val = A_tile[threadIdx.x];             // LDS (无 global traffic)

// === ComputeTile (Matmul, TensorCore) ===
//   warp = threadIdx.x / 32;
//   frag_a = ldmatrix(A_tile, warp);             // ldmatrix.sync
//   frag_b = ldmatrix(B_tile, warp);             // ldmatrix.sync
//   frag_c = mma.sync.aligned.m16n8k16(frag_a, frag_b, frag_c);  // TensorCore MMA
//   // 中间 tile [16×128] 留在 shared memory (SetConnect edge at shared memory level)

// === ComputeTile (BiasAdd, CUDA Core) ===
//   float bias_val = bias_tile[threadIdx.x % 128];
//   acc += bias_val;                             // FADD (CUDA Core)
//   // 同一 warp，MMA 完成后执行 FADD — 时间复用

// === ComputeTile (ReLU, CUDA Core) ===
//   acc = max(acc, 0.0f);                       // MAX.F32 (CUDA Core)

// === StoreTiles ===
//   output[blockIdx.x] = acc;                    // STG
```

**跨平台差异最大点**：
1. **GraphCore IPU**：无硬件 shared memory → LoadTiles 直接加载到本地 SRAM/tile-local memory；ComputeTile 在 MIMD 核心上以 tile 粒度独立执行，无 warp/SIMT 概念；SetConnect 的 "shared memory level connection" 退化为 "IPU SRAM level connection"。
2. **ROCm MI50**：与 CUDA 近似但 TensorCore 对应 MFMA (Matrix Fused Multiply-Add) 指令，`ldmatrix` 无直接等价 → WELDER 需用 buffer load + broadcast 指令模拟。
3. **Host memory 扩展**：在 CUDA 上通过 `cudaMemcpyAsync`（pinned memory + double buffer + CUDA stream）实现 LoadTiles 跨 host↔device，需额外 stream 同步逻辑。

---

### Round 5

**Q**：关于影响 WELDER 并发效率的架构限制与可缓解性，请补充：

1. **瓶颈根因的精确分解**：你说 memory bandwidth util 96.7% 是核心瓶颈——这 96.7% 是**全局 HBM 带宽利用率**还是包括 L2→SMEM 层级？多 fused kernel 并发执行时（Rammer multi-stream），多个 fused kernel 是否共享同一 HBM 带宽并导致**带宽竞争饱和**？如果多 kernel 吃满 HBM bandwidth，追加 concurrent kernel 后延迟是否线性增长？有无定量测量——例如 2/4/8 个 concurrent stream 时每个 kernel 的单独延迟 vs 总吞吐？

2. **Shared memory 容量作为硬限制的可缓解边界**：你说 cost model 对超容配置施加 infinite penalty 淘汰。但如果 tile-graph 的 footprint 刚好略超 capacity（例如 100KB > 96KB），WELDER 是否有**自动 partition** 的 fallback 路径（例如将 tile 切小后分两轮加载）？还是必须回退到 global memory 连接？如果有 partition 机制，引入的 extra traffic 比例如何？论文是否比较了"一次 shared memory fusion（fit within capacity）" vs "分两次加载的 partial shared memory fusion" 的性能差异？

3. **并发对访存模式的影响**：当 WELDER 融合了含 Matmul（coalesced 访存）和含 Gather（Attention 中 index 访存，例如 Gather from past KV cache）的算子时，同一 kernel 内两种访存模式如何共存？是否需要在 shared memory 发生 layout 转换（如从 row-major→column-major）？这种转换的 data movement 开销被量化过吗？

**A**：**Q1. 瓶颈根因的精确分解**

96.7% memory bandwidth utilization 来自 Figure 1 的 **global memory throughput**（即 HBM），使用 ONNXRuntime 在 V100 上 10 模型 batch=1 的总平均。论文**未进一步分解**为 HBM↔L2 vs L2↔SMEM 层级——这是片上 performance counter 级测量，通常不在编译论文中报告。但从 microarchitecture 推理：HBM 饱和（96.7%）意味着 L2 cache 的 miss rate 很高（典型 DNN 的 L2 hit rate 约 30-50%，因权重和激活规模远超 L2 容量），所以瓶颈主要在 HBM→L2 链路。

**Multi-stream 并发与 HBM 竞争**：WELDER 构建于 Rammer 之上利用 multi-stream 调度并发 kernel。论文**未测量**多 concurrent kernel 共享 HBM 带宽时的竞争量化数据——当 Rammer 水平发射多个独立 kernel 时，它们确实共享 HBM BW 上限（V100 ~900 GB/s）。若 BW 已饱和，追加 concurrent kernel 的理论预期是各 kernel 延迟按比例增加而总吞吐持平（反线性 scaling）。论文仅给出 WELDER vs Rammer 端到端 1.96×（batch=1），说明垂直融合减少的 DRAM traffic 在多 stream 环境下仍正交有效。**无 2/4/8 concurrent stream 的分辨率数据**——这不在论文实验范围内。

边界案例证据：大 batch（64）时加速比缩小（vs Rammer 从 1.96× 降至 2.1×？实际 batch=64 时 WELDER vs Rammer 2.1×——持平），说明垂直融合在大 batch 下收益不衰减，HBM 竞争力证非融合主要抑制因素。

**Q2. Shared memory 容量硬约束的可缓解边界**

当 tile-graph footprint 超过容量（如 100KB > V100 96KB），WELDER **不会**自动做 "partial fusion + 两轮加载" 的 fallback。cost model 对该配置施加 **infinite penalty 直接淘汰**（§3.2 Figure 7 line 16-17: `if MemFootprint(g) > level.capacity → continue`）。调度器转而搜索更小的 tile 配置（EnumerateSubtiles 从 size=1 开始向减 traffic 方向扩展），直到找到满足容量的配置或放弃该连接层。

唯一的自动 partition 机制是 **reduction tiling**（§3.1）：仅针对含 reduction 轴的 input tile，可顺序加载 subtile 并 accumulate 到 output tile。这面向 Matmul 的 K 维、Conv 的 C 维等——不是通用的 footprint 超标回退。

因此：footprint 略超 capacity 时不产生 extra shared→global traffic（该配置根本不被执行），而是**缩小 tile size** 使 footprint 满足容量。代价是更大的 num_tile_graphs（tensor_size / smaller_tile），从而增加 total traffic。例如 BERT Matmul+Softmax：[16×128] tile 的 traffic 264MB 是 WELDER 的最终选择——如果 [16×128] 超过容量，调度器会试 [8×128] → traffic 变为 528MB（×2），或者放弃 shared memory 连接回退到 global memory（unfused 840MB）。论文**未直接比较**这两种场景的性能差异。

**Q3. 并发对访存模式的影响——混合访存与 layout 转换**

WELDER 的 tile propagation 通过 **tensor expression 分析**处理非连续访问：

- **Gather/Index 访问**（如注意力中的 KV cache lookup）："our expression analysis provides a **conservative upper bound** as the input tile shape"（§3.1）。这意味着 tile propagation 对非规则访问不保证精确 shape，而是用上界覆盖——可能导致比连续访问更大的 footprint 或无法对齐 tile shape。
- **Inconsistent tile shape**：若两次 propagation 结果冲突（如 Gather 的输出 tile 形状无法传播到下游 Matmul 的输入），"we do not connect the latter output node to the current graph"——即**放弃融合**，回退到 global memory 连接。

**Layout 转换**：论文明确支持的 layout 转换是 **Transpose** 和 **2D→1D thread block 映射**（§4.2）：
- Transpose 的 blockIdx 重映射：`remapping relationship is deduced from their tensor expressions`
- 2D→1D 映射：当 TensorCore 需要 2D thread block（threadIdx.x + threadIdx.y）而 element-wise 算子仅 1D 时，WELDER 映射 2D→1D（总线程数相等即可）

但论文**未量化** layout 转换的 data movement 开销。对于更复杂的 Row-major ↔ Column-major 转换（如 fused kernel 内 Matmul 输出 row-major 但下游算子需要 column-major 输入），当前设计依赖 tensor expression 分析选择保守上界，缺乏跨格式的显式转换 cost 建模。这是 WELDER cost model 的一个已知简化——所有访问假设为 coalesced 基准，仅通过 uncoalesced penalty 修正 128B 粒度以上的非对齐访问。

---

### Round 6

**Q**：关于 WELDER 实验方法论的粒度、可信度与建模覆盖，请补充：

1. **延迟测量的粒度与方法**：你说端到端 latency 用 CUDA Event 测量（warmup + ≥5s 重复取平均）。这个测量能否捕捉到**多 stream 并发执行的时序重叠**——例如 Rammer 水平调度多个 fused kernel 时，CUDA Event 以 kernel launch 为粒度，能否分辨各 kernel 的实际执行起止时间和重叠比例？是否有用 NVIDIA Nsight 或 nvprof 做过更细粒度的时序验证（如 GPU trace 显示几个 fused kernel 的 overlapping execution 的比例）？

2. **Cost model 的精度验证**：WELDER 使用解析 cost model（MemFootprint+MemTraffic）搜索最优 tile 配置（仅 20 trials/subgraph）。这个 cost model 在多少种配置上做过**预测精度验证**——即预测 latency 与真实测量值的平均误差（MAPE）是多少？对于 cost model 未建模的因素（intra-kernel resource contention、warp scheduling 影响、cache hit rate 波动），累积误差在什么范围内？论文中是否有类似 Figure 9 的 correlation scatter plot 验证 cost model 预测准确性？

3. **消融实验的隔离性**：消融实验（WELDER-none→base→full）的 52% latency 减少中，**多少来自 kernel launch 减少的 fixed overhead，多少来自 DRAM traffic 减少的 actual computation/memory time**？是否有单独测量 kernel launch overhead 占 unfused 总时间比例的数据（例如 BERT batch=1 时 kernel launch 占 ~15%？）？

**A**：**Q1. 延迟测量的粒度与方法**

测量方法（§5.1）：CUDA Event 测量端到端 latency（GPU kernel 执行时间），warm-up 若干次后重复运行**至少 5 秒**取平均。论文报告 "very little variation in all cases"——即方差极小。几何平均计算跨模型平均加速比。

**Multi-stream 时序重叠的测量缺口**：WELDER 的端到端 latency 测量**无法分辨** multi-stream 并发执行的各 kernel 起止时间和重叠比例。CUDA Event 插在整体模型执行前后，只给出总耗时，不分解 kernel 级 timeline。论文**未使用** NVIDIA Nsight/nvprof 做细粒度 GPU trace 验证——不报告类似 "4 个 fused kernel 在 3 个 stream 上重叠执行的比例" 的数据。从方法论文献标准看，这是合理的：WELDER 不声称贡献在 multi-stream 调度（该部分继承自 Rammer），因此不重复做 kernel 级时序分解。Rammer 原论文（OSDI'20）有 multi-stream 重叠的 GPU trace 验证，WELDER 引用该结果。

可间接推断：WELDER vs Rammer 端到端加速 1.96×（batch=1）——垂直融合减少了 kernel 数量 60-67%，因此即便 multi-stream 存在重叠，融合后仍需发射的 kernel 数更少，stream 竞争天然降低。论文未进一步定量分离这个相互作用。

**Q2. Cost model 的精度验证**

WELDER 使用**解析 cost model + profiling 双层筛选**，而非纯预测：

- **搜索阶段**：解析 cost model（MemFootprint + MemTraffic）作为优先队列排序键，快速选择 top-K 最少 traffic 配置（K 默认值论文未明确给出，但总 trials 为 20 per subgraph-like）。这是**排序/筛选**而非预测 latency。
- **最终选择**：top-K 配置在硬件上 profiling（`d.Profile(configs)` in Figure 7 line 8），取 min latency 为最终选择。因此 cost model 误差**不影响最终性能**——只影响搜索效率（如果 cost model 排序差，好配置可能被排除在 top-K 外）。

论文**未报告**：
- Cost model 预测 latency 与实测值的 MAPE
- Correlation scatter plot（如 predicted traffic vs measured latency）
- Cost model 在不同 tile 配置下的排序准确率（好配置被保留在 top-K 中的比例）

未建模因素的累积误差：cost model 未显式建模的因素包括 intra-kernel resource contention（register pressure、warp scheduling 影响）、cache hit rate 波动、bank conflict 实际损失。这些误差通过 "20 trials/subgraph → profile top-K" 的设计被吸收——仅需排序大致正确，差值由硬件 profiling 兜底。效率证据：WELDER 编译 BERT 244s（651 trials）vs Ansor 15,285s（8000 trials）——62× 加速说明排序精度足够使候选空间大幅缩减。

**Q3. 消融实验的隔离性**

消融实验的指标总表（V100 FP32 batch=64，3 模型几何平均）：

| 指标 | WELDER-none → base（add register connection） | base → full（add shared memory connection） |
|------|:---:|:---:|
| Latency | -52% (2.08×) | -29% (up to 1.82×) |
| Kernel launches | -67% | -60% |
| Global mem transactions | -52% | -25% |
| IRS | -66% | -65% |

**论文未单独分解** kernel launch overhead vs actual compute/memory time 在总 latency 减少中各占多少。但可以间接推断：

- **Register 连接（none→base）**：-67% kernel 数 与 -52% latency 对应——kernel launch overhead 通常在 unfused 总时间中占 ~5-15%（batch=1 时更高，因每个 kernel 计算量小），所以 -67% launch 减少不足以解释 -52% latency 减少——意味着 register connection 也通过消除 intermediate DRAM 往返（-52% transactions）贡献了实质性 compute/memory 加速。
- **Shared memory 连接（base→full）**：-60% kernel 数 与 -29% latency 对应——launch 消除的边际收益递减（残余 kernel 数量已少），但 -25% transactions 和 -65% IRS 证明 shared memory 复用消除了大量 DRAM traffic。注意 "transactions 降幅（25%）< IRS 降幅（65%）" 论文自解释为：融合无法优化 weight memory access（如 Matmul 的 weight 仍要从 DRAM 加载），因此 traffic 降幅被 weight 访问稀释。

绝对数值示例（Matmul+Softmax）：unfused 840MB global traffic（24576 tiles × 35KB/tile）→ fused 264MB（6144 tiles × 44KB/tile），节省 576MB（69%）。fused kernel 0.29ms vs separate 0.36ms，其中 launch overhead 差异可估算：2 kernel launches (Matmul + Softmax) vs 1 fused launch，~5-10µs per launch，贡献约 10-20µs 差异，占总 70µs 差值的 14-28%——其余来自 DRAM traffic 减少。

---

