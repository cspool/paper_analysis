## Welder Scheduling Deep Learning Memory Access via Tile-graph

- baseline方法是什么？
  Baseline是现有的DNN编译器和框架，它们将DNN视为计算密集型工作负载，采用计算中心化的优化策略：

  **PyTorch [10]**（eager execution）：每个operator独立执行，Python/C++ dispatch → shape inference → kernel selection → argument preparation → kernel launch，每个operator的中间结果通过global memory传递。小模型batch=1时Python overhead主导，大batch时依赖cuBLAS/cuDNN library kernel。

  **ONNXRuntime [8]**（graph optimization）：移除Python overhead，实现pattern-based graph optimizations（如operator fusion规则），但融合仅限于预定义模式（如Conv+ReLU）。

  **Ansor [50]**（search-based compiler）：通过ML-guided search生成高性能tensor program，支持register-level element-wise fusion（如Matmul+BiasAdd、Conv2D+ReLU），但无法exploit shared-memory-level inter-operator data reuse，也无法fuse两个reduction-based operator（如Matmul+Softmax）。

  **TensorRT [7]**（vendor-specific inference library）：NVIDIA手工优化的kernel库，含expert-designed fusion rules和in-house kernels。对popular models（如BERT、Swin-T）有专门优化，但对新模型（如NAFNet）无覆盖，依赖通用kernel。

  **Rammer [31]**（horizontal fusion）：将独立并行kernel通过multi-stream调度并发执行（horizontal fusion），但不支持通过shared memory复用中间数据的dependent kernel fusion（vertical fusion）。

  **BladeDISC/AStitch [51]**（rule-based shared memory fusion）：通过预定义fusion rules对特定operator组合做shared memory fusion，但遇到不支持的operator则fallback到PyTorch runtime。

  全栈执行例子（以Ansor在V100上执行BERT attention block：Matmul Q*K → Softmax → Matmul P*V为例）：
  - 算法层：标准Transformer attention——Q,K,V projection (Matmul) → Q*K^T (Matmul) → Softmax → P*V (Matmul) → O projection (Matmul)
  - 系统框架层：PyTorch trace → ONNX export → Ansor编译。Ansor对每个operator独立tune（800 trial per operator），生成optimized kernel。
  - 编译框架层：Ansor search-based compiler——program sampling → ML cost model training → top-k performance evaluation → final kernel selection。Register-level fusion rules：Matmul+BiasAdd、Conv2D+ReLU等element-wise融合。无法fuse Matmul+Softmax（两者都是reduction-based operator，tile shape冲突）。
  - Kernel调度层：每个operator独立kernel launch。Matmul Q*K^T kernel：从DRAM加载Q tile和K tile → shared memory → TensorCore MMA → 输出C [seq_len×seq_len] 写入DRAM。Softmax kernel：从DRAM读C → 执行softmax → 写回DRAM。Matmul P*V kernel：从DRAM读P和V → TensorCore MMA → 写入DRAM。中间tensor C和P在DRAM中完整物化，global memory traffic高。
  - 硬件架构层：NVIDIA V100 GPU (16GB, SIMT Core + TensorCore)。Global memory bandwidth 900 GB/s。Ansor kernel执行：Matmul Q*K^T 占用TensorCore但shared memory仅用于intra-operator tiling，无法与Softmax复用。Softmax受限于memory bandwidth（大中间结果从DRAM读取）。

  Baseline核心缺陷：
  1. **计算中心化思维**：将DNN视为compute-intensive，实际上现代DNN（ViT、Conformer、NeRF、NAFNet等）memory bandwidth utilization高达96.7%而compute utilization仅51.6%。基线优化器仍聚焦于加速计算而非减少内存访问。
  2. **缺乏跨算子内存复用**：operator间的中间tensor必须完整物化在global memory中，造成大量inter-operator DRAM traffic。Ansor/TVM的fusion仅覆盖register-level的element-wise算子对（如Conv+ReLU），无法处理含reduction的fusion（如Matmul+Softmax）。
  3. **融合规则vs通用性trade-off**：TensorRT/BladeDISC的expert-designed fusion rules对不支持的算子组合完全失效。缺乏一套通用的、以内存为中心的优化框架。
  4. **tile shape冲突未解决**：不同算子的最优tile shape不同（如Matmul的[32×64] vs Softmax的[4×128]），独立优化时无法在shared memory中复用中间数据。现有方法要么放弃shared memory复用，要么依赖有限的手工规则。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  WELDER提出以**内存访问优化为核心**的DNN编译框架，核心创新是**tile-graph抽象**——将DNN计算从operator级dataflow graph下沉到tile级dataflow graph，从而暴露operator内部tile间的细粒度数据复用机会。

  **方法一：Tile-Graph + Tile Propagation** → 解决"tile shape冲突"
  - 将DNN计算建模为operator-tile组成的数据流图。每个operator-tile处理一个输出数据tile。
  - 通过SetConnect接口在两个相邻operator-tile之间建立tile级连接（指定data reuse所在的memory level），通过Propagate接口从output tile shape链式推断整个tile-graph的所有tile配置。
  - 关键insight：给定operator的tensor expression（准确的输入→输出映射），可以从output tile shape反向推导其依赖的input tile shape。因此，只要两个相邻operator共享一个连接的output/input tile shape，它们的tile配置自动对齐——这是Matmul+Softmax得以在shared memory中复用的数学基础。

  **方法二：Inter-Layer Independence + Traffic Cost Model** → 解决"优化空间爆炸"
  - 发现total memory traffic仅由当前memory layer的output tile配置决定（inter-layer independence）。这允许将整个DNN的多层memory hierarchy优化空间解耦为多个独立子空间。
  - Traffic cost model：Σ(input_tile_sizes + output_tile_size) × num_tile_graphs。基于tile size和tensor shape解析计算，无需实际执行。
  - 两层调度算法：外层Graph Connecting枚举每条edge的连接层（L0/L1/L2），内层Sub-Graph Tiling基于traffic cost model搜索最优tile配置。MemFootprint检查容量约束，MemTraffic作为排序键。

  **方法三：Hierarchical Tile-Graph + 四条硬件抽象** → 解决"通用性"
  - 将优化后的执行计划递归编译为分层tile-graph——从最低memory level开始，递归分裂为上层sub-graph。
  - 仅需四条硬件接口（Allocate/LoadTiles/ComputeTile/StoreTiles）即可映射到任意有层级memory的加速器（CUDA GPU、ROCm GPU、GraphCore IPU）。
  - 无需per-operator-type fusion rules——所有operator只要能用tensor expression描述即可参与tile-graph优化。

  全栈执行例子（以WELDER在V100上执行同一BERT attention block：Matmul Q*K → Softmax fusion为例）：
  - 算法层：同baseline。WELDER不改变模型算法，仅改变数据在memory hierarchy中的流动方式。
  - 系统框架层：PyTorch trace → ONNX export → WELDER tile-graph compiler。编译过程：常量折叠 → operator-tile decomposition → SetConnect for each edge（枚举L0/L1/L2）→ Propagate tile shapes → Traffic cost evaluation → Profile best configs → Code Generation。
  - 编译框架层（核心差异）：WELDER替代了Ansor的per-operator tiling + rule-based fusion + ML cost model。Graph Connecting决定Matmul→Softmax edge在shared memory (L1)连接。Propagate从output tile [BM×BN] 反向推断Matmul的input tile [BM×BK] 和 [BK×BN]。Traffic cost model计算不同tile shape的global memory traffic，自动选择最优配置（如 [16×128] vs [4×128] vs [32×64]）。Hardware-aligned penalty过滤uncoalesced/inadequate parallel/over-capacity配置。最终选择 [16×128] output tile → 264MB total traffic（vs unfused 840MB）。
  - Kernel调度层（核心差异）：替代Ansor的两次独立kernel launch + DRAM中间物化。WELDER fused kernel：LoadTiles从DRAM加载Q tile [BM×BK] 和K tile [BK×BN] → shared memory buffer 0,1 → ComputeTile: Matmul operator-tile via TensorCore MMA (warp-level) → 中间结果 C_tile [BM×BN] 直接留在shared memory ← SetConnect at L1 → ComputeTile: Softmax operator-tile从shared memory读取C_tile → 执行softmax (SIMT) → StoreTiles将结果D_tile写回DRAM。循环覆盖全部tiles。消除了一次global memory write (C) + 一次global memory read (C) 的DRAM往返。
  - 硬件架构层：NVIDIA V100 GPU。从"Matmul TensorCore→DRAM→Softmax SIMT"的两段执行变为"Matmul TensorCore→Shared Memory→Softmax SIMT"的单kernel流水线。Global memory traffic 840MB→264MB (saving 69%)。Kernel launch count减半（单kernel fused vs 两个独立kernel）。1.26× speedup on Matmul-Softmax pair。

  扩展到更大范围：
  - BERT attention：Q*K fused with Softmax（seq_len=128时）。当seq_len=512（Conformer）时auto decision不fuse。
  - NeRF 7-layer MLP：full auto-fusion to single GPU kernel（前6层TensorCore + 输出层SIMT Core），全部中间结果存shared memory，5× speedup。
  - NAFNet：back-to-back pointwise convolutions fused with normalization。Auto decide fusion order（top layers: DWConv+PWConv cache feature map in shared memory；bottom layers: PWConv+DWConv cache full channel）。
  - 89种非常规fusion pattern自动发现：含Dual Matmul + Relu chain (13 ops)、48-operator fusion chain（DepthwiseConv+Broadcast+Divide+Erf+Multiply+Convolution × multiple cycles + Concat）。

  设计思路核心：WELDER将DNN内存优化问题从"为每对算子设计专门的fusion规则"转变为"在tile-graph上搜索最优tile连接配置"的单一通用优化问题。三个关键insight支撑这一转变：(1) tile propagation——给定tensor expression，output tile shape可以唯一确定所有input tile shape，自动对齐operators之间的tile配置；(2) traffic cost model——给定aligned tile config，traffic可通过解析计算（无需实际执行），且traffic仅依赖当前layer的tile config（inter-layer independence）；(3) 分层编译——通过SetConnect/Propagate两条原语和两条cost接口(MemFootprint/MemTraffic)，整个优化可递归分解为各memory layer的独立子问题。这本质上将operator fusion从"rule engineering problem"消解为"graph optimization problem"。
