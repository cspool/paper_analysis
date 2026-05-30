## Welder Scheduling Deep Learning Memory Access via Tile-graph

- 属于编译框架的实现是什么？实验比较什么？
  实现是WELDER，一个基于tile-graph抽象的DNN编译器，核心机制包括：(1) Tile-graph——将DNN计算建模为tile级数据流图，每个节点处理一个数据tile，通过SetConnect接口设置每个edge的数据复存memory level，通过Propagate接口由输出tile shape链式推断整个tile-graph的tile配置；(2) 两阶段调度算法——Graph Connecting Scheduler枚举每条边的连接层（register/shared memory/global memory），Sub-Graph Scheduler通过解析cost model（MemFootprint + MemTraffic）搜索最优tile配置，利用inter-layer independence和intra-layer independence解耦优化空间；(3) 分层tile-graph编译——将优化后的执行计划映射到四条硬件抽象接口（Allocate、LoadTiles、ComputeTile、StoreTiles），递归展开为全模型计算程序。

  实验比较了WELDER与PyTorch (v1.12)、ONNXRuntime (v1.12)、Ansor (v0.9)、Rammer、TensorRT (v8.4)、FasterTransformer (v5.2)、BladeDISC (v0.3.0，含AStitch)、Nimble。在10个SOTA DNN模型上评估（MobileNet、BERT、ViT、Conformer、MobileViT、Swin-Transformer、NeRF、NAFNet、Restormer、BSRN）。在NVIDIA V100上FP32 batch=1：vs PyTorch 4.29×、vs ONNXRuntime 2.07×、vs Ansor 1.44×、vs TensorRT 1.47× 几何平均加速。在TensorCore FP16 batch=1：vs PyTorch 7.18×、vs ONNXRuntime 3.08×、vs TensorRT 1.53×。在RTX-3090上vs TensorRT 1.40× 平均加速。在AMD MI50上vs PyTorch 2.62×、vs Ansor 1.53×。自动发现89种非常规fusion pattern（含两个以上reduction operator），最大fuse 48个算子为单kernel。

- 硬件平台是什么，配置是什么。
  (1) Azure NC24s_v3 VM，Intel Xeon E5-2690v4 CPU，NVIDIA Tesla V100 (16GB) GPU，Ubuntu 16.04，CUDA 11.0。(2) 本地工作站，Intel Xeon E5-2678 v3 CPU，NVIDIA GeForce RTX 3090 GPU，Ubuntu 18.04，CUDA 11.3。(3) AMD GPU server，Intel Xeon E5-2640 v4 CPU，AMD Radeon Instinct MI50 (16GB) GPU，Ubuntu 18.04，ROCm 5.2.3。(4) Azure ND40s_v3 VM，Intel Xeon Platinum 8168 CPU，16 IPU，Poplar-sdk 3.0。

- 开源编译框架是什么。修改了什么。
  WELDER基于开源DNN编译器TVM、Roller和Rammer构建。核心修改：新增tile-graph抽象、tile propagation、两层调度算法、分层tile-graph执行计划code generation共5.2k行代码。利用TVM编写kernel schedule、Roller枚举高效tile配置、Rammer做端到端图优化。输入ONNX graph，执行常量折叠和简单element-wise fusion后转换为tile-graph进行holistic memory scheduling。实现于CUDA和ROCm GPU以及GraphCore IPU。对于CUDA/ROCm GPU，调度三层memory hierarchy：global memory (DRAM)、shared memory、register。还扩展了host memory层以处理大规模输入。

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  开源地址：https://github.com/microsoft/nnfusion/tree/osdi2023welder（artifact branch）。Artifact包含完整源码、Docker环境配置、预编译logs和模型。

  作用：WELDER将DNN内存访问优化统一为tile-graph调度问题，自动发现最优的inter-operator和intra-operator数据复用模式——在给定内存层级约束下，通过SetConnect/Propagate接口和两层调度算法递归决定每个算子的最优tile配置和edge连接层。

  全过程（以Conv + ReLU + MaxPool在V100三层memory hierarchy (L2→L1→L0) 上编译为例）：
  ```
  输入：ONNX model graph (Conv → ReLU → MaxPool)
    Conv: input[1×H×W×C], weight[3×3×C×F], output[1×H'×W'×F]
    ReLU: element-wise op, same shape
    MaxPool: kernel[2×2], output[1×H''×W''×F]

  Step 1 — Graph Preprocessing:
    常量折叠、简单element-wise fusion
    转换为初始tile-graph（各operator分解为operator-tile）

  Step 2 — Graph Connecting (外层枚举):
    按拓扑序遍历所有node的output edge:
      for level in [Register, SharedMem, GlobalMem]:
        SetConnect(edge, level)  // 设置data reuse目标层
        subgraph = ExtractSubgraph(node, level=0)  // 提取连接层级>0的连通子图
        configs = SubGraphTiling(subgraph, level=0, tensor_shapes)
        latency = min(Profile(configs))
        if latency < best: best_level = level
      SetConnect(edge, best_level)  // 选择最优连接层

    例：Connect(Conv→ReLU) at L0 (register/shared memory),
         Connect(ReLU→MaxPool) at L1 (shared memory)

  Step 3 — Sub-Graph Tiling (内层优化):
    For 每个sub-graph at current level:
      EnumerateSubtiles: 从size=1开始扩展tile shape（类似Roller）
        → 朝减少traffic方向 + 对齐硬件feature的方向搜索
      For each output tile config:
        Propagate(g, config): 链式shape inference推断所有tile size
        MemFootprint(g) > level.capacity → skip
        configs.push(priority=MemTraffic(g))
      TopK(configs): 选择最少traffic的K个配置
      For each node in subgraph:
        upper_subgraph = ExtractSubgraph(node, level+1)
        SubGraphTiling(upper_subgraph, level+1, config)  // 递归上层

  Step 4 — Hardware-Aligned Tile Search:
    对枚举的tile配置施加penalty:
    - uncoalesced memory access: 按128B transaction计算额外traffic
    - insufficient parallelism: 按core utilization比例增加traffic
    - footprint > capacity: infinite penalty直接淘汰
    TensorCore约束: tile轴尺寸为MMA fragment size的整数倍

  Step 5 — Code Generation:
    递归展开分层tile-graph生成device code:
    ┌─ Level 0 (Global Memory): ────────────────────────────┐
    │ Allocate(workspace, L0)                                 │
    │ LoadTiles(input_tiles from DRAM → shared memory)       │
    │                                                          │
    │ ┌─ Level 1 (Shared Memory): ──────────────────────┐   │
    │ │ Allocate(workspace, L1)                           │   │
    │ │ // Conv+ReLU tile-graph at shared memory level    │   │
    │ │ for each node n in tile-graph:                    │   │
    │ │   ┌─ Level 2 (Register): ──────────────────┐    │   │
    │ │   │ ComputeTile(Conv operator-tile)          │    │   │
    │ │   │ ComputeTile(ReLU operator-tile)          │    │   │
    │ │   │ // register-level tile connection         │    │   │
    │ │   └───────────────────────────────────────┘    │   │
    │ │ StoreTiles(ReLU output → L0 shared mem buffer) │   │
    │ └──────────────────────────────────────────────┘   │
    │                                                          │
    │ ┌─ Level 1 (shared memory, next sub-graph): ───────┐   │
    │ │ // Conv+ReLU+MaxPool tile-graph                    │   │
    │ │ for each node:                                     │   │
    │ │   ┌─ Level 2: ────────────────────────────┐      │   │
    │ │   │ ComputeTile(MaxPool operator-tile)      │      │   │
    │ │   └───────────────────────────────────────┘      │   │
    │ │ StoreTiles(MaxPool output → DRAM)                 │   │
    │ └──────────────────────────────────────────────┘   │   │
    │ StoreTiles(output_tiles → DRAM)                      │   │
    └──────────────────────────────────────────────────────┘   │

  Step 6 — Kernel-level Code Transformations:
    - Load/Store Rewriting: global→shared memory访问改写 (TIR pass)
    - Block/threadIdx Remapping: Transpose等算子需要blockIdx映射;
      2D→1D thread block mapping
    - Memory Management: buffer liveness分析 + bestfit offset计算
    - Padding: 避免bank conflict
    - Memory Fences: 防止race condition

  输出：单个fused CUDA kernel（或少量kernel），中间数据tile在shared memory中复用。

  编译加速：
    - 多进程并行编译评估各配置
    - sub-graph签名缓存（如BERT 12层复用第1层结果）
    - WELDER编译time 244s (BERT) vs Ansor 15285s
  ```
