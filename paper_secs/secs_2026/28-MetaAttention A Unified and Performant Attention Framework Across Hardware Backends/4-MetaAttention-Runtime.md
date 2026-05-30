# 4 MetaAttention Runtime

To retain high performance across various hardware backends and various attention defined using our interface, MetaAttention employs a structured approach to optimize execution performance. First, it utilizes IntermediateTensor components with configurable attributes to model the scheduling space. Concurrently, DeviceConfig components abstract hardware configurations into a unified representation, which constrains the scheduling space. The scheduling policy (Section [4.2\)](#page-7-0) then selects a scheduling plan from this space, which is subsequently dispatched to the attention runtime for kernel execution.

#### 4.1 Scheduling Space

IntermediateTensor. MetaAttention defines IntermediateTensor to represent all transient tensors in device memory during attention computation.

As attention mechanisms fuse multiple operators, they generate numerous intermediate tensors whose placement critically impacts on-chip memory utilization and computational latency. By focusing on intermediate tensors, MetaAttention can systematically deduce the tiling, memory allocation, and pipeline requirements for attention mechanisms.

Key attributes of IntermediateTensor include:

• Tensor tile shape (tile): By dividing tensors into smaller tiles, we can perform operations tile-by-tile and allocate buffers efficiently. Using the computation graph, we propagate the tiling scheme across all operations to infer the tile shapes of , , and other tensors, ensuring an optimal balance between computation and memory.

- Tensor location (mem): Intermediate tensors can be stored in various levels of memory, such as global memory, shared memory, or registers. Each location offers a trade-off between latency, bandwidth, and resource availability.
- Pipeline stage (pipelineStage): Operations involving intermediate tensors are divided into multiple pipeline stages, such as memory copy and computation. The number of stages determines the buffer requirements and scheduling flexibility, enabling overlapping operations to maximize throughput and minimize resource contention.

This component ensures that all elements of the attention mechanism, including inputs, outputs, and intermediate results, are unified under a consistent scheduling strategy.

Take the Parallel Pattern in Fig. [4](#page-5-0) as an example. Intermediate Tensor includes Q, K, scores, V, state, output and the tensor created in customizable\_function. The attributes of these intermediate tensors significantly impact the performance, such as the tile size of Q, K and V influence the balance between the parallelism of computation and the on-chip memory consumption.

DeviceConfig. The DeviceConfig component provides hardware-specific constraints that refine the scheduling space defined by intermediate tensors. It encapsulates attributes such as:

- Base tile shape (basetile): Specifies the optimal tile shape for computations on the target hardware, ensuring alignment with hardware-specific constraints, such as alignment with matrix multiplication computing instruction and memory transaction.
- Memory hierarchy (memoryInfo): Provides details about the available memory tiers (e.g., registers, shared memory, global memory) and their respective capacities, enabling efficient allocation and minimizing contention.

DeviceConfig determines the feasible tiling and memory strategies during scheduling. For instance, the base tile shape ensures hardware-aligned tiling configurations, while memory capacity constraints prevent resource overcommitment. Attention runtime. The attention runtime executes scheduling plans efficiently across heterogeneous hardware backends. Attention runtime takes the scheduling plans as input and instantiate the plans into attention computation kernels tailored to hardware.

To ensure efficiency, attention runtime integrates universal optimization tailored to attention mechanisms. For parallel patterns, the runtime implements online techniques for efficient row-wise normalization, drawing inspiration from FlashAttention's approach [\[9\]](#page-11-0). For recurrent patterns, it employs chunk parallelism techniques [\[32\]](#page-12-13) to maximize hardware utilization and throughput.

Additionally, attention runtime supports multiple implementations tailored to different backends, such as those implemented in TileLang [\[29\]](#page-12-14) and CUTE [\[7\]](#page-11-15). Leveraging a unified scheduling for different backends, MetaAttention can dynamically support all attention across different hardware and different configurations.

#### <span id="page-7-0"></span>4.2 Scheduling Policy

The scheduling plan contains the attributes of IntermediateTensors, including tile size, memory location, and pipeline stages. The scheduling policy generates the execution plan by determining the attributes of all intermediate tensors in attention.

To find an optimized execution plan within the scheduling space, multiple constraints must be considered. First, the combination of fixed computations and customizable functions in attention forms a tile computation graph, which requires adjacent IntermediateTensor instances to share the same tile size. Additionally, different attention patterns and input tensor shapes demand distinct memory placement strategies and pipeline stages to balance latency and on-chip resource utilization.

As illustrated in Fig. [10,](#page-7-1) MetaAttention employs a twolayer scheduling policy to generate the optimal scheduling plans represented by IntermediateTensors. This policy operates at two levels: tile config scheduling and tile resource scheduling. The outer layer, tile config scheduling, explores tile size attributes of intermediate tensors. The inner layer, tile resource scheduling, determines the memory location and pipeline stage attributes for intermediate tensors.

Tile config scheduling. This layer takes as input the attention computation graph (Graph) composed of IntermediateTensor objects and hardware configuration details (DeviceConfig). It enumerates all possible tile sizes for the output tensor and propagates these tile sizes through all intermediate tensors, thereby generating a set of tile graphs (lines [2](#page-7-2)[–3\)](#page-7-3).

For each tile configuration (line [5](#page-7-4) - [6\)](#page-7-5), the policy generates a set of execution plans using the tile resource scheduling layer and evaluates their performance through profiling (line [7](#page-7-6) [-8\)](#page-7-7).

Tile resource scheduling. This layer optimizes the memory location and pipeline stage attributes of intermediate tensors for a given tile configuration. The process begins by initializing all intermediate tensors to the highest available memory tier (e.g., registers) to minimize memory I/O overhead (line [15\)](#page-7-8). The policy then generates candidate plans by enumerating unconfigured attributes—such as pipeline stages—and checks their feasibility against hardware constraints (lines [18–](#page-7-9)[20\)](#page-7-10). If no valid plan is found, the policy iteratively demotes tensors to lower memory tiers and reattempts plan generation (line [24\)](#page-7-11).

```
1 Func TileConfigScheduling(g: Graph, D:DeviceConfig)
2 tiles = EnumerateTiles(g.output_shape, D.basetile)
3 tensor_tile_graphs = PropagateTileGraphs(g, tiles)
4 plans = []
5 for tile_graph in tensor_tile_graphs do
6 plans += TileResourceScheduling(tile_graph,
          D);
7 for plan in plans do
8 if Profile(plan) < best_latency
 9 best_latency = Profile(plan);
10 best_plan = plan;
11 return best_plan;
12 Func TileResourceScheduling(g: TileGraph,
   D:DeviceConfig)
13 tensor_list = GetIntermediateTensors(g);
14 SetTile(tensor_list, g.tiles);
15 SetMem(tensor_list, "L0");
16 tensor_list_sorted = tensor_list.sort(key=lambda t
       :(len(g[t].use_list), size(t.tile)));
17 for tensor_i in tensor_list_sorted do
18 plans =
          EnumerateUnsetAttributes(tensor_list);
19 for plan in plans do
20 if not MeetMemoryConstraint(plan,
             D.memoryInfo)
21 plans.remove(plan);
22 if not plans.isEmpty()
23 return plans;
24 LowerMemLocation(tensor_i.mem)
25 return EmptySet();
```

<span id="page-7-11"></span><span id="page-7-10"></span><span id="page-7-9"></span><span id="page-7-8"></span>Fig. 10. Scheduling algorithm. We employ a two-layer scheduling strategy to generate the execution plan. The first layer, tile configuration scheduling, explores the tile size attributes of intermediate tensors. The second layer, tile resource scheduling, determines the memory allocation and pipeline stage assignments for intermediate tensors.

