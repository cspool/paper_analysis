## CUDA Graph

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CUDA Graph 是 NVIDIA CUDA 10 引入的机制，允许将一系列 GPU kernel launch、memory copy 和 memory allocation 操作预录制为一个有向无环图（DAG），后续通过单次 `cudaGraphLaunch` API 调用回放整个图，消除逐个 kernel launch 时产生的 CPU-GPU 同步开销和 CUDA driver 调度开销。

CUDA Graph 的生命周期分为三个阶段：
1. **Graph Construction（图构建）**：可以使用两种方式构建图。Stream Capture 方式是通过 `cudaStreamBeginCapture`/`cudaStreamEndCapture` 包裹目标 stream 上的一组 GPU operations，CUDA runtime 自动记录这些操作及其依赖关系为图节点。Explicit API 方式是手动调用 `cudaGraphAddKernelNode`/`cudaGraphAddMemcpyNode` 等函数显式添加节点和依赖边。
2. **Graph Instantiation（图实例化）**：通过 `cudaGraphInstantiate(exec, graph)` 将图编译为可执行对象（`cudaGraphExec_t`）。实例化过程会进行静态验证、内存预分配和优化，为后续快速回放做准备。实例化是一次性开销。
3. **Graph Launch（图启动）**：通过 `cudaGraphLaunch(exec, stream)` 将整个图提交到指定 CUDA stream。与逐个 kernel launch 不同，单次 graph launch 仅产生一次 CPU-GPU 同步，图中所有 kernel 由 GPU 自主调度执行。

此外 CUDA Graph 支持更新（Update）：`cudaGraphExecKernelNodeSetParams` 允许修改已实例化图中 kernel 节点的参数（如 tensor pointers），无需重新实例化。这在 serving 场景中特别有用，因为每轮 iteration 的 KV cache 地址变化，但 kernel shapes 不变。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Nimble 中 CUDA Graph 的 AoT (Ahead-of-Time) scheduling 流程：

```
// ===== Phase 1: AoT Preparation (一次) =====
// Step 1: TorchScript tracing
model_ts = torch.jit.trace(model, dummy_input)
// → 生成包含所有算子的 static computation graph

// Step 2: Stream Assignment (Graph Rewriter)
dag = build_dag(model_ts)                   // 从 TorchScript graph 构建 DAG
meg = compute_minimum_equivalent_graph(dag)  // 计算最小等价图
bipartite = construct_bipartite(meg)         // 构建二分图
matching = ford_fulkerson_max_matching(bipartite)  // 寻找最大匹配
stream_assignment = assign_from_matching(matching) // 算子 → stream 分配
// 输出: 每个算子被分配到特定 CUDA stream (stream_0, stream_1, ...)

// Step 3: CUDA Graph Capture
for each stream in streams:
    cudaStreamBeginCapture(stream)           // 开始捕获该 stream
    // 执行分配到此 stream 的所有算子（使用 dummy input）
    for op in stream_assignment[stream]:
        op.forward(op_input)                 // PyTorch operator → CUDA kernel launch
    cudaStreamEndCapture(stream, &graph)     // 结束捕获，生成 CUDA GraphNode
// 同时记录跨 stream 的 CUDA event 同步点

// Step 4: Graph Instantiation
cudaGraphInstantiate(&exec, graph)           // 编译 CUDA Graph → 可执行对象
// 内存预分配: 所有中间 tensor 的 GPU memory 在此阶段分配
// AoT preparation 耗时: mean 0.35s, max 1.07s (NASNet-A large)

// ===== Phase 2: Runtime Inference (每次新输入) =====
// 与 PyTorch baseline 不同，完全绕过 Python/C++ framework runtime:
// ✗ 无 operator dispatch (Python/C++ autograd 查找)
// ✗ 无 output shape inference (meta-data computation on CPU)
// ✗ 无 GPU kernel selection (cuDNN implementation choosing)
// ✗ 无 kernel argument preparation (grid/block dims, strides)
// ✗ 无 per-operator memory allocation (cudaMalloc/cudaFree)

// 仅更新 input/output buffer pointers 后单次 launch:
cudaMemcpy(graph_input_buffer, new_input_data, ...)
cudaGraphLaunch(exec, stream_main)           // 单次 launch + 单次 CPU-GPU sync
// GPU 自主执行所有预录制 kernel，多个 stream 并行
```

Nimble 的关键设计决策：整个模型在一次 CUDA Graph 中录制完成（非 per-layer 或 per-iteration 录制），因为 static DL model 的 DAG 形状在输入变化时不改变。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUDA Graph 通过 CUDA Runtime API 实现，核心函数为：
- `cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode)` — 开始捕获
- `cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph)` — 结束捕获并生成图
- `cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, ...)` — 实例化
- `cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)` — 启动回放
- `cudaGraphExecKernelNodeSetParams(...)` — 更新 kernel 节点参数

在 LLM serving 场景中广泛应用（vLLM, SGLang, TensorRT 等）：
- Decode phase：固定 batch size + 固定 sequence length 的 decode step → CUDA Graph 预录制 → 消除数百次小 kernel 的 launch overhead
- 限制：(a) 图内所有 kernel 的 grid/block dims、shared memory 大小必须固定（静态 shapes）；(b) 不支持 dynamic control flow（conditional kernels）；(c) 内存地址变化时需通过 Update API 更新节点参数；(d) 多 shape 场景需预录制多个 graph instances，增加 GPU memory 开销
- Nimble 的特殊用法：AoT scheduling 中使用 CUDA Graph 不仅消除 launch overhead，更关键的是**完全绕过 PyTorch framework runtime**——因为 CUDA Graph 已包含所有 kernel 的完整执行拓扑，GPU 可脱离 CPU framework 自主执行

CUDA Graph 的限制（Nimble 论文指出）：不支持 dynamic neural network models（有 data-dependent control flow）；每个图 instance 需额外 GPU memory 存储 meta-data

涉及论文标题：
- Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning

---
