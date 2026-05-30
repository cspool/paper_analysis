## Ahead-of-Time (AoT) GPU Scheduling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ahead-of-Time (AoT) GPU Scheduling（提前GPU调度）是 Nimble 提出的核心技术，指在模型实际推理执行之前完成全部 GPU task scheduling 流程——包括 operator dispatch、kernel selection、output shape inference、memory allocation 和 kernel argument preparation——将整个模型的 GPU 计算录制为可直接回放的执行图。运行时仅需将新输入数据拷贝到 GPU、更新 buffer pointers，然后通过单次 CUDA Graph launch 回放预录制的执行图。

与 JIT (Just-in-Time) compilation 对比：JIT 在运行时编译和优化，AoT 在首次执行前完成全部 scheduling。与 PyTorch eager mode 对比：eager mode 每轮 iteration 都重复完整的 scheduling pipeline（dispatch → shape inference → kernel selection → argument preparation → launch），AoT 将其全部前端化并一次性完成。

核心理念来源于 CUDA Stream Capture API 和 CUDA Graph：利用 "record-then-replay" 能力，将 GPU 执行与 CPU framework runtime 完全解耦。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// ================ AoT Scheduling 流程 ================

// Input: PyTorch model (e.g., NASNet-A mobile)
// Output: Executable CUDA Graph (含多 stream 分配)

// Step 1: DAG Extraction
dummy_input = torch.randn(batch_size, channels, height, width)
traced_graph = torch.jit.trace(model, dummy_input)
// traced_graph 包含:
//   - 所有算子及其类型 (Conv2d, BatchNorm, ReLU, MaxPool, Concat, ...)
//   - 算子间的数据依赖 (tensor producer → consumer)
//   - 每个算子的 input/output tensor shapes

// Step 2: Stream Assignment (详见 Stream Assignment Algorithm)
dag_nodes = extract_nodes(traced_graph)    // ~700 nodes for NASNet-A
dag_edges = extract_edges(traced_graph)     // data dependency edges
stream_plan = assign_streams(dag_nodes, dag_edges)
// Output: map<node_id, stream_id>, e.g.:
//   sep_conv_1 → stream_0, sep_conv_3 → stream_1 (并行分支)

// Step 3: Memory Plan
// 分析所有 intermediate tensor 的生命周期
// 实现 memory pre-allocation: 一个 tensor 的 memory 释放后可被后续 tensor 复用
liveness = analyze_tensor_liveness(traced_graph)
memory_plan = allocate_memory_pool(liveness)
// 与传统 PyTorch 的 per-operator malloc/free 不同:
// AoT 阶段一次分配所有需要的 GPU memory，无运行时 alloc/dealloc

// Step 4: CUDA Graph Capture + Instantiation
for each stream in streams:
    cudaStreamBeginCapture(stream)
    for op in get_ops_for_stream(stream):
        // 执行 operator → CUDA kernel launch 被记录
        op_impl = select_best_kernel(op)  // cuDNN vs PyTorch native
        op_impl(op_input, op_output)
    cudaStreamEndCapture(stream, &graph)
// 跨 stream 同步在 concat/add 等合并操作前插入 CUDA event
cudaGraphInstantiate(&exec_graph, graph)

// Step 5: Runtime Execution
// 对每个新输入:
cudaMemcpyAsync(nimble_input_buffer, new_input, ..., copy_stream)
cudaGraphLaunch(exec_graph, main_stream)
// No PyTorch framework participation — GPU 自主完成所有计算
```

AoT Scheduling 的本质转换：
```
Baseline (PyTorch eager, per-operator):
  for each operator op in model:
      CPU: dispatch(op) → infer_shape(op) → select_kernel(op) → 
           prepare_args(op) → cudaLaunchKernel(op)  ← 每 operator ~100μs CPU overhead
      GPU: execute kernel (~10μs for small kernels)
      → GPU idle while CPU schedules next op

Nimble (AoT Scheduling):
  AoT Phase (once): trace → assign_streams → capture CUDA Graph → instantiate
  Runtime (per input): cudaMemcpy + cudaGraphLaunch (single call)
      GPU: execute ALL kernels autonomously ← zero CPU overhead per operator
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AoT Scheduling 在 Nimble 中的实现基于 PyTorch TorchScript + CUDA Graph Capture API。用户接口极简（两行代码）：
```python
import nimble
nimble_model = nimble.Nimble(original_pytorch_model)  # 包装
output = nimble_model(input_tensor)                     # AoT preparation + inference
```
首次调用 forward 时触发 AoT preparation（trace + stream assignment + graph capture + instantiation），后续调用直接 replay CUDA Graph。

AoT Scheduling 的适用条件和限制：
- 适用：静态 DL 模型（无 data-dependent control flow，shapes 固定）。覆盖大多数 CNN、Transformer（inference）、ResNet、NAS 网络等 → "covers a wide range of models with practical, real-world impacts"（类比 TensorRT 的适用范围）
- 不适用：动态模型（dynamic control flow, variable-length sequences 且无法 padding）、训练中 batch size 变化循环
- 开销：AoT preparation 一次性平均 0.35s，后续无限次 amortize；额外 GPU memory 用于 CUDA Graph metadata 和 pre-allocated buffers
- 与 TensorRT/TVM 的关系：正交优化。TensorRT/TVM 做 graph optimization（operator fusion）+ kernel autotuning，Nimble 做 runtime scheduling overhead 消除。Nimble 叠加部分 fusion 后超越 TensorRT

涉及论文标题：
- Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning

---
