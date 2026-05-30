## Framework Scheduling Overhead

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Framework Scheduling Overhead（框架调度开销）是深度学习框架（PyTorch、TensorFlow 等）在每个 DL 算子的执行过程中，CPU 端为准备 GPU kernel launch 而产生的非计算开销。Nimble 论文指出，framework overhead 并非单一来源，而是由多个串行执行的 CPU-side 步骤组成：

1. **Operator Dispatch**：Python/C++ autograd engine 根据 tensor types/shapes/devices 查找对应的 Function 对象和 kernel 实现。涉及虚函数调用、type dispatch、autograd graph 构建等多个步骤。
2. **Output Shape Inference**：在 kernel launch 之前，CPU 计算输出 tensor 的 shape（meta-data computation）。对于每个 operator，需要根据 input shapes 和 operator-specific rules 推断 output shape，用于后续 operator 的内存分配和 shape 检查。
3. **GPU Kernel Selection**：从多个 candidate kernel implementations 中选择最优的。例如 Conv 算子可能有多达 20+ 种 cuDNN algorithm candidates（implicit gemm, winograd, fft 等），auto-tuner 需要根据 shape 和 hardware 选择最优的。
4. **Kernel Argument Preparation**：准备 CUDA kernel launch 参数——grid/block dimensions、shared memory size（dynamic）、tensor strides 和 pointers。每个 kernel 的这些参数都必须从 tensor metadata 转换而来。
5. **GPU Kernel Launch**：通过 CUDA driver API (`cuLaunchKernel`) 提交 kernel 到 GPU work queue。涉及 CUDA driver 的内部数据结构和 CPU→GPU command 传输（通过 MMIO 或 PCIe）。
6. **Memory Allocation**：`cudaMalloc`/`cudaFree` 在每次 operator 执行时可能发生，尤其是中间 tensor 的生命周期管理（PyTorch 的 caching allocator 部分减轻了此开销）。

这些 overhead 串行累积：例如一个 GPU execution time 仅 10μs 的小 separable conv，其 CPU scheduling overhead 可能高达 100μs，导致 GPU 在 90%+ 的时间 idle。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PyTorch eager mode 下每个 operator 的完整 overhead 流程：

```
// 用户代码: output = model(input)
// 模型 forward pass: for each operator op in [conv1, bn1, relu1, conv2, ...]:

// ========== Operator Execution (per-operator overhead) ==========

// Step 1: Operator Dispatch (~20-30μs)
op_fn = torch._C._get_operation(op_name)       // Python→C++ dispatch
autograd_ctx = AutogradContext()                // 构建 autograd 计算图
saved_tensors = pack_for_backward(inputs)       // 保存 backward 所需中间值

// Step 2: Output Shape Inference (~10-20μs)
output_dims = op_fn.infer_output_shape(input.shape)
output_dtype = input.dtype  // 通常与 input 相同
output_layout = infer_memory_layout(op, input)

// Step 3: Kernel Selection (~5-50μs, depends on operator type)
// 例如 Conv2d: cuDNN 提供多种 algorithm
algorithms = cudnnFindConvolutionForwardAlgorithm(
    input_desc, weight_desc, conv_desc, output_desc, 
    max_algorithms=20
)
best_algo = min(algorithms, key=lambda a: a.time)  // auto-tune
kernel_fn = algorithms[best_algo].kernel

// Step 4: Argument Preparation (~5-10μs)
grid_dim = compute_grid(output_dims, block_dim)
block_dim = min(block_dim, MAX_THREADS_PER_BLOCK)
shared_mem = estimate_shared_memory(op, block_dim)
kernel_args = pack_kernel_launch_params(grid_dim, block_dim, shared_mem, 
                                         input_ptr, weight_ptr, output_ptr, 
                                         strides, padding, ...)

// Step 5: Kernel Launch (~3-5μs)
cudaLaunchKernel(kernel_fn, grid_dim, block_dim, kernel_args, shared_mem, stream)

// Step 6 (implicit): Memory Allocation (amortized by caching allocator)
// PyTorch caching allocator reduces malloc/free cost but still has pool lookup overhead
// New intermediate tensors still require allocator round-trips

// ========== GPU Execution ==========
// GPU executes kernel: ~10μs (small separable conv)
// CPU idle waiting for GPU (or processing framework overhead for next op)

// === Summary for ~700 operators (NASNet-A mobile) ===
// GPU compute time: ~700 * 10μs = 7ms
// CPU overhead: ~700 * 100μs = 70ms
// GPU idle ratio: 70ms / (7ms + 70ms) ≈ 91%
// → PyTorch baseline measured GPU idle up to 91%
```

Nimble 的 AoT scheduling 如何消除这些 overhead：

```
// AoT Scheduling: 所有 overhead 发生在 AoT preparation 阶段 (一次)
// Runtime: 每个 operator 的执行被简化为 CUDA Graph 中的节点重放

// Per-operator overhead 的消除:
// Step 1 (Operator Dispatch):     ✗ eliminated — CUDA Graph 已包含 kernel 引用
// Step 2 (Shape Inference):       ✗ eliminated — shapes pre-determined in AoT
// Step 3 (Kernel Selection):      ✗ eliminated — kernel pre-selected in AoT
// Step 4 (Argument Preparation):  ✗ eliminated — args pre-recorded in graph nodes
// Step 5 (Kernel Launch):         → single cudaGraphLaunch (not per-op)
// Step 6 (Memory Allocation):     ✗ eliminated — memory pre-allocated in AoT

// Runtime cost per inference:
// cudaGraphLaunch overhead: <100μs total (vs baseline's 70ms CPU overhead)
// GPU execution: ~7ms (same as baseline)
// → 22x speedup for NASNet-A mobile
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
除 Nimble 的 AoT scheduling 外，业界也存在其他减少 framework overhead 的方法：

1. **Memory Pre-allocation**：PyTorch caching allocator、TensorFlow memory planner 减少 runtime alloc/free，但仅解决 Step 6。
2. **CUDA Graph (per-step)**：vLLM 和 SGLang 将 decode step 录制为 CUDA Graph，消除 decode iteration 的 launch overhead。但仅适用于 shapes 固定的重复性步骤。
3. **TorchScript / torch.compile**：通过 JIT compilation 和 operator fusion 减少 total operator count（从而减少 dispatch 次数），但不能消除每个 operator 的 dispatch。
4. **Framework Redesign**：如 TensorFlow XLA 将整个计算图编译为单一可执行对象。但论文指出 "Redesigning the framework to remove all sources is very challenging"，而且仍有 JIT compilation 开销。

Nimble 的关键 insight：与其逐个修复 overhead 来源，不如**完全绕过** framework runtime——利用 CUDA Graph 的 record-then-replay 能力，将 GPU 执行与 CPU framework 解耦。

影响程度取决于模型特性：当模型由 large kernels（大矩阵乘、大 conv）组成时，framework overhead 占比小 → limited speedup（如 BERT training、ResNet-50 ImageNet training）。当模型包含大量 small kernels（mobile-optimized CNNs、NAS architectures 等）时，framework overhead 主导 → dramatic speedup (up to 22x)。

涉及论文标题：
- Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning

---
