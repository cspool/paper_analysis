## Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是Nimble DL执行引擎，包含两个核心技术：(1) Ahead-of-Time (AoT) Scheduling——在模型执行前，通过CUDA Stream Capture API预运行模型一次（使用dummy input），拦截所有GPU kernel调用和内存分配，生成CUDA Graph（包含dispatched GPU kernels、函数参数、提交顺序和task-to-stream assignment的记录）。运行时通过CUDA Graph Launch API重放，完全绕开PyTorch框架的runtime scheduling overhead。(2) Automatic Multi-Stream Execution——自动将算子分配到多个CUDA stream在单GPU上并行执行。核心是stream assignment算法：计算DAG的Minimum Equivalent Graph (MEG)，从MEG构建bipartite graph，通过Ford-Fulkerson算法寻找maximum matching，将独立节点分配到不同stream同时最小化跨stream同步。理论证明该算法实现最大逻辑并发度+最小同步数。

  实验比较：(a) vs PyTorch inference speedup；(b) vs TorchScript speedup；(c) vs Caffe2 speedup；(d) vs TensorRT v7.1 inference speedup；(e) vs TVM v0.6.1 inference speedup；(f) vs PyTorch training speedup；(g) Nimble single-stream vs multi-stream 消融实验。

- 后端平台是什么，配置是什么。
  NVIDIA V100 GPU + 2.10GHz Intel Xeon CPU E5-2695 v4。Software: PyTorch v1.4, CUDA 10.2, cuDNN 8.0.2。使用TorchScript graph作为输入。

- 评估性能的软件/脚本是什么。修改了什么。
  Nimble基于PyTorch构建，使用TorchScript graph作为输入。用户只需将PyTorch模型包装进Nimble对象（两行额外代码）：`nimble_model = nimble.Nimble(model)`，然后对`nimble_model`执行inference或training。修改：(1) Graph Rewriter —— 执行stream assignment算法，将TorchScript graph中的算子分配到多个CUDA stream；(2) AoT Scheduler —— 使用CUDA Stream Capture API记录GPU kernel call trace（包括kernel launches和memory allocations）并生成CUDA Graph；(3) 实现部分operator fusion（不如TensorRT aggressive）和Conv算子的basic kernel selection（cuDNN vs PyTorch native）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接：https://github.com/snuspl/nimble

  评估原理：
  1. 将PyTorch模型包装为Nimble对象（`nimble.Nimble(model)`）
  2. AoT preparation阶段：使用dummy input预运行模型，Nimble通过CUDA Stream Capture API记录所有GPU kernel调用和内存分配，生成优化的CUDA Graph（含多stream分配）。AoT preparation平均耗时0.35s，最大1.07s（NASNet-A large），这是一次性开销。
  3. 运行时：对每个新输入，Nimble直接通过CUDA Graph Launch API重放录制的trace，完全绕开PyTorch的operator dispatch、shape inference、kernel selection和argument preparation等框架开销。
  4. 多stream并行：重放时自动在多个CUDA stream上并发执行独立kernel，跨stream同步点由stream assignment算法最小化。
  5. 测量inference latency（batch size 1）和training throughput。

  全过程（以NASNet-A mobile在V100上的inference为例）：
  ```
  Step 1: 用户代码
    import nimble
    model = torchvision.models.nasnetamobile()  # 加载PyTorch模型
    nimble_model = nimble.Nimble(model)          # 包装为Nimble对象
  
  Step 2: AoT Preparation (执行一次)
    dummy_input = torch.randn(1, 3, 224, 224)
    nimble_model.prepare(dummy_input)
    
    Nimble内部流程:
    ┌─ TorchScript Trace ───────────────────────────────────────┐
    │ torch.jit.trace(model, dummy_input)                       │
    │ → 生成TorchScript graph（包含所有算子及其依赖）             │
    └───────────────────────────────────────────────────────────┘
    
    ┌─ Graph Rewriter (Stream Assignment) ──────────────────────┐
    │ ① Build computation DAG from TorchScript graph            │
    │ ② Compute Minimum Equivalent Graph (MEG)                  │
    │ ③ Construct bipartite graph from MEG                      │
    │ ④ Ford-Fulkerson maximum matching → stream assignment     │
    │ ⑤ 每个算子被分配到一个CUDA stream，独立算子分配到不同      │
    │    stream，有依赖的算子通过CUDA event同步                   │
    │                                                             │
    │ Example: NASNet-A cell (branch structure):                 │
    │   sep_conv1 (stream 0)  ∥ sep_conv3 (stream 1)           │
    │   sep_conv2 (stream 0)  ∥ sep_conv4 (stream 1)           │
    │   sep_conv5 (stream 0)  ∥ sep_conv6 (stream 1)           │
    │   → concat (stream 0, sync across streams)                │
    └───────────────────────────────────────────────────────────┘
    
    ┌─ AoT Scheduler (CUDA Graph Capture) ─────────────────────┐
    │ ① CUDA Stream Capture API: cudaStreamBeginCapture(stream) │
    │ ② 在多个stream上执行模型forward pass:                    │
    │    - 每个stream按分配执行其算子                            │
    │    - Memory allocation被拦截和记录                         │
    │    - CUDA kernel launches、arguments全部被record           │
    │ ③ cudaStreamEndCapture(stream, &graph)                    │
    │ ④ CUDA Graph Instantiation: cudaGraphInstantiate(&exec)   │
    │ ⑤ 输出的CUDA Graph包含:                                   │
    │    - 所有GPU kernel调用序列                                │
    │    - 多stream执行的拓扑                                    │
    │    - 跨stream同步点                                       │
    │    - Pre-allocated memory buffers                         │
    │ AoT preparation time: 0.35s (mean), 1.07s (NASNet-A large)│
    └───────────────────────────────────────────────────────────┘

  Step 3: Runtime Inference (每次新输入)
    output = nimble_model(new_input)
    
    执行流程:
    ┌─ CUDA Graph Replay ──────────────────────────────────────┐
    │ cudaGraphLaunch(exec, stream)                             │
    │                                                             │
    │ GPU端直接执行预录制的kernel序列（无PyTorch框架参与）:       │
    │                                                             │
    │ Stream 0: sep_conv1 → sep_conv2 → concat_wait             │
    │ Stream 1: sep_conv3 → sep_conv4 → concat_signal           │
    │ ↓ cudaEventSynchronize (cross-stream barrier)              │
    │ Stream 0: concat → avg_pool → fc → softmax                │
    │                                                             │
    │ 完全绕过PyTorch的runtime overhead:                        │
    │   ✗ operator dispatch (no Python/C++ operator lookup)     │
    │   ✗ output shape inference                                │
    │   ✗ GPU kernel selection (already recorded)               │
    │   ✗ kernel argument preparation (already recorded)        │
    │   ✗ memory allocation (pre-allocated in AoT phase)        │
    └───────────────────────────────────────────────────────────┘
  
  输出性能：
    - vs PyTorch inference: up to 22.34× speedup (NASNet-A mobile, batch_size=1)
    - vs TensorRT v7.1: up to 2.81× speedup (NASNet-A mobile)
    - vs TVM v0.6.1: up to 1.70× speedup (EfficientNet-B5)
    - Multi-stream vs single-stream (Nimble内部): up to 1.88× speedup
    - Training speedup (CIFAR-10 small models): up to 3.61×
    - GPU idle time in baseline: PyTorch up to 91%, TensorFlow up to 71%
    - Large model training (BERT, ResNet-50 ImageNet): limited speedup（kernel本身计算量大，框架overhead占比小）
  ```

  关键设计要点：
  - CUDA Graph Capture使AoT preparation仅需一次dummy forward pass，之后所有执行跳过框架runtime
  - Stream assignment algorithm的理论保证：最大并发度+最小同步数，通过MEG+Bipartite Matching实现
  - 限制：仅支持static neural network models（不支持dynamic control flow），与TensorRT类似
  - 框架overhead来源：output shape inference, kernel dispatch, kernel argument preparation, memory allocation等多个方面，非仅memory allocation一项
  - Nimble与TensorRT/TVM的关系：正交优化。TensorRT/TVM做graph optimization和kernel tuning，Nimble做runtime scheduling overhead消除。Nimble叠加operator fusion后超越TensorRT（除MobileNet V2的TVM特例：TVM花>1天tuning Conv kernel找到了比cuDNN更高效的实现）
