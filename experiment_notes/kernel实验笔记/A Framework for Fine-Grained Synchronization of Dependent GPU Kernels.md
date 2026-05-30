## A Framework for Fine-Grained Synchronization of Dependent GPU Kernels

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  cuSync 是一个 header-only 的 CUDA 库，实现四个核心机制来实现依赖 GPU kernel 的细粒度同步：(i) 将依赖 kernel 分别发射到不同 CUDA stream 上消除 stream synchronization；(ii) wait-kernel 机制确保 producer kernel 的 thread block 先于 consumer kernel 被调度到 SM 上；(iii) 通过 atomic global counter 实现自定义 tile 处理顺序（如 RowMajor）以最小化 consumer 的等待时间；(iv) 使用 global memory semaphore 数组 + memory fence 实现 post/wait 的 tile 级依赖同步。
  实验比较：cuSync 的多种同步策略（TileSync、RowSync、StridedSync、Conv2DTileSync 及带 W/R/T 优化的变体）vs. CUDA Stream Synchronization（StreamSync）和 Stream-K。评估指标是各 ML 模型端到端推理延迟的减少百分比。

- 后端平台是什么，配置是什么。
  NVIDIA DGX-2 系统，含 8 块 NVIDIA Tesla V100 32GB GPU，通过 NVLINK 互联。CPU 为 2.60GHz 12-core Intel Xeon E5-2690，448GB RAM。CUDA 12.2。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 NVIDIA CUTLASS 3.1 的 GeMM 和 Conv2D CUDA kernel。修改包括：在 GeMM kernel 中添加 cuSync 的 stage.start()、stage.tile()、stage.wait()、stage.post() 调用（约 25 行，占 CUTLASS GeMM 代码的 0.5%）；在 Conv2D kernel 中添加类似调用（约 22 行，占 0.6%）；自研 Softmax-Dropout 融合 kernel 中添加同步调用（约 5 行，占 1%）。评估脚本位于 `src/ml-bench/volta_transformer/eval_llm.py`（LLM）和 `src/ml-bench/volta_conv2d/eval_conv.py`（CV），运行 20 次取平均，warmup 5 次。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/microsoft/cusync，CGO 2024 Artifact Evaluation 分支 `cgo-24-ae`。

  评估原理：cuSync 在每个 tile-based kernel 内部插入同步点。Producer kernel 计算完一个 tile 后调用 `stage.post(row, col)`，通过 `__threadfence_system()` 保证 global memory 写入可见后，对 policy 指定的 semaphore 做 `atomicAdd`。Consumer kernel 加载 tile 前调用 `stage.wait(tile, grid)`，第一个线程在 global memory semaphore 上 busy-wait 直到值达到预期，其余线程被 `__syncthreads` 阻塞。这样 consumer 的 tile 只需等待依赖的 producer tile 完成，而非等待整个 producer kernel。

  执行流程：
  1. 用户用 cuSyncGen DSL 描述 kernel 间的 tile 依赖关系（如 MLP 中第二个 GeMM 的每个 consumer tile 依赖第一个 GeMM 同行所有列 tile）
  2. cuSyncGen 生成 policy 类（sem/value 方法）和 tile 处理顺序函数
  3. 用户修改 CUDA kernel，在加载前调用 wait，计算后调用 post，使用 CuStage 实例化 policy
  4. 主函数创建 CuStage 对象，声明依赖关系（CuSync::dependency），在独立 stream 上发射 kernel
  5. consumer stream 先发射 wait-kernel（单线程 busy-wait 等待 producer 开始），然后发射 consumer kernel
  6. 测量 kernel 执行时间，比较不同 policy（TileSync/RowSync/StridedSync）和优化组合（+W 去 wait-kernel, +R 重排 tile load, +T 去自定义 tile order）的性能
