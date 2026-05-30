## FineMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - FineMoE 的 kernel 调度/运行时计算实现：
    1. **C++ Expert Cache with CUDA Runtime API**：Expert management in GPU 使用 CUDA Runtime API 实现。Expert ID 通过 hash map 映射到不同 GPU devices，按 round-robin 分配以平衡 GPU 负载。GPU space 中的 task pool 使用异步线程调度和执行 expert prefetching 与 on-demand loading 任务。
    2. **异步 Publisher-Subscriber 架构**：将 Expert Map Searcher 的 map searching 和 expert prefetching 与 inference process 解耦。Expert Map Store 作为 message broker，inference process 持续 publish context 数据（semantic embeddings + expert probability distributions），Expert Map Searcher subscribe context 并异步 prefetch experts 到 Expert Cache。
    3. **Multi-GPU Expert Parallelism**：支持 multi-GPU inference with EP，experts 映射到不同 GPU devices 进行加载和 offloading。Expert 分配遵循 round-robin 均衡负载。
    4. **On-demand Expert Loading**：当 expert miss 发生时（gate network 指定的 expert 不在 GPU cache 中），FineMoE 暂停所有 expert prefetching 任务，立即从 CPU 加载缺失的 experts 到 GPU 内存以进行 fast serving。
    5. **Expert Eviction Priority**：基于 LFU + searched probability 的联合优先级：PRI^{evict}_{l,j} = 1 / (p_{l,j} * freq_{l,j})，低概率 + 低频使用 = 高 eviction 优先级。
  - 实验比较：
    - System overheads：图 17 展示 one iteration 的 latency breakdown——context collection、on-demand loading、异步的 map searching/prefetching/map update 各占多少
    - 结果：除异步操作外的总延迟 < 50ms（<1% iteration），可忽略不计
    - Ablation study on caching：FineMoE vs LRU vs LFU

- 后端平台是什么，配置是什么。
  - **RTX 3090 测试台**：6× NVIDIA GeForce RTX 3090 24GB, NVLink 互联, PCIe 4.0 32GB/s, AMD Ryzen Threadripper PRO 3955WX, 480GB CPU memory
  - **A100 测试台**：1× NVIDIA A100 80GB HBM2e, 2 TB/s 内存带宽
  - CUDA Runtime API 用于 expert management，多 GPU 间通过 EP (expert parallelism) 分布

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 **MoE-Infinity 代码库**（https://github.com/TorchMoE/MoE-Infinity）进行修改
  - Expert Cache 在 C++ 层修改 MoE-Infinity 的 expert management：
    1. 新增 async task pool 用于 prefetching 和 on-demand loading 任务的调度
    2. 修改 CUDA memory management 逻辑以支持 similarity-aware prefetching priority
    3. 修改 eviction 逻辑：从纯 LFU 改为 LFU + probability-based priority
  - 评估方法：测量 latency breakdown（图 17），TPOT under varying cache limits（图 12），prefetch distance sensitivity（图 15）

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：基于 MoE-Infinity（https://github.com/TorchMoE/MoE-Infinity），FineMoE prototype 未发现独立开源仓库
  - **Expert Cache Kernel 执行全过程**（单 GPU, Mixtral-8×7B 推理为例）：
    1. **Expert Prefetching Task 入队**：Expert Map Searcher 确定 E_prefetch = {E_{l,j}}，计算每个 expert 的 prefetching priority = p_{l,j} / (l - l_now) → 按 priority 降序入队到 GPU task pool
    2. **Asynchronous Prefetching 执行**：GPU space 异步线程从 task pool 取最高 priority 任务 → CUDA Runtime API cudaMemcpyAsync(host_ptr, device_ptr, expert_size, cudaMemcpyHostToDevice, stream) → PCIe 4.0 32GB/s 传输 expert weights 从 CPU 到 GPU → 更新 Expert Cache（hash map expert_id → cached location）
    3. **Inference Forward Pass 同步执行**：推理进程持续执行 forward——每层 gate network 选 top-K experts → 查 Expert Cache hash map：若命中则 CUDA GEMM kernel 直接使用 GPU 上的 expert weights 计算 → 若 miss 则暂停 prefetching task pool，立即 cudaMemcpy 该 expert 从 CPU to GPU → 执行 forward
    4. **Expert Eviction**：当 Expert Cache 达到 GPU memory budget 时 → 遍历所有 cached experts → 计算 eviction_priority = 1/(p_{l,j} * freq_{l,j}) → cudaFree 释放最高 eviction priority（最不重要）的 expert 的 GPU memory → 可用空间继续容纳新 prefetch 的 experts
    5. **Performance Metric**：TPOT = (推理计算时间) + (expert miss count × T_e on-demand loading time)。expert hit rate = 1 - (expert miss count / total expert activations)。Latency breakdown 通过 profiling 各操作的 wall-clock time 获得
    6. **Evaluation 原理**：通过控制 Expert Cache GPU memory budget（6GB-96GB），测量不同 memory 约束下的 TPOT。FineMoE 在相同 GPU memory 下通过更精确的 expert prediction（higher hit rate = fewer on-demand loads）实现更低 TPOT
