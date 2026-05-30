## APTMoE Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  APTMoE 的**需求优先级调度策略（Demand-Priority Scheduling Strategy）**负责协调三层加载阶段产生的 CUDA 数据移动 kernel，解决它们对同一 PCIe 带宽的竞争和互相阻塞问题。核心实现包括：
  - **优先级队列调度**：三个加载阶段分别维护队列，按需求紧迫度分配优先级——inter-expert phase（最高）> inter-layer phase（中等）> inter-stage phase（最低）。程序通过 PriorityQueue 管理，周期性查询 GPU 加载状态后动态选择最高优先级的待加载 kernel 发起。
  - **CUDA Event 前探机制**：在加载 stream 的倒数第二个 action 前插入 `torch.cuda.Event()`，利用 `event.query()` 检测加载进度，当 event 触发时表示前一个数据移动 kernel 仍在执行，可在此时决定下一个加载 kernel，隐藏 kernel launch overhead。
  - **双 Stream 重叠**：维护 `comp_stream`（计算 stream）和 `load_stream`（加载 stream），通过 `torch.cuda.Event()` 建立 inter-stream dependency，确保加载完成后再触发对应计算 kernel。
  实验比较吞吐量（tokens/s）、不同设备拓扑下的 speedup、强扩展性（4→16 GPU）。

- 后端平台是什么，配置是什么。
  NVIDIA A800 GPU (40GB)，Intel Xeon Gold 6348 CPU (28核)，GPU间通过PCIe Switch通信，节点内存1024GB。软件栈：Ubuntu 22.04.3 + PyTorch 2.0.0+cu117。

- 评估性能的软件/脚本是什么。修改了什么。
  APTMoE 基于自研 pipeline 框架（APTMoE/Runtime/PipelineRuntime/pipeline_runtime.py）实现，基线包括 GPipe、GPipeOffload、Mobius。核心修改：
  - `comm_scheduler.py`：实现 PriorityQueue 管理三层加载队列，torch.cuda.Event.query() 检测加载状态，周期性查询+动态调度
  - `offload.py`：实现三层加载阶段的数据移动决策，通过添加/移除 block 名称到对应队列管理加载
  - `R_solver.py`：实现 Equation 1 的最优 CPU/GPU 分配方案求解
  - 使用 `psutil.Process().cpu_affinity()` 绑定不同数量的 CPU 核心到特定进程以设置不同设备拓扑

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  **开源**：https://github.com/Atopos-309/APTMoE

  **执行命令示例**：
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 ./main.py \
    --is_moe=True --num_training_steps=50 --model_config=S \
    --num_experts=16 --gini=0.3 --topo=C1+G2 --pipeline=APTMoE
  ```

  **Demand-Priority Kernel 调度执行流程**：
  1. **输入**：三层加载队列（inter-stage queue / inter-layer queue / inter-expert queue）中各包含待加载的 model block 名称
  2. **优先级决策**：调度器检查三层队列，按 inter-expert > inter-layer > inter-stage 优先级从非空队列中选择下一个加载目标
  3. **CUDA Event 插入**：在 load_stream 的倒数第二个数据移动 kernel 前插入 cuda_event，event 触发时发起新 kernel，隐藏 launch latency
  4. **Kernel 发起**：将选定的 model block 数据移动 kernel（host→device cudaMemcpy）提交到 load_stream
  5. **Inter-Stream 同步**：每个 model block 关联一个 torch.cuda.Event()，load_stream 完成数据移动后 record event，comp_stream 的对应计算 kernel 通过 stream wait 等待该 event
  6. **计算执行**：comp_stream 执行被加载 block 的 forward/backward 计算（MHA、gate、expert 等），与 load_stream 的下一次加载并行
  7. **输出**：每 iteration 的吞吐量（tokens/s），Step 结束后报告整体吞吐

  **调度关键点**：由于中断和恢复 CUDA kernel 执行极其困难且昂贵，APTMoE 选择在 kernel **启动前**而非执行中进行调度决策。通过 event.query() 的主动轮询机制，scheduler 可以在前一个加载 kernel 仍在执行时决定下一个加载内容，确保 PCIe 带宽得到最大化利用。
