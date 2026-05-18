## Daemon Kernel for Device-Side Launch

术语是什么？通过联网搜索让回答具体和精准。
Daemon Kernel是Infera中常驻GPU一个SM的persistent kernel，负责device-side kernel launch和管理。它通过CUDA Dynamic Parallelism (CDP) 的cudaLaunchDevice接口以fire-and-forget方式直接从device发射fused kernel，无需host端参与每次launch。daemon kernel维护DKQ（shared memory double-ended queue），从中取出kernel并发射，kernel完成后执行cudaGetLastError错误检查。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Daemon kernel在Infera serving pipeline中的运转流程：
1. **初始化**：Infera server启动时，在一个SM上launch daemon kernel（persistent grid，永不退出）
2. **等待kernel**：daemon kernel spin-wait在DKQ（device-side shared memory double-ended queue）上，检查是否有新kernel到达
3. **发射kernel**：DKQ非空→daemon kernel取出kernel pointer+argument pointer+launch config→cudaLaunchDevice(kernel, args, gridDim, blockDim, sharedMem, stream=0)→fire-and-forget：GPU scheduler立即调度该kernel，不等待前序grids完成
4. **Completion检查**：GPU通知daemon kernel某kernel完成→daemon执行cudaGetLastError()→若error则通知host TSU
5. **低延迟优势**：fire-and-forget launch latency <10μs，无stream tracking overhead，无Head-of-Line blocking
6. **Preemption响应**：daemon kernel检测preemption signal→保存DKQ和shared memory kernel上下文→等待恢复

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera的daemon kernel独占一个SM（A100上即1/108≈0.93% SM资源），使用CUDA Dynamic Parallelism（需GPU Compute Capability ≥3.5）。daemon kernel的shared memory double-ended queue缓冲kernel描述符，支持快速入队/出队。传统host-side kernel launch需通过CUDA runtime→CUDA driver→GPU work queue，引入host-device同步和排队延迟；device-side launch跳过这些开销，特别适合Infera的高频micro-kernel发射（每个scheduling cycle可能发射数十个fused kernel）。daemon kernel开销低于1/#SM（A100上<1%），preemption latency Infera-P约10μs、Infera-R约5μs。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling
