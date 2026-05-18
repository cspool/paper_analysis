## Wavefront Specialization（波前专业化）

术语是什么？通过联网搜索让回答具体和精准。

Wavefront Specialization（也称warp specialization）是GPU kernel编程优化技术，将workgroup内的wavefront（NVIDIA术语warp）分配不同角色，而非所有wavefront执行相同指令（传统SIMT模式）。典型producer-consumer分工：一个或多个dedicated producer wavefront专门执行memory transfer（如ATT load从global memory搬运tile到LDS），其余consumer wavefronts专门执行computation。这种分工引入compute heterogeneity但enable fine-grained overlap of data movement and computation——producer wavefront在后台异步搬运数据时，consumer wavefronts在前台同时计算前一个tile。该技术要求精确同步（custom barriers）防止data race和死锁。Wavefront specialization最初由Bauer等人（PPoPP'14, Singe）提出，后在NVIDIA Hopper架构上因TMA+WGMMA的异步能力成为主流优化模式（CUTLASS 3.x, FlashAttention-3均采用）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

QuCo论文中wavefront specialization + Operand Queue的kernel执行伪代码（以Matrix-Matrix为例）：

```
// ===== Kernel Dispatch =====
// Workgroup = N wavefronts: WF 0 是 Producer, WF 1..N-1 是 Consumers

// ===== Producer Wavefront (WF 0) =====
if (wf_id == 0):
    for each tile_i in K_dimension:
        queue.Push(tile_i)          // 发起 ATT async tile load → LDS
        async_barrier.arrive()      // non-blocking: 通知数据就绪
        queue.Wait_For_Push()       // 确保 ATT 写入 LDS 完成
    async_barrier.exit_producer()

// ===== Consumer Wavefronts (WF 1..N-1) =====
else:
    while has_work:
        async_barrier.wait()        // 阻塞直到 producer arrive
        tile_data = queue.Peek(idx) // 只读访问 LDS 中 tile
        partial += compute(tile_data) // 矩阵乘法等计算
        queue.Pop(idx)              // 释放 LDS slot 供下一 tile
```

关键参数决策（QuCo自动化的内容）：
1. **tile size**：决定每次ATT transfer数据量→影响memory bandwidth utilization和LDS占用→需与kernel compute intensity匹配
2. **queue slots数量**：决定多少tile可同时in-flight→slots过少无法充分overlap（pipeline bubble），slots过多浪费LDS且增加memory contention
3. **consumer wavefront数量**：决定compute parallelism→更多consumer更快消费tile但增加scheduling pressure

传统手动wavefront specialization的痛点：同一kernel在不同GPU上的最优配置不同（R9 Nano→MI-100差1.4×），不同kernel间不能复用（可达1.2× degradation），需exhaustive DSE（Matrix-Matrix需2.6×10^14次kernel launch逐一尝试）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

现代GPU上wavefront specialization实现方式：
1. **CUDA/CUTLASS**：通过thread block内thread ID判断角色——if (threadIdx.x == 0)执行TMA load，其余线程执行MMA compute。CUTLASS 3.x使用warp group specialization（4 warps per warp group, producer vs consumer groups）
2. **Triton语言**：Tawa项目（arXiv 2510.14719）在Triton IR层面引入asynchronous references (aref)自动将程序分区为producer/consumer warps
3. **AMD GPU (ROCm/HIP)**：使用wavefront ID区分角色（__builtin_amdgcn_workitem_id_x() / warpSize）
4. **关键硬件前提**：需async memory transfer（ATT/TMA/cp.async）+ async barrier（mbarrier或等价机制）+ per-warp register allocation（如setmaxnreg控制register file占用）

涉及论文标题：
- QuCo: Efficient and Flexible Hardware-Driven Automatic Configuration of Tile Transfers in GPUs

