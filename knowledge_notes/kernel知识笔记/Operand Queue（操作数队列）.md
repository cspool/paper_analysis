## Operand Queue（操作数队列）

术语是什么？通过联网搜索让回答具体和精准。

Operand Queue是QuCo论文提出的高层ATT（Asynchronous Tile Transfer）编程抽象，封装ATT descriptors管理、LDS buffer分配和producer-consumer同步，将底层异步数据传输暴露为对程序员透明的队列接口。每个Operand Queue由以下参数组成（由QuCo自动配置）：tile size（每slot元素数）、number of slots（queue深度，决定pipeline overlap程度）、queue type（streaming——数据单次消费后释放 vs stationary——数据持久化供多次消费）、LDS base address和barrier indices。灵感来自NVIDIA cuda::pipeline API（单/多阶段pipeline包装TMA操作）、CUTLASS3+CuTe的TMA pipelines和ThunderKittens的asynchronous I/O抽象。核心差异：这些高级框架仍要求程序员手动选择tile sizes, stages, descriptor参数，而Operand Queue的所有参数由QuCo hardware在kernel launch时自动计算。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// ===== Queue 配置（QuCo 自动完成） =====
// Host: driver.RegisterQueue(K_dim=2048, elem=4B, TYPE_STREAMING)
//  QuCo → tile_size=512, slots=2, LDS[offset:offset+4096B]

// ===== 2-Slot Streaming Queue 时间轴 =====
// t0: Producer Push(slot_0) → ATT engine load tile_0
// t1: [ATT transfer ████████]  Consumer idle (waiting)
// t2: [ATT transfer ████████]  Wait_For_Push → arrive()
// t3: Producer Push(slot_1) → ATT load tile_1, 
//     Consumer wait() → Peek(slot_0) → compute
// t4: [ATT transfer ████████]  Consumer compute(slot_0)
// t5: [ATT transfer ████████]  Consumer Pop(slot_0)
//     Producer Wait_For_Push → arrive()
// t6: Producer Push(slot_0) → ATT load tile_2,
//     Consumer wait() → Peek(slot_1) → compute
// ...pipeline continues with 2-slot double buffering

// ===== Queue API =====
queue.Push(tile_idx)        // Producer: 发起 ATT async load → slot[tile_idx % slots]
queue.Wait_For_Push()       // Producer: 等待 ATT engine 完成写入 LDS
data = queue.Peek(idx)      // Consumer: 只读访问 LDS 中 tile（不释放）
queue.Pop(idx)              // Consumer: 标记 slot 为 free
```

Streaming vs Stationary queues：
- **Streaming**：tile单次消费后释放，适合无数据复用的kernel（Elementwise, Dot-Product）。QuCo用Little's Law（slots = memory_transfer_time / compute_time）计算slot数
- **Stationary**：tile在LDS中持久化供多次访问，适合有数据复用的kernel（Matrix-Matrix weight tile）。slot数由剩余LDS容量均分

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

当前实现方式（类比框架）：
1. **NVIDIA cuda::pipeline**：cuda::pipeline_shared_state + memcpy_async + pipeline_commit_wait，支持单/多阶段pipeline。但stages数和tile size需程序员手动选择
2. **CUTLASS 3.x pipelines**：MainloopPipeline/TileSchedulerPipeline等，内部管理TMA descriptor pass、shared memory tile buffer allocation、warp group barrier synchronization
3. **ThunderKittens async I/O**：提供高层次的asynchronous copy抽象，封装底层TMA/cp.async路径
4. **QuCo Operand Queue**：仍为学术proposal，核心创新在于queue参数（tile size, slots, LDS layout）由硬件自动计算，而非程序员指定

涉及论文标题：
- QuCo: Efficient and Flexible Hardware-Driven Automatic Configuration of Tile Transfers in GPUs

