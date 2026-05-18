## Load-Evict Overlap with CUDA Stream Pipeline（CUDA流管线的KV Cache加载-驱逐重叠）

术语是什么？

Load-Evict Overlap是TokenFlow Hierarchical KV Cache Manager中的CUDA stream调度技术，专用于请求抢占和恢复场景下的KV cache在GPU HBM和CPU DRAM之间的传输优化。核心思想是利用CUDA stream的异步并发能力，将"被抢占请求KV cache的GPU显存释放（evict）"与"恢复请求KV cache的CPU→GPU加载（load）"在时间上重叠执行，避免两者串行执行时的I/O stall。

从kernel调度角度拆解术语：

Load-Evict Overlap的执行流程（以请求R1被抢占、R3恢复为例）：

```
// Step 1: Scheduler决定抢占R1、恢复R3
pending_evict = get_unsynced_chunks(R1)  // write-through已同步大部分，剩余少量chunk
pending_load = get_kv_chunks(R3)         // R3的KV cache在CPU DRAM中

// Step 2: 创建/复用CUDA streams
compute_stream = get_cuda_stream("compute")
load_stream = get_cuda_stream("load")  
evict_stream = get_cuda_stream("evict")

// Step 3: 记录compute stream当前进度
cudaEventRecord(compute_done, compute_stream)

// Step 4: 使load和evict stream等待compute完成
cudaStreamWaitEvent(load_stream, compute_done)
cudaStreamWaitEvent(evict_stream, compute_done)

// Step 5: 并行发射load和evict操作
// load_stream: H2D传输 R3的KV chunk
for each chunk in pending_load:
    cudaMemcpyAsync(gpu_kv[chunk.dst], cpu_kv[chunk.src], 
                    chunk.size, cudaMemcpyHostToDevice, load_stream)

// evict_stream: 释放R1未同步chunk的GPU显存
// 已同步的chunk直接free(无需等待)，未同步的chunk先D2H再free
for each chunk in pending_evict:
    if chunk.is_synced:
        free_gpu_block(chunk.gpu_addr)  // 即时释放
    else:
        cudaMemcpyAsync(cpu_kv[chunk.dst], gpu_kv[chunk.src],
                        chunk.size, cudaMemcpyDeviceToHost, evict_stream)
        // 传输完成后释放
        cudaFreeAsync(chunk.gpu_addr, evict_stream)

// Step 6: 同步barrier
cudaEventRecord(load_done, load_stream)
cudaEventRecord(evict_done, evict_stream)

// Step 7: compute stream等待load完成
cudaStreamWaitEvent(compute_stream, load_done)

// Step 8: 恢复compute——R3的decode开始
launch_decode_kernel(R3, compute_stream)
```

关键设计点：
1. **动态Chunk Sizing**：`chunk.size`不是在启动时固定。estimator根据预估的下一轮compute duration自适应选择chunk size，确保`transfer_time(chunk) ≤ compute_duration`，最大化compute-I/O overlap。
2. **Batched Transfer**：多个请求的KV chunk合并为单次cudaMemcpyAsync调用→减少DMA engine setup次数→提升PCIe带宽实际利用率。
3. **CUDA Event同步**：使用cudaEventRecord/cudaStreamWaitEvent实现细粒度stream间依赖（而非全局cudaDeviceSynchronize），避免阻塞无关stream。
4. **Write-Through预同步**：大部分KV cache已在正常decode过程中通过write-through同步到host→抢占时pending_evict通常极小（仅最近1-2次decode iteration的增量）→evict延迟远小于load延迟→load成为瓶颈→load-evict overlap的价值在于让evict不阻塞load。

术语一般如何实现？如何使用？

在TokenFlow中通过PyTorch CUDA stream API（torch.cuda.Stream）实现：系统维护三类persistent CUDA stream对象（而非每次创建/destroy），Python threading为每类stream分配独立控制线程。CUDA events在compute→load/evict→compute的依赖链上建立同步点。消融实验：去掉evict-load overlap后完成时间明显增加。TokenFlow未开源。

与普通KV cache offload方案的区别：普通方案通常在显存压力下被动触发evict（reactive），且evict/load串行；TokenFlow的load-evict overlap主动（proactive，write-through提前准备）+重叠（overlap，CUDA stream并发），将KV cache搬移从系统瓶颈转化为可与计算重叠的后台操作。

涉及论文标题：
- TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling

