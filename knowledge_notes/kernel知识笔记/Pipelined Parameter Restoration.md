## Pipelined Parameter Restoration

术语是什么？通过联网搜索让回答具体和精准。

Pipelined Parameter Restoration是TZ-LLM提出的TEE内LLM推理的运行时调度机制，将LLM推理由传统串行流程（先完整加载/解密参数→再计算）改造为allocation→loading→decryption→computation四种operator的DAG流水线。核心理念：利用LLM computation graph的确定性拓扑顺序（parameters按layer顺序访问），将参数restoration操作（CMA allocation、flash I/O、AES decryption）与CPU/NPU computation重叠执行，把restoration latency隐藏到computation latency下。配合priority-based greedy scheduling、preemptive micro-operator scheduling和partial parameter caching，使按需动态扩展secure memory的overhead不再完全暴露在TTFT critical path上。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Pipelined Parameter Restoration的调度算法：

```
// 扩展computation graph: 在每个computation operator前插入restoration operators
for layer_idx in topological_order:
    comp_op = original_graph[layer_idx]
    alloc_op = new AllocOp(layer_idx, param_size[layer_idx])
    load_op  = new LoadOp(layer_idx, param_size[layer_idx])
    decr_op  = new DecrOp(layer_idx, param_size[layer_idx])
    // 依赖: alloc → load → decr → comp
    // 跨层无依赖: comp[layer_i] 和 {alloc,load,decr}[layer_{i+1}] 可重叠

// Priority-based Greedy Scheduler
ready_queue = PriorityQueue()
while has_pending_operators():
    ready_queue.update()  // 加入依赖已满足的operators
    if ready_queue.has(COMPUTE):
        op = ready_queue.pop_highest_priority()  // 优先computation
    else:
        earliest_comp = find_earliest_computation_operator()
        op = get_associated_restoration(earliest_comp)
    execute(op)

// Preemptive micro-operator: alloc/decrypt切为~64KB micro-ops
// computation就绪时抢占当前micro-op，减少pipeline bubble
```

具体执行时序（Llama-3-8B 512-token prompt）：
```
Layer0: [Alloc0][Load0][Decrypt0]======[Compute0]==========
Layer1:          [Alloc1][Load1][Decrypt1]======[Compute1]==
Layer2:                   [Alloc2][Load2][Decrypt2]=========
```
Restoration latency被隐藏到computation latency下，strawman中串行的~11.6s restoration overhead被pipeline化。

关键调度策略：
1. **Priority Policy**：CPU computation operator优先级最高（避免NPU/CPU idle），其次与earliest computation关联的restoration
2. **Preemptive Scheduling**：large alloc/decrypt切为~64KB micro-operator，computation就绪时抢占。实验：preemptive进一步降低TTFT最多16.2%
3. **Partial Parameter Caching**：REE memory pressure允许时缓存早期prefill参数（按topological order），按reverse order释放。cache比例阈值前TTFT近似线性下降

Pipeline-aware extend/shrink：利用LLM参数first-in-last-out模式保证TZASC连续物理内存。性能：pipeline scheduling距critical path lower bound仅0.01%-9.9%，vs strawman TTFT降低77.1%-91.1%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

TZ-LLM基于llama.cpp实现（~1.2K LoC扩展）：提取computation graph拓扑顺序→插入restoration operator→priority queue scheduler（C++ std::priority_queue）+ preemptive micro-operator机制。CMA allocation通过REE Linux CMA API，flash I/O通过REE异步IO，decryption通过OpenSSL AES。依赖LLM参数访问顺序确定性（MoE/early-exit可能预取未使用参数）。开源：Zenodo artifact DOI 10.5281/zenodo.17213486。

涉及论文标题：
- TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone

