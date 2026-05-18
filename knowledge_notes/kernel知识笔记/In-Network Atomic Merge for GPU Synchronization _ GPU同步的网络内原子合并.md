## In-Network Atomic Merge for GPU Synchronization / GPU同步的网络内原子合并

术语是什么？

In-Network Atomic Merge是一种在网络（NoC/interconnect）内部对跨chiplet的同步atomic请求进行合并处理的技术，由LRM-GPU提出。传统GPU中，每个SM发出的atomic同步操作独立通过网络传输到LLC执行，当多个SM竞争同一同步变量（如spin lock的atomicCAS）或更新同一共享地址（如atomicAdd更新histogram bin），这些atomic请求在网络中产生大量冗余流量，尤其在跨chiplet场景下inter-chiplet有限带宽成为瓶颈。In-network atomic merge在网络中嵌入专用的Synchronization Atomic Merge Unit (AMU)，检测去往同一地址的多个atomic请求，根据可合并性将其合并为单一的aggregated request发送，响应返回时再通过multicast向所有参与SM广播结果。这种方法减少跨chiplet atomic传输次数和LLC atomic执行次数，缓解inter-chiplet bandwidth pressure。

从kernel调度角度拆解：

AMU对atomic同步操作的合并调度流程：

```
// AMU request processing pipeline
function AMU_Process_Atomic_Request(req):
    if not is_cross_chiplet(req.target_addr):
        forward_directly(req)  // local chiplet requests bypass AMU
        return
    
    // CAM lookup: match status=VALID + mergeable opcode + same address
    entry = merge_table.cam_lookup(req.opcode, req.addr)
    
    if entry and entry.status == VALID and can_merge(entry, req):
        // HIT: merge with existing entry
        entry.data = alu_merge(entry.opcode, entry.data, req.data)
        entry.sm_list.append(req.sm_id)
        if entry.sm_list.size >= MAX_SM_LIST:  // SM list full → send
            send_merged_request(entry)
            entry.status = RESERVE
        return
    
    if has_free_entry(merge_table):
        // MISS + free entry: allocate new entry, start timer
        new_entry = alloc(status=VALID, opcode=req.opcode,
                          addr=req.addr, sm_list=[req.sm_id], data=req.data)
        start_countdown_timer(new_entry)
        return
    
    // No free entry: bypass AMU, send directly
    forward_directly(req)

// Timer callback or SM list full
function send_merged_request(entry):
    send_to_llc(entry.opcode, entry.addr, entry.data)
    entry.status = RESERVE  // block further merging until response

// Response processing
function AMU_Process_Response(rsp):
    entry = merge_table.lookup_by_addr(rsp.addr)
    if entry and entry.status == RESERVE:
        // Multicast broadcast to all participating SMs
        for sm_id in entry.sm_list:
            forward_to_sm(sm_id, rsp)
        release_entry(entry)  // entry → INVALID, reusable
    else:
        forward_directly(rsp)  // pass-through for non-merged responses
```

支持的atomic类型及合并规则：
- **atomicAdd/Sub/Min/Max/And/Or/Xor**: 可自由合并（commutative/unordered），如 atomicAdd(a, 1) + atomicAdd(a, 1) → atomicAdd(a, 2)
- **atomicCAS**: 仅在comparison data相同时合并 → 选一个作为combined request，其余等待返回fail结果
- **Cross-cache-line**: 同一coarse-grained地址区域内不同offset的请求按operation-mask合并

术语一般如何实现？如何使用？

In-network atomic merge的实现要点：(1) 网络内处理 vs 端点处理——传统atomic合并方案（ARC[11]在warp内、LAB[10]在SM内的atomic buffer、Atomic Cache[54]在cache内实现atomic）在请求进入网络前合并，受限于SM/warp局部范围。In-network merge (AMU) 在网络中合并，能看到跨多个SM的atomic请求，合并机会显著更大。(2) 合并窗口设计——通过countdown timer或SM list阈值控制合并窗口，需要在合并机会（窗口长）和latency overhead（窗口短）之间trade off。论文在4-chiplet系统中AMU贡献1.16×加速和12% traffic reduction。(3) 正确性保证——可交换/无序的atomic（Add/Sub/And/Or/Xor/Min/Max）天然可合并且结果等价于串行执行；atomicCAS需要comparison data相同才合并（否则不同比较值的atomicCAS不应互相影响）。(4) 与cache coherence的交互——AMU在LLC之前合并请求，不改变LLC对atomic操作的执行语义；multicast broadcast根据SM list向所有参与SM返回结果。AMU与LRC互补：LRC减少cache invalidation overhead，AMU减少inter-chiplet atomic bandwidth。(5) Broadcast efficiency——一个合并请求的响应替代多个单独请求的响应，进一步减少inter-chiplet返回流量和LLC响应端口压力。

涉及论文标题：
- LRM-GPU: Alleviating Synchronization Overhead for Multi-Chiplet GPU Architecture

