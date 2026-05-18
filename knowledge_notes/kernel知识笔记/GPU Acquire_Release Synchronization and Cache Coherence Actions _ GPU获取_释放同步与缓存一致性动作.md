## GPU Acquire/Release Synchronization and Cache Coherence Actions / GPU获取/释放同步与缓存一致性动作

术语是什么？

GPU的Acquire/Release同步是GPU编程模型中实现线程间同步和内存一致性的核心机制。在CUDA等GPU编程模型中，acquire（获取）和release（释放）操作标记了同步边界：acquire表示后续内存访问必须看到release之前的所有写入；release表示之前的所有内存写入对后续acquire可见。在硬件层面，传统GPU（monolithic）实现acquire/release同步的方式是：acquire操作invalidate所有本地私有cache（L1 cache），确保后续load从全局共享的LLC读取最新数据；release操作flush所有dirty data从本地cache写回LLC，确保之前的所有store对后续acquire可见。Atomic同步操作（atomicCAS、atomicAdd等）通常绕过本地cache直接在LLC中执行，以确保原子性和一致性。

在多chiplet GPU中，由于额外的cache层级（L1.5 cache），acquire/release的coherence action变得更加昂贵：(1) Acquire不仅需要invalidate L1，还需要invalidate整个L1.5 cache（更大容量、更多way、影响chiplet内所有SM），开销远大于单L1 invalidation；(2) Release若L1.5为write-back policy需flush L1.5 dirty data到LLC，若write-through则无flush开销但acquire仍需invalidate；(3) 跨chiplet atomic同步受inter-chiplet有限带宽约束，多个SM对同一地址的atomic操作（如spin lock的atomicCAS重试）产生大量跨chiplet流量。

从kernel调度角度拆解：

GPU同步操作的硬件执行流程（以lock-based synchronization + CUDA kernel为例）：

```
// CUDA kernel 伪代码
__global__ void kernel_with_lock() {
    // ... local computation ...
    
    // ACQUIRE phase: spin until lock acquired
    while (atomicCAS(&lock, 0, 1) != 0);
    __threadfence();  // acquire fence: invalidate local caches
    // Hardware action (MCM-GPU): invalidate L1 + L1.5 cache
    // Hardware action (LRM-GPU): query sync-val directory;
    //   if owner=local chiplet: skip L1.5 invalidation;
    //   if owner=remote: flush remote L1.5 + invalidate local L1.5
    
    // CRITICAL SECTION: access shared data
    val = shared_data;       // load (must see latest value)
    shared_data = new_val;   // store
    
    // RELEASE phase: release lock
    __threadfence();  // release fence: ensure stores visible
    atomicExch(&lock, 0);
    // Hardware action (MCM-GPU w/ write-through L1.5): no flush needed
    // Hardware action (LRM-GPU): write sync var to LLC, delay L1.5 actions
}
```

同步操作的coherence action组合（以4-chiplet GPU, write-through L1/L1.5为例）：
- **MCM-GPU acquire**: invalidate L1 (per-SM) + invalidate L1.5 (per-chiplet, affects ALL SMs!)
- **MCM-GPU release**: write-through L1/L1.5 → LLC is up-to-date, no flush
- **MCM-GPU atomic**: bypass L1/L1.5 → route to LLC → cross-chiplet via inter-chiplet network
- **LRM-GPU acquire**: query directory → local owner: LLC read only; remote owner: flush old L1.5 + update owner + invalidate new L1.5
- **LRM-GPU release**: write sync var to LLC → local owner: no extra actions; remote: flush L1.5 + update owner + invalidate L1.5
- **LRM-GPU atomic**: AMU in-network merge → combined request sent → multicast response

术语一般如何实现？如何使用？

GPU acquire/release同步的实现要素：(1) Memory fence指令——CUDA中`__threadfence()`（global fence）、`__threadfence_block()`（block-level fence），PTX中对应`membar.cta`（CTA scope）和`membar.gl`（GPU scope）指令。这些指令在硬件层面触发cache coherence actions。(2) Hardware coherence actions——传统GPU通常采用software-driven coherence而非硬件自动coherence protocol（如CPU的MESI）。acquire/release通过fence指令显式触发硬件invalidate/flush。(3) 同步原语——GPU支持多种同步模式：barrier (`__syncthreads`用于block内同步，global barrier需atomic+spin实现)、lock（atomicCAS实现spin lock）、semaphore（atomicAdd/Sub实现计数信号量）、atomic update（atomicAdd更新共享数据结构如histogram的bin）。每种模式对acquire/release coherence action的需求不同。(4) Multi-chiplet差异化——额外cache level迫使同步机制考虑多级cache incoherence问题。HMG使用完整coherence protocol（类似VI protocol）维护所有cache line状态；hLRC缓存同步变量在多级cache中但引入跨SM同步变量write-back等待和重试；LRM-GPU采用折中：同步变量不缓存但引入lightweight owner tracking减少L1.5 coherence action（仅在跨chiplet ownership迁移时触发）。

涉及论文标题：
- LRM-GPU: Alleviating Synchronization Overhead for Multi-Chiplet GPU Architecture

