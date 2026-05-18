## LRM-GPU: Alleviating Synchronization Overhead for Multi-Chiplet GPU Architecture

- baseline方法是什么？
  Baseline是MCM-GPU[2]的同步机制，沿用传统monolithic GPU的软件驱动coherence protocol。同步变量不被本地cache维护一致副本，acquire操作invalidate L1和L1.5 cache，release操作确保dirty data写回LLC，atomic同步操作绕过本地cache直接到LLC执行。MCM-GPU同时采用L1.5 cache（仅缓存remote data）、first-touch page allocation和distributed CTA scheduling来优化普通数据访问的NUMA问题，但这些优化不解决同步瓶颈。

  全栈执行例子（以lock-based synchronization + MCM-GPU + 4-chiplet GPU为例）：
  - 算法层：CUDA kernel使用atomicCAS实现spin lock，多个SM竞争同一lock，acquire/release包围临界区访问共享数据A。
  - 系统框架/Serving层：论文未明确说明（GPGPU-Sim直接运行CUDA kernel，无Serving框架层）。
  - 编译框架层：论文未明确说明（CUDA 11.1, O3编译，无定制编译pass）。
  - kernel调度层：SM0 (chiplet0) 发出atomicCAS(lock)→绕过L1/L1.5直接路由到LLC执行→成功acquire→SM0发出acquire同步→invalidate chiplet0 L1.5（保守地清掉可能stale的数据）→执行load/store修改共享数据A→release同步→write-through L1.5保证dirty data已写回→SM1 (chiplet0) acquire→再次invalidate L1.5（即使连续同步在同一个chiplet内，L1.5仍被重复清空）→SM2 (chiplet1) 的atomicCAS(lock)→跨chiplet发送到LLC→受inter-chiplet 768GB/s有限带宽和网络排队影响。每次acquire/release都触发L1.5 invalidate/flush（即使大部分同步变量owner未变），多SM对同一lock的atomicCAS spin产生大量跨chiplet重试流量。
  - 硬件架构层：MCM-GPU的coherence行为——L1/L1.5 write-through，LLC write-back。Acquire: invalidate L1+L1.5。Release: write-through保证LLC不持有stale data，无需额外flush。Atomic同步: bypass L1/L1.5直接到LLC执行。该方案在4-chiplet系统上与等价monolithic GPU对比平均性能下降50.5%（其中额外L1.5 invalidation导致22.5%损失，remote atomic access导致23.5%损失）。
  - 芯片设计层：4-chiplet GPU，每chiplet 64 SMs + 2MB L1.5 + LLC slice + DRAM partition，inter-chiplet concentrated hierarchical crossbar 768GB/s，DRAM 64 channels/3TB/s。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出LRM-GPU，核心洞察是利用同步行为的locality：(1) 同步变量ownership locality——当同步操作在一个chiplet内发生时，后续同步操作很可能也在同一chiplet；(2) 同步atomic数据locality——跨chiplet的atomic操作可能同时访问同一地址，可在网络内合并。基于此，LRM-GPU实现两个机制：

  **缺陷1：MCM-GPU每次acquire/release都保守地invalidate/flush L1.5，即使连续同步发生在同一chiplet也重复付费**
  → LRM-GPU引入Lazy Release Consistency：在LLC实现sync-val directory（64 entries, 0.4KB），为每个同步变量记录owner chiplet。Acquire时：
  - 若directory无记录→分配entry+记录owner+invalidate本地L1.5
  - 若owner=本地chiplet→直接从LLC读同步变量，不做coherence action（延迟到owner真正变化）
  - 若owner=远端chiplet→flush远端L1.5+更新owner+invalidate本地L1.5
  - 若directory满→LRU evict+flush被驱逐owner的L1.5
  Release时类似：local owner仅写同步变量到LLC，延迟coherence action。这利用chiplet内连续同步的时间locality减少L1.5 invalidate/flush。相比MCM-GPU减少30% L1.5 cache invalidation。

  **缺陷2：跨chiplet atomic同步操作受inter-chiplet有限带宽严重制约，大量同地址atomicCAS spin或atomicAdd update产生冗余跨chiplet流量**
  → LRM-GPU在每个chiplet网络中嵌入AMU (Synchronization Atomic Merge Unit)，检测并合并同地址cross-chiplet atomic请求。多个atomicAdd(addr,1)合并为atomicAdd(addr,n)；多个对同一lock的atomicCAS在比较值相同时只发送一个可能成功的请求。AMU的merge table (2K entries, 16 banks, CAM+SRAM dual-port) 通过countdown timer控制合并窗口，到期或SM list满后发送合并请求，响应通过multicast unit广播给所有参与SM。相对MCM-GPU减少28% inter-chiplet traffic（AMU单独贡献12%）。相对HMG减少52% inter-chiplet traffic。AMU总功耗301.44mW/面积1.84mm² (40nm)，仅占系统0.13%能耗。

  **缺陷3：相比hLRC（多级cache追踪同步变量+write-back），同步变量跨SM迁移时需写回远端cache，高竞争下增加等待时间和重试流量**
  → LRM-GPU不同：同步变量不被缓存在L1/L1.5中（与传统GPU一致直接bypass到LLC），避免跨SM迁移时的多级cache write-back等待；仅在有跨chiplet ownership变化时才对L1.5做coherence action，不跟踪每个SM的注册状态。hLRC虽减少56% L1.5 invalidation，但因同步变量write-back等待和重试导致性能反而不如MCM-GPU。

  **缺陷4：相比HMG（完整cache coherence protocol + hierarchical sharer tracking），所有写入数据需向sharer发送invalidation request，在atomic密集workload上产生大量write-invalidation traffic**
  → LRM-GPU只跟踪同步变量的owner chiplet（而非所有数据的sharer），directory仅64 entries/0.4KB（HMG 12K entries/chiplet），状态空间小得多。在MST和pagerank等atomic密集benchmark上，HMG因大量write-invalidation performance degradation，而LRM-GPU无此问题。相对HMG平均加速1.22×，减少52% inter-chiplet traffic，减少32% energy。

  论文方法全栈执行例子（以lock-based synchronization + LRM-GPU + 4-chiplet GPU为例）：
  - 算法层：CUDA kernel使用atomicCAS实现spin lock，SM0→SM1 (chiplet0) →SM2 (chiplet1) 顺序执行。与baseline相同的应用代码，差异全在硬件同步路径。
  - 系统框架/Serving层：论文未明确说明。
  - 编译框架层：论文未明确说明（CUDA 11.1, O3编译，无定制编译pass）。
  - kernel调度/硬件层：SM0 acquire lock→directory无X→分配entry+owner=chiplet0+invalidate L1.5→SM0修改A=1后release→仅写X到LLC不flush L1.5（owner未变）→SM1在同chiplet0 acquire→directory命中owner=chiplet0→不invalidate L1.5直接读X+访问最新A→SM1修改A=2后release→仅写X→SM2 (chiplet1) acquire→directory发现owner=chiplet0→flush chiplet0 L1.5（write back A=2到LLC）→更新owner=chiplet1→invalidate chiplet1 L1.5→SM2读最新A=2→SM2 release。跨chiplet atomic路径：SM0/SM1同时发atomicAdd(addr,1)→AMU merge table合并→单一atomicAdd(addr,2)跨chiplet发送→响应广播。关键差异：MCM-GPU在SM0和SM1的acquire时各invalidate一次L1.5（共2次），LRM-GPU仅第一次invalidate 1次；MCM-GPU的atomic请求每次单独跨chiplet，LRM-GPU先merge再发送。
  - 硬件架构层：sync-val directory在LLC中——64 entries, 2-bit owner/chiplet, 48-bit tag address, 1-bit valid→每entry 51 bits→总0.4KB。AMU per chiplet——16 channels, 2K merge entries/16 banks, CAM key查找+SRAM data存储 dual-port, countdown timer控制合并窗口。
  - 芯片设计层：multi-chiplet inter-chiplet network中每个chiplet嵌入AMU，merge table以TSMC 40nm定制电路实现。Chiplet数从4增至8时加速比从1.33×降至1.21×（locality下降）。Inter-chiplet latency 8-48 cycles范围内性能几乎不变（GPU warp switching隐藏延迟，性能对bandwidth敏感而非latency）。
