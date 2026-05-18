## Asynchronous Tile Transfer (ATT / 异步瓦片传输)

术语是什么？通过联网搜索让回答具体和精准。

异步瓦片传输（ATT）是GPU中将多维数据"tile"直接在global memory和on-chip scratchpad（如LDS/shared memory）之间进行批量异步搬运的硬件机制。与传统的同步load/store指令（以cache-line粒度操作，经寄存器中转，需大量寄存器追踪依赖）不同，ATT允许程序员指定multidimensional tile的整体属性→ATT硬件自行管理地址生成、stride计算和边界条件→数据以bulk方式从global memory直接写入scratchpad，不经过寄存器、不占用issue slot、不产生传统data dependency tracking开销。ATT的核心优势是使能fine-grained overlap of data movement with computation——在compute-bound kernel中将idle cycles转化为有用工作，提升memory-bound workload的utilization。QuCo论文将ATT视为通用机制：NVIDIA TMA是ATT的state-of-the-art商业实现，但ATT概念同样适用于任意支持异步global-to-shared memory transfer的GPU。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

ATT硬件在GPU Compute Unit (CU) 内部与L1 cache解耦，直接连接到global memory subsystem：
```
GPU Compute Unit (CU) 内部：
  ┌────────────────────────────────────────┐
  │  Wavefronts (producer)                 │
  │    ↓ 单线程发起 ATT copy descriptor     │
  │  ┌──────────┐                          │
  │  │ ATT Unit │ ←→ Global Memory (DRAM)  │
  │  └────┬─────┘    (bypass L1 cache)     │
  │       ↓ 直接写入 LDS                   │
  │  ┌──────────┐                          │
  │  │   LDS    │ ←→ Wavefronts (consumer) │
  │  │(scratch- │    通过 Peek/Pop 消费     │
  │  │  pad)    │                          │
  │  └──────────┘                          │
  │  ┌──────────┐                          │
  │  │ Sync Unit│ async barriers           │
  │  └──────────┘ (arrive/wait)            │
  └────────────────────────────────────────┘
```
运转流程（以QuCo论文的Operand Queue为例）：
1. **Descriptor setup**：Host/kernel初始化ATT copy descriptor——包含global memory base address、tile dimensions（x, y）、element size、memory strides、LDS destination address
2. **Producer trigger**：Dedicated producer wavefront中的单线程执行Push(tile_idx)→ATT hardware读取descriptor→自动生成地址序列（含stride jumps）→直接从global memory读取数据→写入LDS
3. **Synchronization**：Producer wavefront执行non-blocking arrive on async barrier→继续独立工作（不stall）→Consumer wavefronts需要数据时执行wait→barrier在expected bytes到达后自动trip→consumers通过Peek/Pop消费LDS中tile
4. **Overlap**：同一时刻，ATT engine在后台搬运下一tile，consumer wavefronts在前台compute当前tile

QuCo论文在MGPUSim模拟器中实现了architecture-neutral的ATT模型，支持R9 Nano (GCN3)、MI-100和Radeon 530三种GPU的ATT评估。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

NVIDIA TMA是目前唯一的商业ATT实现，在CUDA中通过cuda::pipeline API使用。AMD GPU目前无ATT硬件（QuCo在MGPUSim上建模了假设的AMD ATT）。一般ATT实现需要：
1. **硬件侧**：Compute Unit内集成ATT engine（地址生成器 + bulk transfer controller），bypass L1直接读写global memory，支持multidimensional tile descriptor解析
2. **软件侧**：提供producer-consumer queue抽象（如NVIDIA cuda::pipeline）封装ATT descriptor管理；提供async barrier（如NVIDIA mbarrier）支持arrive/wait两阶段同步
3. **编程限制**：仅适用于strided access pattern（能用规则tensor坐标描述的数据访问），不适用于随机访问；需预先在LDS中分配tile buffer空间；tile大小受LDS容量物理约束

涉及论文标题：
- QuCo: Efficient and Flexible Hardware-Driven Automatic Configuration of Tile Transfers in GPUs

---

