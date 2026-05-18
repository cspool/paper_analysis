## Lazy Release Consistency (LRC) for GPUs / GPU惰性释放一致性

术语是什么？

Lazy Release Consistency (LRC) 是一种放宽的内存一致性模型，最初在分布式共享内存系统中提出，后由hLRC[1]扩展到GPU。LRC的核心思想是：在release操作时不立即将dirty data传播给所有可能读取它的线程/处理器，而是延迟coherence action，直到另一个处理器/线程实际需要该数据时（即acquire时或ownership转移时）才执行。与传统GPU的"eager" coherence（每次acquire都invalidate本地cache、每次release都flush dirty data）相比，LRC利用同步变量访问的locality减少冗余的cache invalidate/flush操作。在multi-chiplet GPU中，LRM-GPU将LRC应用到跨chiplet同步路径上：将每个同步变量关联到last accessed chiplet（owner），仅在owner跨chiplet迁移时才触发L1.5 cache的invalidate/flush coherence action，同chiplet内的连续acquire/release不触发额外cache操作。

从硬件架构角度拆解：

LRM-GPU中LRC的硬件执行流程（以lock-based synchronization + 4-chiplet GPU为例）：

1. Directory结构：LLC中维护sync-val directory，每entry记录：valid bit (1 bit)、tag address (48 bits)、owner chiplet ID (2 bits, 支持4-chiplet系统)。总共64 entries (0.4 KB)。

2. Acquire同步操作四种场景：
   - **Invalid（无记录）**: Directory无此同步变量记录且有free entry → 分配entry、记录owner=当前chiplet → 从LLC读取同步变量 → invalidate本地L1.5确保后续load读全局最新数据。
   - **Local chiplet（owner=本地）**: Directory命中且owner=当前chiplet → 直接从LLC读取同步变量 → 不触发L1.5 coherence action（最新数据仍在本地L1.5中，同chiplet内连续同步零cost）。
   - **Remote chiplet（owner=远端）**: Directory命中但owner≠当前chiplet → 若L1.5 write-back则flush远端owner的L1.5到LLC → 更新owner=当前chiplet → 从LLC读取同步变量 → invalidate当前chiplet的L1.5。
   - **Evicted（directory满）**: Directory满且无匹配entry → LRU evict一个entry → 若被驱逐entry的owner L1.5为write-back则flush → 分配新entry → 处理同Invalid场景。

3. Release同步操作四种场景：
   - **Invalid**: 分配entry → 将同步变量写入LLC → invalidate本地L1.5。
   - **Local chiplet**: 仅将同步变量写入LLC → 不flush/invalidate L1.5（延迟coherence action到owner真正变化）。
   - **Remote chiplet**: flush远端L1.5（若write-back）→ 更新owner → 写入同步变量到LLC → invalidate本地L1.5。
   - **Evicted**: flush被驱逐owner的L1.5 → 分配entry → 写入同步变量到LLC → invalidate本地L1.5。

4. 延迟收益示例：SM0→SM1（同chiplet0）→SM2（chiplet1）顺序执行的lock程序中，MCM-GPU需要3次L1.5 coherence操作（每次acquire）；LRM-GPU仅2次（SM0首次acquire的invalidate + SM2跨chiplet acquire时flush chiplet0 L1.5 + invalidate chiplet1 L1.5），SM1的acquire因owner未变而零cost。相比MCM-GPU平均减少30% L1.5 cache invalidations。

术语一般如何实现？如何使用？

LRC for GPUs的关键实现要素：(1) 同步变量ownership追踪——需要一个directory结构记录每个同步变量的当前owner（可以是SM、SM cluster或chiplet级别），directory可以位于LLC（如LRM-GPU）、专用的sync controller或CP（Command Processor）中。LRM-GPU的directory仅64 entries/0.4 KB（仅追踪同步变量，不追踪所有数据），远小于完整cache coherence directory（如HMG的12K entries/chiplet）。(2) Coherence action触发策略——采用"ownership变化驱动"而非"操作驱动"，即不是每次同步操作都触发coherence action，而是仅在同步变量owner迁移时触发。(3) Cache write policy选择——L1.5 write-through时release操作可省略flush但acquire必须invalidate；L1.5 write-back时release可延迟flush但acquire需负责flush远端dirty L1.5。(4) 与memory consistency model的兼容——LRC需要保证在同步变量owner迁移时所有之前写入对该同步变量的后续访问可见，满足acquire/release语义的happens-before要求。对于multi-chiplet场景，LRC的收益取决于跨chiplet同步locality：同chiplet内连续同步越多（如distributed CTA scheduling将相关线程组映射到同一chiplet），LRC节省的coherence操作越多；chiplet数增多时跨chiplet同步频率上升，LRC收益递减（4→8 chiplet: 1.33×→1.21×加速）。

涉及论文标题：
- LRM-GPU: Alleviating Synchronization Overhead for Multi-Chiplet GPU Architecture

