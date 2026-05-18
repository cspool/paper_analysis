## Multi-Chiplet GPU (MCM-GPU) Architecture / 多芯粒GPU架构

术语是什么？

Multi-Chiplet GPU（也称MCM-GPU）是一种将多个较小的GPU chiplet通过高密度封装（如interposer、硅桥）集成为单个逻辑GPU的架构。与传统的monolithic GPU（单个大die制造）不同，multi-chiplet GPU将计算资源（SM、cache、memory controller等）分布到多个物理die上，各chiplet之间通过inter-chiplet interconnect（如interposer上的NoC或crossbar）通信。每个chiplet包含多个SM及其私有L1 cache、一个chiplet内共享的cache层（如L1.5或L2 cache）、一个全局LLC slice和一个本地DRAM partition。所有chiplet的DRAM partition共同提供统一的全局内存地址空间（single logical GPU abstraction），对编程模型完全透明——CUDA kernel、数据结构和launch配置无需修改，差异仅在硬件和runtime如何partition数据和调度thread blocks。

从芯片设计角度拆解：

Multi-chiplet GPU的核心设计问题：(1) NUMA效应——由于inter-chiplet bandwidth远低于intra-chiplet bandwidth（通常相差数倍），跨chiplet的内存访问延迟显著高于本地chiplet内存访问，导致非均匀访存性能。缓解方法包括：增加chiplet内额外cache层（L1.5/L2）缓存remote data以利用数据局部性；采用first-touch page allocation将page映射到首次访问该page的chiplet的本地memory partition；使用distributed CTA scheduling将连续CTA组分配到同一chiplet以最大化CTA间数据局部性。(2) Inter-chiplet interconnect设计——需要权衡带宽、延迟和面积/功耗，常见拓扑包括concentrated hierarchical crossbar（各chiplet两两直连）、mesh、或基于interposer的NoC。(3) 同步开销放大——额外cache level使得同步操作（acquire/release）需要invalidate/flush更深cache层次；跨chiplet atomic同步操作受限于inter-chiplet有限带宽。(4) Cache hierarchy设计——每chiplet的cache level（memory-side LLC vs SM-side LLC）、容量分配、write policy（write-through vs write-back）和caching策略（是否仅缓存remote data）需要trade off数据局部性和coherence复杂度。

以4-chiplet GPU系统为例（LRM-GPU论文配置）：每chiplet 64 SMs + 128KB L1/SM (write-through, private) + 2MB L1.5 (write-through, shared within chiplet, 仅缓存remote data) + 全局8MB LLC (write-back, shared across all chiplets, 每LLC slice仅缓存本地DRAM data)。Inter-chiplet network为concentrated hierarchical crossbar，每chiplet与其他所有chiplet直连，768GB/s bandwidth，32 cycles/hop。DRAM 64 channels、3TB/s。First-touch page allocation (4KB pages) + distributed CTA scheduling。编程模型保持CUDA兼容，所有chiplet呈现为单个逻辑GPU device。

术语一般如何实现？如何使用？

Multi-chiplet GPU的实现涉及：Die partitioning——将GPU的SM、cache、memory controller等资源映射到多个物理die；Interposer/封装——使用硅interposer (CoWoS)、EMIB或类似技术提供高密度die-to-die连接；Inter-chiplet network——在interposer上构建NoC或crossbar连接各chiplet；Cache coherence——在必要时维护跨chiplet的cache一致性（通常GPUs采用更简单的software-driven coherence以避免硬件复杂度）；Memory mapping——通过page allocation policy将物理地址映射到各chiplet的本地DRAM partition；Runtime/CTA scheduling——通过硬件或runtime将thread blocks分配到各chiplet以优化数据局部性。实际产品中：AMD的MI250X/MI300X采用multi-die设计（MI300X有8个compute die）；NVIDIA的Blackwell B200将两个reticle-limited die通过NV-HBI (10 TB/s)连接为一个逻辑GPU。

涉及论文标题：
- LRM-GPU: Alleviating Synchronization Overhead for Multi-Chiplet GPU Architecture
