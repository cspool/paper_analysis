# SA抢占Preemption

复杂度决定计算时间，调度决定延迟。

# baseline流程

其他baseline的抢占或计算流程：

GPU（虚拟化、抢占、SIMD）、

CGRA（配置动态）、

Acc（Dataflow、Reconfig、指令）、

MLX（Spatial指令定义）、

PISA（tile SoC）

为什么不支持抢占？

为什么开销大？

Eyeriss的内存访问控制不干净？抢占开销大？

简化地址生成，大地址字，简化控制流来减少抢占，依赖重排和flexible tiling解决使用率。

spatial的特点是核心的动态调度

## GPU

专利表述繁杂，学习总结按照图来解说。图没讲清楚的，文字中也难找到答案。

图解说完成后，写总结。

MID-Primitive：Primitive之中的抢占过程。

CGA：增加资源组织层次，将关系紧密的CTAs组织成CGA，映射到访存效率更高的资源组（DSMEM）。

large DATA Access（Conv）：SM加载张量（TMA）的指令格式，img2col mode load，traversalStride是访问pixel的位置间隔。

Config for Mult Sep Processor：GPU内多个VM或SMC划分的Ctx切换。

HW-Acc Sync、Fast Sync：不同SM运行相同程序的不同线程，进度不同，数据需要同步才能继续。paper设计SM之间通信机制（DSMEM）。

Programmable MultCast：TMA合并多个SM的LD请求，一次性读取L2-Cache后广播返回到多SM。

多任务需要共享内存，来处理数据依赖，让PE之间能传递数据。

大PE阵列Spatial Acc（tiled Acc）的设计是设计PE之间路由（类似**Adyna**），一般不存在Cache。

GPU/Node阵列等通用并行单元的设计是共享内存，多任务切换本身是延迟，并且出现Cache Miss降低性能。

# 我的设计

我的抢占流程和计算设计，优势是RTL设计的datapath，粗粒度指令。

dedicated Acc上多任务抢占的场景，不如通用多Node的PPU上的多任务抢占。

dedicated Acc组成**异构Node**时，需要设计**多Node/tile Acc**之间的数据传输（类似**Adyna**），但这样仍然要扩展通用算术指令，并且需要考虑**Mapping策略**。

可否？**通用Core+datapath核**，类似GPU的TensorCore+CudaCore设计。通用Core完成通用计算，datapath核完成特殊计算。

**DiT是最可能的延伸场景，因为VQVAE需要Conv算子**。

将**im2col融入LD的mode**，类似GPU的设置？

baseline：计算访存解耦控制或指令控制，我的是计算访存协同控制？只是datapath，如何包装？

硬件调度是固定的指令执行序列。

flexible tiling是因为tile不可配置，硬件定义一些并行形状，更灵活需要依赖指令的精细控制（CGRA）。

img2col+gemm-SA内没有多余指令，只有顶层粗粒度指令，缺点是僵硬固定的im2col+GEMM的数据流pipeline。

Spatial Acc怎么做的实验？单任务或算子序列的端到端模拟。

Sense、Spots、Feather：Im2COL+Conv的Acc。

# 实验设计

baseline和我的设计如何验证

metric根据baseline决定。