## Intra-SM Overlapping (GPU Compute-Communication Fusion)

术语是什么？
Intra-SM overlapping在同一SM内并发执行compute和inter-GPU通信：利用TMA单线程异步特性，一条线程发出TMA通信指令后立即返回，其余warp同时执行tensor core WGMMA指令。所有SM的所有tensor core保持繁忙，通信在后台进行。ParallelKittens推导出BF16 GEMM+RS on H100的完全隐藏条件：K ≥ sR/(2B) ≈ 2197，实测K=4096时non-overlapped communication <1%。

从kernel调度角度拆解术语：
GEMM+RS时间线（intra-SM，单SM视角）：
```
Warp0(loader,1 thread):  | TMA load A_tile | wait | TMA load next | ...
Warp1-3(consumer):       | WGMMA C+=A×B     | WGMMA | WGMMA ...
Warp4(storer,1 thread):  | TMA store_add_async to peer | wait | ...
                           ↑ TMA不占用tensor core，单线程异步
```
优势：所有tensor core满利用率；mbarrier同步仅~64ns延迟。局限：(1) 通信跟随计算数据流，无法利用in-network reduction；(2) 对remote cache unfriendly场景（Ring Attention），重复remote访问快速饱和NVLink带宽。

术语一般如何实现？如何使用？
PK: TMA store_async + store_add_async原子加到peer PGL。适用：(a) 通信模式与计算模式对齐（GEMM+RS——每个tile写入唯一目标）；(b) 通信量可被计算覆盖的大K场景（K>2197）；(c) 不需要in-network acceleration的P2P传输。

涉及论文标题：
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels

---
