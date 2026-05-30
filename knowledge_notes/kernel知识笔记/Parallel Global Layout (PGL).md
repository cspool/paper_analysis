## Parallel Global Layout (PGL)

术语是什么？
Parallel Global Layout (PGL)是ParallelKittens的多GPU核心数据结构，表示所有参与GPU上具有相同shape/size的HBM内存区域集合。PGL封装multicast memory（VMM创建）和IPC-exported memory的统一寻址，以tile-indexed坐标(int4: {batch, depth, row, col})访问。PGL是PK 8种原语中P2P通信（store_async, store_add_async）和in-network collective（reduce, all_reduce）的操作目标。

从kernel调度角度拆解术语：
PGL操作示例：
```
// tile坐标: int4{b, d, r, c}
store_async(dst_PGL, src_stile, coord)           // TMA异步存储到multicast memory
store_add_async(dst_PGL, src_stile, coord)       // TMA原子加(实现reduce-scatter)
reduce<ROW, COL, OP::ADD>(dst_local, d_coord, src_PGL, s_coord) // in-network reduction local
all_reduce<ROW, COL, OP::ADD>(PGL, coord)        // in-network all-reduce
```
address duality: local address（写到本GPU物理HBM）vs multicast address（写到multicast object → NVSwitch broadcast；读+multimem.ld_reduce → in-network reduction）。PGL自动处理coalesced NVLink access和tensor core-friendly swizzle。

术语一般如何实现？如何使用？
PK utility层在kernel启动前完成VMM分配+fd交换+multicast create+bind+map，kernel内仅通过coord寻址PGL。

涉及论文标题：
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels

---
