## Inter-SM Overlapping (Multi-GPU Kernel Scheduling)

术语是什么？
Inter-SM overlapping将GPU的SM分区为两组：compute SM执行HBM→SMEM→tensor core计算流水线，communication SM独立执行inter-GPU数据传输或collective操作。与intra-SM（同一SM内分warp）不同，inter-SM让通信独立于计算数据流，使in-network reduction和remote cache-friendly批量传输成为可能。ParallelKittens通过LCSC template的num_comm_sms参数控制SM分区。

从kernel调度角度拆解术语：
GEMM+AR fused kernel的inter-SM执行：
```
Compute SMs (num_SMs - num_comm_sms):
  loader:   tma::load_async A, B tiles → SMEM (pipelined)
  consumer: warpgroup::mma_AB accumulate C
  storer:   tma::store_async to local HBM → signal barrier

Communication SMs (num_comm_sms):
  wait barrier(NUM_DEVICES)           // 所有GPU local compute完成
  __syncthreads()
  all_reduce<ADD>(G.C, coord)         // NVSwitch in-network reduction via multimem.ld_reduce
```
trade-off: inter-SM牺牲部分SM做通信（tensor core利用率降低），但：(1) 利用in-network reduction将all-reduce通信量从O(N) peer写入降为O(1) multicast读归约，GEMM+AR加速3.62x；(2) Ring Attention中通信SM批量传输KV block到local HBM复用L2 cache，避免per-block重复remote访问；(3) HBM同步延迟~832ns vs intra-SM mbarrier ~64ns。最优num_comm_sms与问题大小相关，PK运行时自动搜索。

术语一般如何实现？如何使用？
PK: lcsc::launch_kernel + num_comm_sms配置。适用：(a) 利用in-network reduction的all-reduce/collective；(b) 需要remote cache reuse的Ring Attention；(c) 通信模式与计算模式不对齐的场景。

涉及论文标题：
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels

---
