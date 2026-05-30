## mbarrier (CUDA Asynchronous Memory Barrier)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
mbarrier (asynchronous memory barrier) 是 NVIDIA Hopper (CC 9.0+) 引入的 CUDA 同步原语，是一种基于 shared memory 的硬件 barrier 用于在 thread block cluster 内协调多个 CTA 之间的同步。关键能力包括：(1) transaction tracking——barrier 追踪 expected transaction bytes，直到所有 TMA 数据到达才释放；(2) scope_cluster——同步范围可扩展到整个 cluster 内所有 CTA（vs __syncthreads 仅 block 内，cooperative_groups::grid_group 需 global memory）；(3) many-to-many synchronization——仅需参与同步的 CTA 子集 arrive/wait，不需全 cluster 同步（vs CUTLASS 默认 all-to-one cluster-sync）；(4) phase parity——通过 parity-based wait 实现高效的多次复用。

从kernel调度角度拆解术语：
FlashFuser 中 mbarrier 用于控制 dsm_comm 原语的同步：

```
// Prologue: 初始化 mbarrier
__shared__ uint64_t mbar_shuffle[NUM_GROUPS];
if (is_leader) {
  for g in 0..NUM_GROUPS:
    mbarrier.init.shared.b64(&mbar_shuffle[g], expected_arrivals[g]);
}

// GEMM1 mainloop: dsm_shuffle with mbarrier
// Producer CTA: send C tile slice to consumer
producer:
  write C_slice to consumer's SMEM via TMA shared::cluster;
  mbarrier.arrive.expect_tx(&mbar_shuffle[group_id], tx_bytes);
  // arrive + signal expected transaction bytes

// Consumer CTA: receive C tile slice from producer
consumer:
  mbarrier.try_wait.parity.shared::cta.b64(&mbar_shuffle[group_id], phase);
  // wait until all producers arrived AND all TMA data written
  read C_slice from SMEM;
  // compute E partial with received C tile
```

区别于 CUTLASS 默认的 all-to-one cluster-sync（需要所有 CTA 都到达才能继续），mbarrier 的 many-to-many 模式允许 FlashFuser 在同一 cluster 内独立同步不同的 shuffle group 和 reduce group。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUDA API：(1) `cuda::ptx::mbarrier_init` 初始化 barrier；(2) `cuda::ptx::mbarrier_arrive` 非阻塞 arrive；(3) `cuda::ptx::mbarrier_arrive_expect_tx` arrive 并声明期望的 TMA 传输字节数；(4) `cuda::ptx::mbarrier_try_wait_parity` parity-based wait 直到所有 arrive 完成。使用限制：(1) H100+ only；(2) 需要显式管理 phase parity 以避免 ABA 问题；(3) shared memory 128B 对齐。

涉及论文标题：
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection
