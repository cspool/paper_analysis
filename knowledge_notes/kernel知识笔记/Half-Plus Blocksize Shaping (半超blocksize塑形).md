## Half-Plus Blocksize Shaping (半超blocksize塑形)

术语是什么？

Half-Plus Blocksize Shaping（半超blocksize塑形）是μShare的核心技术，通过将CUDA kernel的blocksize参数设置为略大于SM总thread容量一半的值，间接操控NVIDIA GPU硬件调度器的block placement决策，实现intra-SM scattered co-location。原理：当blocksize > SM_thread_capacity/2时，同一SM无法容纳两个该kernel的block（2×blocksize > SM_thread_capacity），迫使硬件scheduler将同kernel blocks散布到不同SM，剩余threads可分配给其他kernel的小block。"half-plus"中的"half"指SM thread capacity的一半（如A40: 1536/2=768），"plus"指在此基础上加一个小偏移α（最小为warp size=32，避免thread fragmentation），最终blocksize=800(A40)或704(A800)。

从kernel调度角度拆解术语：

Half-plus blocksize shaping的数学定义：

```
// GPU: A40 (1536 threads/SM, CUDA max blocksize=1024)
// half-plus blocksize b 的定义域：
//   {b | 768 < b ≤ 1024, b ≡ 0 (mod 32)}
//   最小: 768 + 32 = 800
//   可选: 800, 832, 864, 896, 928, 960, 992, 1024

// 选择策略：
//   slack s_k > 0: b = 800 (最小half-plus，减少thread waste)
//   slack s_k < 0: b = previous + 32（逐步增加加速kernel执行）

// GPU: A800/A100/H200 (2048 threads/SM)
// 1/3-plus blocksize b 的定义域：
//   {b | 683 < b ≤ 1024, b ≡ 0 (mod 32)}
//   最小: 704

// 调度的间接控制原理：
//   Condition for block scheduling to SM:
//     SM.available_threads >= b
//   
//   Half-plus guarantee:
//     2 × b_min > SM_thread_capacity
//     2 × 800 = 1600 > 1536  ✓  同kernel两个block不能在同一SM
//     剩余threads = 1536 - 800 = 736 < 800 → 不能放第三个half-plus block
//     但 736 ≥ 512 (default roll block) → 可以放小block
```

μShare的消融实验证明：
- half-plus shaping贡献最大：μShare w/o shape（无blocksize调整）→ throughput下降30.95%，SLO violation增加6.33%
- fixed blocksize (1024) vs dynamic half-plus：固定1024时throughput下降3.36%，因为无法根据slack调整α控制加速力度
- 动态调整的必要性源自static preset blocksize在co-location下不再optimal：roll kernel exclusive执行时最优512，co-locate with vectorized kernel时最优变为1024（1.98× improvement）

术语一般如何实现？如何使用？

实现步骤：
1. Offline profiling：确定SM thread capacity → 计算half-plus/1/3-plus阈值
2. Kernel interceptor通过mmap共享内存读取kernel launch slack（s_k = tLaunch - tIntercept）
3. Block shaper根据slack sign和kernel类型决定α：
   - GPUs with 1536 threads/SM (A40, RTX 4090, RTX 3080 Ti): b_min = 800, 最大1024
   - GPUs with 2048 threads/SM (A100, A800, H200): b_min = 704, 1/3-plus替代half-plus（因为max blocksize=1024 < 2048/2, 两个1024 block可stacked co-locate）
4. 将修改后的blocksize写入共享内存 → dlsym获取的原始cudaLaunchKernel被调用时使用新blocksize
5. α值遵循warp对齐原则（32倍数），避免thread资源碎片

涉及论文标题：
- μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

