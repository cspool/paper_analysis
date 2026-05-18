## Bubble-free DP Block Selection（无气泡动态规划块选择）

术语是什么？通过联网搜索让回答具体和精准。

Bubble-free DP Block Selection是FlashPS提出的用于扩散模型mask-aware serving pipeline的运行时决策算法。在FlashPS中，每个transformer block可以选择"加载cached unmasked activation + 仅计算masked tokens"或"全量计算所有tokens"。仅使用cache的问题在于cache loading（从host memory通过PCIe到GPU HBM）可能比直接全量计算更慢，导致GPU computation stream等待cache load stream，产生pipeline bubble。Bubble-free DP通过动态规划在O(N)时间内为N个transformer block决定每个block的最优策略：比较每个block在"加载cache后计算masked tokens"的完成时间与"全量计算"的完成时间，并考虑相邻block间的pipeline overlap，选择总latency最小且无bubble的块级执行方案。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Bubble-free DP的伪代码：

```
输入: N个transformer blocks, mask_ratio m, 每个block的:
      compute_full[i] - block i全量计算的GPU time
      compute_masked[i] - block i仅计算masked tokens的GPU time (≈m * compute_full[i])
      load_cache[i] - block i从host memory加载cached Y到HBM的PCIe time
输出: use_cache[1..N] - 每个block是否使用cache

// DP状态定义
dp[i][0] = min total time for blocks 1..i when block i uses FULL compute
dp[i][1] = min total time for blocks 1..i when block i uses CACHE

// 初始化
dp[0][0] = dp[0][1] = 0

// DP递推
for i = 1 to N:
    // Case A: block i用全量计算
    // 前一个block可以是任何状态，计算时间独立
    dp[i][0] = max(dp[i-1][0], dp[i-1][1]) + compute_full[i]
    
    // Case B: block i用cache
    // 需要考虑cache loading与上一个block计算的重叠
    // block i的cache loading可以与block i-1的计算重叠
    if i == 1:
        dp[i][1] = max(load_cache[i], compute_masked[i])  // 第一个block无前置重叠
    else:
        // cache loading可以与前置block的计算并行
        dp[i][1] = max(
            dp[i-1][0] + load_cache[i],                    // loading在prev block完成后才开始
            max(dp[i-1][0], 0) + load_cache[i]             // 重叠: loading可与prev compute并行
        )
        // 取min后再加masked compute time（与loading串行）
        dp[i][1] = max(dp[i][1], load_cache[i]) + compute_masked[i]

// 回溯
use_cache[N] = argmin(dp[N][0], dp[N][1])
for i = N-1 downto 1:
    use_cache[i] = backtrack(dp, use_cache[i+1])

return use_cache[1..N]
```

核心洞察：
1. **Bubble产生条件**：当`load_cache[i] > compute_full[i]`时，GPU computation stream将等待cache load完成才继续，产生bubble。
2. **DP的作用**：并非所有block都值得用cache。对于compute-intensive block（如attention），全量计算可能很快，cache loading反而成为瓶颈。DP确保只在"loading+masked计算 < 全量计算"的block使用cache。
3. **Pipeline overlap建模**：DP状态转移中，block i的cache loading可与block i-1的计算重叠。DP选取总completion time最小的策略，等同于最小化bubble时间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现细节：
1. **离线profiling**：对每个block，在目标GPU上profiling得到`compute_full[i]`、`compute_masked[i]`（作为mask ratio m的函数）、`load_cache[i]`（作为unmasked token数的函数）。FlashPS用线性模型估算。
2. **在线DP执行**：每个请求到达时，scheduler根据其mask ratio m执行O(N) DP（N通常为~20-30个transformer blocks for SDXL/Flux），DP耗时可忽略（微秒级）。
3. **CUDA Stream编排**：DP输出`use_cache[i]`后，对use_cache[i]=true的block，提前在cache load stream上发起`cudaMemcpyAsync`；computation stream在完成block i-1后直接对masked tokens执行attention/FFN。两个stream间的同步通过CUDA event实现——computation stream在block合并点等待cache load stream完成。
4. **与传统kernel fusion的区别**：Bubble-free DP不是kernel fusion——它不合并kernel，而是在更高层级（transformer block）选择执行路径。它是pipeline级别的调度优化。

涉及论文标题：
- FlashPS: Efficient Generative Image Editing with Mask-aware Caching and Scheduling

