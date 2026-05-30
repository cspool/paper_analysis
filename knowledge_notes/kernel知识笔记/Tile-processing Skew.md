## Tile-processing Skew

术语是什么？
Tile-processing skew是Stream-K分解的一种副作用：当output tile数t不能被grid size g整除时，各CTA的起始k-offset不同。例如在384×384×128问题上g=4 CTA、BLK_K=4、每CTA 72个MAC iterations时，CTA_0起始k=0、CTA_1起始k=288、CTA_2起始k=576等。这种offset差异持续整个GEMM计算过程（persistent skew），可能导致不同CTA加载的k-axis fragments来自不同的k位置，从而阻止这些fragments在GPU L2 cache中被跨CTA复用。

从kernel调度角度拆解术语：
Tile-processing skew与cache reuse的关系：

```
384×384×128 GEMM, BLK_K=32, 4 CTA:
  CTA_0: iter [0, 72)   → k∈[0, 288)
  CTA_1: iter [72, 144) → k∈[288, 576) [部分覆盖tile 2-4]
  CTA_2: iter [144, 216)→ k∈[576, 864)
  CTA_3: iter [216, 288)→ k∈[864, 1152) [但k max=128 所以wraparound]

此时各CTA在相同相对进度下访问不同的k-axis A/B fragments
→ L2 cache中的fragments难以被多个CTA复用
→ cache hit rate下降
```

解决方案："two-tile Stream-K + data-parallel"混合调度——减少Stream-K的应用范围，使大部分CTA执行完整的data-parallel wave，仅对最后wave的剩余tile做iteration balancing。这限制了skew的持续时间，使大部分执行在无skew的完整wave中进行，同时cache locality得到改善。

术语一般如何实现？如何使用？
混合调度（Section 5.2）通过以下策略控制skew：
- "Data-parallel + one-tile Stream-K"：仅将Stream-K应用于最后部分wave的tile
- "Two-tile Stream-K + data-parallel"：每CTA接收1-2个tile的iteration量，既限制skew，又确保每tile最多2个CTA覆盖（良好同步隐藏）
- 对于w≥2（至少2个完整wave），每accumulating CTA仅需接收1个其他CTA的partials
混合调度的实现使用与基本Stream-K相同的kernel结构，仅通过修改grid size和iteration分配逻辑来实现不同的调度策略。

涉及论文标题：
- Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU

---
