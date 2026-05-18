## GPU Global Memory Bandwidth Saturation and In-Flight Data（GPU全局内存带宽饱和与飞行中数据）

术语是什么？通过联网搜索让回答具体和精准。
Global memory bandwidth saturation是GPU kernel性能优化的核心概念：由于GPU global memory (HBM)的访问延迟很高（A100约200ns），kernel必须维持足够的"in-flight data"（已发起但未完成的memory请求的总数据量）来覆盖这一延迟，否则memory bus会处于空闲等待状态而无法达到峰值带宽。In-flight data量 = 并发CTA数 × 每个CTA的in-flight K/V tile data量。PAT将此概念应用于decode attention的tile size选择：KV tile size n必须足够大，使得所有concurrent resident CTA的总in-flight data ≥ memory latency × peak bandwidth（即Little's Law: D_flight ≥ L × B），否则HBM bandwidth无法打满。在A100上，L≈500ns, B≈1.8TB/s→2TB/s(peak)，推导得n ≥ LB/(S*C*h*b)，其中S=108 SM, C=per-SM resident CTA数, h=head dim, b=datatype size。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
A100 GPU上的in-flight data分析和bandwidth saturation验证（论文Figure 8a-8d）：

```
// === Micro-benchmark: memory latency and bandwidth ===
// 测量global→shared transfer latency vs data size (1000次平均)
Data_size  Latency
  8B       ~500ns  ← 固有memory latency L (flat region)
 16B       ~500ns
  32B      ~500ns
 64B       ~510ns
128B       ~550ns  ← 开始线性增长 (bandwidth-limited region)
256B       ~650ns
512B       ~850ns
 1KB      ~1.2us

// 线性区斜率 → sustainable bandwidth B ≈ 1.8 TB/s

// === Bandwidth saturation condition ===
// D_flight = S * C * n * h * b (所有resident CTA的K tile in-flight data)
// 要求: D_flight ≥ L * B

// A100参数: S=108, L≈500ns, B≈1.8e12 B/s (≈1.8 TB/s)
// 假设 C=2 (每SM 2个resident CTA), h=128, b=2 (FP16)
// L*B = 500e-9 * 1.8e12 ≈ 900 bytes
// n ≥ 900 / (108 * 2 * 128 * 2) ≈ 0.016 → 极小的n即可满足
// 但实际C受shared memory和register限制, 通常C=1-2
// 且考虑到tile select的overhead和实际使用, n ≥ 16即可

// === 验证: 不同tile size的bandwidth utilization ===
// Figure 8c (A100, batch 1134, KV length 1024, no shared prefix):
(m,n)     Bandwidth Util  Latency(ms)
(16,16)   86.2%           2.73           ← feasible (✓ all constraints)
(16,32)   86.3%           2.73
(16,64)   85.4%           2.74
(16,128)  84.5%           2.79
(32,16)   84.0%           2.80
(32,32)   85.2%           2.76
(32,64)   86.3%           2.73
(32,128)  84.5%           2.79
(64,16)   73.9%           3.19           ← 带宽下降(m大→C小→D_flight)
(64,32)   83.3%           2.82
(64,64)   83.9%           2.81
(64,128)  84.4%           2.79
(128,16)  50.4%           5.62           ← m=128时register压力大,C极小
(128,32)  59.5%           3.96
(128,64)  68.0%           3.54
(128,128) 40.3%           6.08           ← register spill severely

// Feasible set (satisfying all 3 constraints): 下划线标记的11组
// 所有feasible configs达到83-86% bandwidth util, latency差异<2%
```

H100上的类似分析（Figure 9, batch 1188）：
feasible configs达到92.3-94.2% bandwidth utilization，更优越的HBM bandwidth (3.35 TB/s)使得high-bandwidth配置的约束更宽松。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Memory latency/bandwidth micro-benchmark通过CUDA实现：重复执行global→shared memory async copy (cp_async) 1000次，sweep data size从8B到数KB，记录每次transfer的latency（通过CUDA event timing）。从latency vs data size曲线提取flat region的latency值(L)和linear region的斜率(B)。该profiling是offline tile set solver的关键输入，用于推导bandwidth constraint lower bound。移植到新GPU（如A100→H100）时需重新运行micro-benchmark获取该GPU的L和B值，并结合shared memory/register constraints重新推导feasible tile set。

涉及论文标题：
- PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

