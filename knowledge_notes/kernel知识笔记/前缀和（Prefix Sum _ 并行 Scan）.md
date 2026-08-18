## 前缀和（Prefix Sum / 并行 Scan）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
并行计算中的基础原语：给定数组 a[0..n-1]，输出 b[i] = Σ_{j≤i} a[j]（包含式 inclusive）或 Σ_{j<i} a[j]（排他式 exclusive）。高效并行实现用工作高效的并行扫描算法（Blelloch 的 up-sweep/down-sweep 树形两阶段：O(n) 工作、O(log n) 深度）。在现代加速器上，GPU 用 CUB/Thrust 的 warp/block scan（含 Kogge-Stone、Brent-Kung 等变体 + 跨 block 的 chunk 扫描与流水）；在 Ascend NPU 上则受 32 字节段内禁止 SIMD 的约束，ENEC 用 IDD-Scan 绕行（见 IDD-Scan 条目）。ENEC 论文中前缀和占解压 kernel 30% 的开销（基础版），是解压性能的关键瓶颈之一。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ENEC 中前缀和的用途：解压时 bit mask（0/1）→ 前缀和 → 每个元素在压缩流中的起始偏移，供逆 gather 取数：
```
mask = [1,0,1,1,0,1,0,1]          # 8 个组是否"超出 m 位需要额外字节"
offset = prefix_sum(mask)          # [1,1,2,3,3,4,4,5]（排他）
for i in range(N):
    low[i] = stream[offset[i]]     # 按偏移逆 gather，偏移=前面需要额外字节的组数
```
Annotations：mask 中 1 的数量决定每个组相对起始位置的偏移；前缀和把"动态偏移计算"向量化。GPU 侧（ENEC-GPU-V1）用 CUB 的 parallel prefix sum + warp 内建快速通信；CPU 侧用 AVX2/BMI2 PEXT 处理非字节对齐。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：GPU 用 CUB cub::DeviceScan/BlockScan（Blelloch 或 decoupled look-back 变体）；Ascend 用 IDD-Scan（转置+列扫描+行扫描）或朴素逐元素（低效）；CPU 用 std::partial_sum 或 AVX2 向量化。使用：除解压偏移计算外，广泛用于 softmax 分母、注意力位置编码、内存压缩、基数排序、负载均衡等。ENEC 的经验：在 Ascend 上扫描是"没有硬件直接支持"的原语，性能取决于能否绕开段内依赖约束。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs
