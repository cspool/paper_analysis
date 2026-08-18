## Stride-based Prefetching（步长预取：多 way 多 stride 动态深度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
基于恒定步长（stride）预测未来地址的预取范式，与空间局部性预取并列为硬件 cache 预取器两大类别：观察访存流中连续地址差恒定时，按 stride 外推预取后续地址。相对空间局部性预取器，stride 预取通常 accuracy 更高、预取请求更少，特别适合多 GPU 页面预取（accuracy 关键）。LIBRA 观察（Takeaway 2）：GPU 每 SM 的 VPN 访问呈多 way（多个独立访存流，按地址距离阈值分组）与多 stride（同一 way 内并发多个 stride 且会发生 stride 迁移，如 stride 8→4、同时 1/6/7）；四个 SM 的 stride 相同但因大规模并行而访问非顺序（Takeaway 1 引申：应逐 SM 独立学习）。设计原则：per-way、per-SM 独立学习；每 SM 维护多 way；每 way 维护多 stride。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
LIBRA MMP 的动态深度预取算法：每 way 维护 4 个最高频 stride 及其计数器，far-fault 时按式(1) stride_likelihood=该 stride counter/所有 stride counter 之和决定预取深度，概率每 +25% 多预取 1 页。伪代码：
```
on far-fault(vpn) 匹配到 SM 的 way w:
  for s in w.top4_strides:
    like = w.counter[s] / w.sum
    n = 1 + floor(like / 0.25)      # 每 +25% 概率多预取 1 页
    for k in 1..n: prefetch(vpn + k*s)
  # 出现第 5 个 stride 时淘汰最不频繁者
```
（Annotations：way 由"VPN 差 < 阈值 512"匹配，多匹配取差最小；far-fault 同时携带该页估计未来访问数以支撑成本收益。图 8 例子：stride +2 概率 0.52 → 预取 0xa2+2、0xa2+4。）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件表实现（MMAT）：每 SM 4 行（way），每行 36-bit last VPN + 8-bit access counter + 4×(6-bit stride + 6-bit occurrence counter) + 8-bit sum + 36-bit monitored VPN；L3 TLB miss 转发学习（更新 stride/计数器/last VPN），far-fault 触发预测。CPU cache-line 级 stride 预取器（IPCP 的 Constant-Stride 类、Berti 等）是同一思想在更细粒度的实现；LIBRA 首次把 stride-based 预取上移到页粒度用于多 GPU UVM 迁移。论文未开源实现（无法确认）。

涉及论文标题：
- LIBRA: A High-Accuracy, Cost-Aware, and Coordinated Multi-GPU Page Prefetcher
