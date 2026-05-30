## XCD (Accelerator Complex Die) — AMD Chiplet GPU

术语是什么？
XCD（Accelerator Complex Die）是 AMD CDNA GPU 的 chiplet 计算 die。MI355X 含 8 个 XCD，每 XCD 有 32 CU + 4MB 私有 L2 cache，XCD 之间通过 Infinity Fabric 互联，共享 LLC 和 HBM。类似 NVIDIA Blackwell 的 2-die 设计，但 XCD 数更多（8 vs 2），cache hierarchy 更复杂。

从硬件架构角度拆解术语：
硬件 scheduler 以 round-robin 方式将 thread block 分配到 XCD。对 kernel 性能的影响：(1) 同 XCD 内 block 共享私有 L2，若 block 访问不重叠的 A/B 区域（如 naive row-major grid），L2 hit rate 低至 36%；(2) 跨 XCD 数据通过 LLC 共享，LLC 带宽远低于 L2（约 3x 差距）。HipKittens chiplet swizzling (Algorithm 1) 通过 XCD grouping（chunks of C blocks 归同一 XCD）+ hierarchical windowing（W 高度窗口垂直遍历）联合优化 L2 (78-79% hit rate) 和 LLC 复用，带宽提升 19%。

术语一般如何实现？如何使用？
XCD 是硬件物理设计，对开发者透明。影响性能只能通过 grid scheduling（block 排列顺序）间接控制。rocminfo 可查询 CU 的 XCD 归属。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---
