## Shared Memory Bank Conflict / Swizzle on AMD CDNA

术语是什么？
AMD CDNA shared memory bank conflict 发生在 wave 内多线程同一 phase 访问同一 bank。Swizzle 通过重排数据布局消除冲突。AMD banking 比 NVIDIA 更复杂：不同指令使用不同 bank 数量（64/32）和不同 phase ordering（非顺序的线程分组），且 AMD MFMA 各形状使用完全不同的 thread-to-element mapping，单一 swizzle 无法覆盖所有访问模式。

从kernel调度角度拆解术语：
HipKittens 自研 solver 反向工程 phase/bank 行为（Table 5），识别常共同出现的 layout 对（如 16x32 row+col load），为这些对设计 bank-conflict-free swizzle。例如 ds_read_b128 要求 128-bit 连续，但 ds_write_b64 的 XOR swizzle 以 64-bit 为单位打散数据，两者冲突。在 attention backward 中需两个不同 swizzle 分别服务于 row-layout 16x16 write（ds_write_b64）和 row-layout 16x32 read（ds_read_b128）。

术语一般如何实现？如何使用？
AMD phase/bank 行为未在 ISA 手册中文档化。HipKittens solver 自动探测。Swizzle 通过 XOR/移位在全局地址（HBM offset）或 LDS offset 上实现。CDNA4 新增指令（如 ds_read_b96）有新的 phase 行为。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---
