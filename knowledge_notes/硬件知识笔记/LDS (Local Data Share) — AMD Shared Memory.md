## LDS (Local Data Share) — AMD Shared Memory

术语是什么？
LDS（Local Data Share）是 AMD CDNA CU 内 wave 共享的片上内存，等价于 NVIDIA Shared Memory。MI355X LDS = 64KB/CU。通过 ds_read/ds_write 指令访问，bank 结构和 phase ordering 因指令而异（ds_read_b128: 64 banks/4 phases；ds_read_b96: 32 banks/8 phases）。

从硬件架构角度拆解术语：
在 HK GEMM kernel 中 LDS 作为 tile double buffer：
```
st_bf<128, 64, st_16x32_s> (&As)[2][2] = al.allocate<...>(2, 2);
```
数据流：HBM → buffer_load_dword → LDS → ds_read_b128 → register → MFMA。AMD LDS bank conflict 比 NVIDIA 更复杂：bank count (64/32) 和 phase ordering（非顺序）均随指令变化（论文 Table 5）。单一 swizzle 无法覆盖所有访问模式（如 ds_write_b64 以 64-bit 打散，破坏 ds_read_b128 所需 128-bit 连续性）。

术语一般如何实现？如何使用？
HIP C++ 通过 extern __shared__ / __shared__ 声明 LDS。HipKittens shared_allocator 封装分配和对齐。64KB 上限在 double buffering 时限制 tile 最大尺寸。MI325X 仅 65KB LDS 无法 double buffer in LDS，改用 register file double buffer + ds_write 中转。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---
