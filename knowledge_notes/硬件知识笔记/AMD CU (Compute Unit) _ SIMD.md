## AMD CU (Compute Unit) / SIMD

术语是什么？
AMD CU（Compute Unit）等价于 NVIDIA SM。每 CU 含 4 SIMD，每 SIMD 可同时执行一个 wave（64 线程的锁步执行组），512 个 32-bit 寄存器，以及 Matrix Core。MI355X 共 256 CU (=1024 SIMD)。

从硬件架构角度拆解术语：
在 8-WAVE PING-PONG 中，每 CU 4 SIMD 各驻留 2 wave = 8 wave total。每 SIMD 的 2 wave 交替 compute/memory 角色。CU 资源：L1 cache（CU 私有）、LDS 64KB（CU 内 wave 共享）、512 register/SIMD（1 wave 时 256 VGPR+256 AGPR；2 wave 时每 wave 256 VGPR）。与 NVIDIA SM 差异：wave 64 vs warp 32，register 512 vs 256/SIMD，Matrix Core 在 SIMD 上直接执行 MFMA（vs 独立 Tensor Core），无 wgmma 从 shared memory 做矩阵乘。

术语一般如何实现？如何使用：
由硬件管理，__launch_bounds__ 控制每 wave 寄存器数以最大化 occupancy。rocprof/rocprofv3 可查询 CU 的 wave occupancy、cache hit rate、bank conflict。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---
