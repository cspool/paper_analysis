## HIP / HIPCC (AMD Heterogeneous-Compute Interface for Portability)

术语是什么？
HIP（Heterogeneous-Compute Interface for Portability）是 AMD 提供的 C++ 运行时 API 和编译器工具链，用于在 AMD GPU 上进行通用计算。HIPCC 是 HIP 的编译器驱动，将 HIP C++ 源码编译为可在 AMD GPU（通过 ROCm 栈）或 NVIDIA GPU（通过 CUDA 栈）上执行的代码。HIP 在语法上高度兼容 CUDA，使得移植 CUDA kernel 到 AMD 平台只需少量修改（如 `cudaMalloc` → `hipMalloc`、`__global__` 保持不变、`threadIdx.x` 保持不变）。

从编译框架角度拆解术语：
HIPCC 的编译流程：HIP C++ 源码 → HIPCC 前端解析 → 生成 LLVM IR → AMDGPU LLVM backend 生成 GCN/CDNA ISA assembly → 链接为 HSA Code Object (.hsaco) → AMD GPU 可执行。然而，论文发现 HIPCC 存在关键限制：(1) 不支持将 AGPR（累加寄存器）作为 MFMA 矩阵指令的输入操作数，即使硬件支持，导致编译器必须插入冗余的 v_accvgpr_read 指令将数据从 AGPR 搬移到 VGPR；(2) 寄存器分配和生命周期管理不完全可控，开发者无法精确调度哪些变量分配到哪些寄存器。论文的解决方法是 bypass HIPCC，通过内嵌汇编（asm volatile）和 pinned register tiles 直接控制寄存器分配。HIPCC 也接受 LLVM compiler hints（如 __builtin_amdgcn_sched_barrier），但 asm volatile 内的代码对编译器是黑盒。

术语一般如何实现？如何使用？
HIP 通过 ROCm 软件栈（包含 ROCclr runtime、ROCm 驱动、HIP runtime library）运行。编译命令：`hipcc -x hip --offload-arch=gfx950 kernel.cpp`，其中 `gfx950` 对应 CDNA4 MI355X。HIP 的核心头文件 `<hip/hip_runtime.h>` 提供与 CUDA API 对应的函数。在 HipKittens 中，HK 作为 C++ header-only 库直接 include，用户编写使用 tile 原语的 kernel，通过 HIPCC 编译（正常路径），或在需要最大控制时通过 explicit register scheduling + asm volatile 绕过编译器限制。Python bindings 通过 pybind11 暴露 HK kernel 给 PyTorch 用户。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---
