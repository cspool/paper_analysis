## Triton（Triton DSL 与多目标编译：Triton-MLIR → GPU/AMDGPU/MTIA dialect → LLVM-IR）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Triton 是 OpenAI 提出的 Python 嵌入式 tile/block 级 GPU 编程 DSL（Philippe Tillet 2019）：程序员用 `@triton.jit` 装饰的 Python 函数描述 kernel，用 `tl.load`/`tl.store`/`tl.dot`/`tl.arange` 等 tile 级原语操作张量块，编译器负责把 tile 映射到线程/块，自动做内存 coalescing、shared memory 分配与同步，屏蔽 CUDA 的显式线程管理。在 KernelEvolve 论文中，Triton 已从 GPU 专属 DSL 演化为"多目标编译"的异构加速器语言：Triton 源码经渐进式 MLIR lowering——平台无关的 Triton-MLIR → 硬件专属 dialect（GPU/AMDGPU/MTIA）→ LLVM-IR → 生成 NVIDIA PTX/CUBIN、AMD AMDGCN/HSACO、MTIA RISC-V 原生二进制（论文图 2）。Meta 内部 Triton kernel 已超 8000 个（60% 年增长率），超过 CUDA 遗留代码，成为推荐模型训练/推理 kernel 的主导编程模型；并延伸出 TLX（Triton Low-Level Extensions，warp 感知 intrinsics + 显式 pipeline 控制，针对 Hopper/Blackwell 的 TMA/WGMMA）、Gluon（Triton 编译器栈内的低层 layout 编码 dialect）、MTIA-Triton（triton_mtia，含 eager launcher `mtia_triton_launcher`）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在编译框架中的角色：Triton 是"编译器输入语言 + 编译流水线本身"的结合体。运转流程例子（KernelEvolve 的 conv1d_gemm_kernel 编译执行）：①`@triton.autotune(configs=20+组, key=["M","N","K","stride",...])` 包裹的 `@triton.jit def conv1d_gemm_kernel(...)` 源码（含 `tl.static_range` 展开 kernel taps、`tl.dot` 矩阵乘、double-buffer 预取）→ ②平台无关 Triton-MLIR 中间表示（TTIR，tile 级语义）→ ③后端专属 lowering：NVIDIA 走 TritonGPU dialect（插入 TMA/WGMMA/异步拷贝与 mbarrier 同步）、AMD 走 AMDGPU、MTIA 走 MTIA dialect（libdevice/SFU、circular buffer）→ ④LLVM-IR → ⑤目标汇编与二进制（PTX/CUBIN、AMDGCN/HSACO、RISC-V）→ ⑥运行时 launch（wrapper 计算 grid）。MTIA-Triton 还支持 `emit_cxx=True` 暴露编译中间 C++（RISC-V vector intrinsics、circular buffer 指针管理、SFU 初始化、core affinity），供 `replay_cpp` 免全量重编译调试。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：开源在 https://github.com/triton-lang/triton（BSD 协议），Python 包 `pip install triton`；Meta 维护 fork/TLX 在 https://github.com/facebookexperimental/triton。标准用法：`@triton.jit` 定义 kernel → `kernel[grid](args, BLOCK_SIZE=...)` 启动 → `@triton.autotune` 自动在多个 config 间按 key 选择最快变体。KernelEvolve 把它作为 agentic kernel 生成的目标语言：LLM 生成 Triton 源码 → TritonBench 验证正确性（torch.allclose vs PyTorch 参考）与 speedup → 平台专属解释器（meta_kernel_gpu/amd/mtia_interpreter）执行。局限：Triton 抽象对某些低层优化（直接 PTX/SASS、MLIR dialect 级变换）不够，KernelEvolve 未来方向提出扩展 TLX/MLIR/PTX 直接生成。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta
