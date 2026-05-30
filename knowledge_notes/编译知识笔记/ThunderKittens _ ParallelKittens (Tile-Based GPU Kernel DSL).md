## ThunderKittens / ParallelKittens (Tile-Based GPU Kernel DSL)

术语是什么？
ThunderKittens (TK) 是Stanford HazyResearch开发的CUDA C++ embedded DSL，以16×16 tile为最小执行单元，提供PyTorch/NumPy风格的操作符在tile上进行bulk操作。核心设计基于三层GPU抽象：(1) warp级——16×16 tile数据结构（rt_bf/rt_fl register tile, st_bf shared tile, gl global layout descriptor），提供PyTorch风格的bulk操作（mma, exp, cumsum, sub_row, div_row等），编译时自动选择shared memory swizzle布局（32B/64B/128B）消除bank conflict；(2) block级——LCSF (Load-Compute-Store-Finish) 统一异步编程模板，基于生产者-消费者范式将kernel分解为四个阶段，框架自动管理multi-stage pipeline buffer、同步barriers和TMA descriptor；(3) grid级——persistent grid + block order scheduling消除launch开销并最大化L2 cache reuse。内部通过C++ template元编程直接包装PTX assembly指令（wgmma, TMA），include/目录<1MB，以约217行代码实现匹配FlashAttention-3性能的attention kernel（FA3 CUTLASS实现约2325行）。ParallelKittens (PK) 是TK的多GPU扩展，添加了8个核心原语和LCSC统一编程模板。

从编译框架角度拆解术语：
TK/PK是C++ header-only库（非独立编译器），编译流程：C++ template代码 → nvcc/host compiler → CUDA binary。作为embedded DSL，TK的"编译"能力体现在：(1) 编译时layout检查——mma_AB要求A=row-major B=col-major，不匹配则编译报错（而非运行时错误或静默错误）；(2) 编译时swizzle选择——根据tile width和数据类型在32B/64B/128B swizzle中自动选择最优布局；(3) 编译时TMA descriptor生成——从gl global layout自动推导TMA tensor map descriptor；(4) 编译时register分配——通过模板参数（NUM_CONSUMER_WARPS）静态确定warp分工和寄存器配额。TK处于CUTLASS（完全手动，22MB include）和Triton（完全编译器自动，12.6MB+LLVM）之间的平衡点：保留C++ full power但仅用<1MB实现意见鲜明（opinionated）的抽象集。

术语一般如何实现？如何使用？
开源：https://github.com/HazyResearch/ThunderKittens（含PK扩展）。用法：`#include` header文件，定义LCSC template struct，调用lcsc::launch_kernel。PyTorch集成通过pybind11。支持H100 (SM 90) 和 B200 (SM 100)。已有Triton Community的移植effort（TileLang）。

涉及论文标题：
- ThunderKittens: Simple, Fast, and Adorable Kernels
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels
- HipKittens: Fast and Furious AMD Kernels
