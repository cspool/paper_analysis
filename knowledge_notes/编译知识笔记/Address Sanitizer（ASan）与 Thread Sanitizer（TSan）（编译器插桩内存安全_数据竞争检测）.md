## Address Sanitizer（ASan）与 Thread Sanitizer（TSan）（编译器插桩内存安全/数据竞争检测）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ASan（AddressSanitizer，Serebryany et al., USENIX ATC'12）是编译器插桩的内存错误检测器：编译期（GCC/LLVM 均支持，-fsanitize=address）在每次 load/store 前插入对 shadow memory 的检查，用"影子内存 + 红区（redzone）"以 ~2× 运行时间开销检测堆/栈/全局的 buffer overrun 与 use-after-free。TSan（ThreadSanitizer，WBIA'09）是数据竞争检测器（-fsanitize=thread）：插桩每次内存访问并记录 happens-before 关系（vector clock），检测并发访问同一地址且至少一方是写、且无同步边连接的竞争。两者都要求编译器在 IR/后端插桩，与 SPEC CPU 的"编译框架"层次（工具链/编译器能力）直接相关。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
从编译框架角度看，ASan/TSan 是编译器插桩通道（instrumentation pass）的产物，运转流程：
```
ASan 编译插桩:
  源码 → 前端 → 优化 → ASan pass(每内存访问插入 __asan_loadN/__asan_storeN 调用 +
        对全局/栈对象加 redzone + 建立 shadow 映射) → 链接 ASan runtime
  运行: 访问地址 → 查 shadow(地址>>3 + offset) → 非零且不匹配 → 报错(含调用栈/分配栈)
TSan 编译插桩:
  每内存访问插入 __tsan_read/write + 函数进入/退出插入 __tsan_func_entry/exit
  运行: 维护 per-thread vector clock + happens-before 边 → 检测无同步保护的乱序访问
SPEC CPU2026 用法: 每个 benchmark 用 GCC 与 LLVM 两家的 ASan 构建并测试
  (发现 767.nest/735.gem5 等内存缺陷, 修复后回馈上游), 多线程 SPECspeed benchmark
  再用 TSan 查数据竞争 (发现 867.nest/837.gmsh 的线程竞争)。
```
SPEC CPU 选择 C/C++/Fortran 而非托管运行时的一个重要原因正是可复现性——JIT 引入 run-to-run 方差，而编译期插桩的 sanitizer 可确定性覆盖。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：GCC 与 LLVM/Clang 内建（gcc -fsanitize=address、clang -fsanitize=address,thread），运行时库（libasan/libtsan）负责 shadow 管理、红区、符号化报告；ASan 需要 shadow memory 映射（Linux 上 1/8 地址空间）+ 红区（默认 128B）；TSan 需要 shadow cell 与内存访问元数据（约 1/2 地址空间）。使用方式：构建期加 -fsanitize=address 跑测试/训练规模，捕获错误后用 ASAN_OPTIONS/TSAN_OPTIONS 调整；KASAN 是内核版本，HWASAN 是 ARM 硬件标记版本（复用 MTE）。局限：ASan 内存开销 ~2×、运行开销 ~2×（CPU2026 用于测试规模验证而非正式 ref 运行）；TSan 开销更高（~5-15×），仅用于多线程 benchmark 的竞争检测。文档：GCC Instrumentation Options（gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html）、Clang AddressSanitizer（clang.llvm.org/docs/AddressSanitizer.html）。

涉及论文标题：
- SPEC CPU: The Next Generation
