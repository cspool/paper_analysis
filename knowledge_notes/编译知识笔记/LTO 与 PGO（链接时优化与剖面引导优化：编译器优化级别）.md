## LTO 与 PGO（链接时优化与剖面引导优化：编译器优化级别）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LTO（Link-Time Optimization，链接时优化）与 PGO（Profile-Guided Optimization，剖面引导优化）是两种跨模块/面向运行特征的编译器优化方式。LTO：把各编译单元的中间表示（GIMPLE/IR）保留到链接期，链接器驱动全局优化（跨函数内联、死代码消除、常量传播、IPA 分析），弥补传统"每文件独立编译"看不到跨模块关系的缺陷（GCC -flto、Clang -flto）。PGO：分三阶段——instrumented build（插桩版本）→ 用代表性输入运行收集 profile（执行频率、分支方向、间接调用目标、热函数）→ 用 profile 重编译（-fprofile-use），使编译器按实际运行特征优化（热路径内联、分支布局、函数分段、冷热函数分离）。两者常组合（-flto -fprofile-use），SPEC 在可移植性验证中把它们与 -O2/-O3 并列作为多优化级别测试矩阵的一部分。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
SPEC CPU2026 中 LTO/PGO 的定位是"可移植性验证的优化级别矩阵"（论文 §III-B Validation Across Multiple Systems）：每个候选 benchmark 须在 GCC、LLVM 及 Intel/AMD/IBM/NVIDIA/HP-Cray/Microsoft 等 vendor 编译器下，以 -O2、-O3、LTO、PGO 多级别构建，且都要与 golden reference 结果在容差内一致，才通过验证。运转流程：
```
SPEC 验证流水线:
  源码 → 配置编译器与优化级别(-O2|-O3|LTO|PGO) → 构建
  → 在 x86/ARM/POWER/RISC-V × Linux/Windows/macOS 矩阵上运行
  → 与 golden reference 结果比对(容差内一致 = pass)
  → 任一级别失败 → 修代码/修编译器 bug, 再验证
```
CPU2026 还因 GCC/LLVM 等社区编译器与 vendor 编译器的性能差异，新增"vendor 支持 vs 社区开源编译器"两类结果报告类别，体现编译器选型对评测结果的实质影响。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：LTO——GCC/Clang 在 -O 编译时输出 IR 而非目标码，链接器（gold/lld/lld-link 或 gcc 驱动）合并 IR 做 IPA 后生成目标码（GCC 用 -flto，Clang 用 -flto=thin 支持 ThinLTO 分片并行）；PGO——GCC/Clang 两遍构建（-fprofile-generate 跑代表输入 → -fprofile-use），profile 文件含边频/值/间接调用信息。使用场景：性能关键代码的最终发布构建（服务器软件、HPC）；SPEC 用其验证基准在不同优化级别下的结果一致性（保证基准对编译器优化不敏感、不被单一优化技巧"crack"）；LTO/PGO 也是基准"多 workload 抗破解"设计的对抗面——编译器优化在某个输入上显著加速但跨输入不泛化即被视为脆弱。注意：SPEC CPU2026 的部分浮点 benchmark 需要精确数学（fast/relaxed math 可能不通过验证），这与激进优化存在张力。

涉及论文标题：
- SPEC CPU: The Next Generation
