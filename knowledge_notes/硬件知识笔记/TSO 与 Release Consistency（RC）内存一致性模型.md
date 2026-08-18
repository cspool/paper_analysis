## TSO 与 Release Consistency（RC）内存一致性模型

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TSO（Total Store Order，x86/SPARC/RISC-V RVTSO）比顺序一致性弱一条序：保留 Load→Load、Load→Store、Store→Store，放宽 Store→Load——写先退休进 store buffer，按 FIFO 顺序 drain 进内存系统，本核后续 load 可经 store-to-load forwarding 提前读到自己的写（故其他核可先于本核观察到写之前的读）。x86 上唯一需要显式屏障的是 StoreLoad（MFENCE/lock 前缀），acquire/release 语义几乎零成本（https://stackoverflow.com/questions/69925465、Rice COMP522 讲义 https://www.cs.rice.edu/~johnmc/comp522/lecture-notes/COMP522-2019-Lecture9-HW-MM.pdf）。RC（Release Consistency）把序挂在同步操作上（acquire load / release store 围出临界区），普通读写几乎无约束，store buffer 内写可乱序 drain、互相重叠。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Dorado 的模拟核模型显式实现两级模型：200 项 LSQ 内有一指针划出 Store Buffer（SB）= 已退休但未与内存系统合并的写区间；TSO 下 SB 严格顺序 drain，RC 下任意序、可重叠；load 地址一知即投机乱序执行，退休前发现一致性违规则冲刷该 load 及后续指令重放；store 地址一知即发 write-exclusive prefetch 把行以独占态拉进 L1（掩盖写延迟）。实验（Fig.17，归一化到 Dir2B-TSO）：RC 使写延迟最长的 Dir2B 提升最大（1.06×），Hier4 1.20→1.24，Dorado 1.36→1.38；Dorado 对 Hier4 的优势 1.13×→1.11×。论文用此证明 Dorado 在两种模型下都稳定领先。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：硬件侧 SB + 投机 load + prefetch；软件侧 x86 靠 MFENCE/lock 实现 StoreLoad 屏障，ARM 等弱序模型由编译器插入 acquire/release fence。模拟侧：gem5 用 writebuffer drain 建模、SST Ariel 被作者改造为高精度 TSO（本条目）。使用要点：一致性模型与一致性协议正交——模型定可见性顺序、协议定行状态维护；写延迟敏感的负载在 RC 下收益大，但这会削弱"低写延迟协议"的相对优势（如 Dorado vs Hier4）。QED 的 TSO 视角补充（表 I/II）：TSO 只放宽异地址 store-load 程序序（load 可先于更老的异地址 store 执行以隐藏 store 延迟），同地址 store-to-load 必须保持程序序（store-to-load bypass/转发后 load 在全局序中仍排在产生值的 store 之后）；TSO 包含 atomic（对排序而言与 store 等价），且要求写原子性（除本线程早读自己的写外）。QED 为 TSO 生成 36 棵探索树/51 个谓词（含 trivial 树 12 棵）。

涉及论文标题：
- Dorado: Clustered Hardware Cache Coherence for 1,000+ Cores
- QED Scalable Consistency Verification of Memory Instruction Reordering in Hardware
