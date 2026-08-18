## Page Coloring（内存着色 / Memory Coloring）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Page coloring 是把"数据页 → 片上 SPM page frame"的分配问题形式化为图着色问题（类比寄存器分配：每个 SPM page frame 相当于一个架构寄存器）[63]。编译器分析静态划分页的 live range（活跃区间），给 live range 冲突（同一时刻都存活）的页分配不同颜色（不同 frame），从而避免冲突页竞争同一 frame。
- 传统 page coloring [63] 的局限（MANATEE 的动机）：它针对连续供电系统，要求冲突页在整个程序执行期都不同色；若颜色不足，冲突页被永久 spill 到 off-chip 内存。这忽略了 EHS 频繁断电产生的短功率周期——断电会清空 SPM，被断电隔开的页根本不会共存，无需不同色。这正是 MANATEE"intermittence-aware speculative page coloring"的改进点。
从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子：①解析 linker .map 得 data section 的 NVM 物理地址，按 64B 划分逻辑页为线性页数组；②在 CFG 上做页级 liveness 分析，得各页 live range；③构造干扰图（live range 相交的页连边），用图着色为页分配颜色（SPM frame）；④颜色不足时（MANATEE 场景）触发滑动窗口投机着色：以功率周期 T_on 为窗口，窃取窗口外页面颜色复用。传统着色在此步直接永久 spill。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：编译器静态分析（LLVM + linker .map 信息），零硬件改动。论文中 Memory Coloring 是 baseline 之一（有安全支持、无滑动窗口），相对 Unsecure 开销 1.93×、页 miss rate ~2.05%；MANATEE 加滑动窗口后 miss rate 降到 ~0.99%、开销 1.71×（平均快 ~12%）。论文未给出公开代码，无法确认是否开源。
涉及论文标题：
- Intermittence-aware Speculative Page Coloring for Secure NVM
