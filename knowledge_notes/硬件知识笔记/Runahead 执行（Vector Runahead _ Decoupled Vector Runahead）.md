## Runahead 执行（Vector Runahead / Decoupled Vector Runahead）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Runahead 执行是一类投机预执行技术：当乱序核因 cache miss 阻塞时，暂停主线程的提交，用空闲资源继续"预执行"未来指令，提前计算并发出后续访存，从而在等待 miss 期间暴露内存级并行（MLP）。Vector Runahead（VR，ISCA 2021）扩展经典 runahead：在阻塞窗口内把未来多个循环迭代的标量操作投机重排为 SIMD 向量操作（如 gather），一次发出一批相互独立的依赖链 miss。Decoupled Vector Runahead（DVR，MICRO 2023）进一步把 runahead 从主乱序流水线中解耦，在专用轻量子线程上下文（on-demand、in-order、speculative）中运行，动态推断循环边界、识别 striding load、向量化间接链，在主核 stall 之前就前瞻发出访存（Web 证据：DVR 相对 5-wide OoO baseline 与 VR 分别提速 2.4× 和 2×，硬件开销约 1139B）。ICP 论文指出其两个缺陷：(1) 硬件复杂度高——VR/DVR 需大幅修改 CPU 核（mode-aware decode、DVR 生成向量化操作的执行支持、保证不更新架构状态的提交/recovery 逻辑、内存压力节流、子线程管理）；(2) 与间接预取器一样从 striding load 开始发现依赖链，限制通用性。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- Runahead 在 CPU 核心流水线中的运转流程（以 ICP 论文的对比评估为例）：VR 在主流水线内——ROB stall 窗口触发 runahead 模式 → 投机执行未来指令、向量化循环迭代为 gather → 一批依赖 miss 同时飞行 → miss 返回后退出 runahead、恢复主线程；DVR 在主流水线旁——专用子线程上下文动态发现 striding load → 向量化间接依赖链 → 提前发出访存（main thread 尚未 stall）→ 对内存压力做节流。ICP 论文把 VR/DVR 集成在主流水线评估（Table III 对比）：DVR 需要专用子线程、mode-aware decode、向量化执行支持、终止/恢复逻辑与节流机制；ICP 则完全不需要这些——不改 decode/execute、无额外线程，只通过 commit 异步 FIFO + 标准 cache prefetcher 接口与核解耦。性能对比：VR 仅在有足够 runahead 时间时有效（SPEC/GAP 均不佳），DVR 改善但仍对复杂依赖（mcf、gcc）失效；ICP 超 VR 15.03%、超 DVR 3.74%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：VR 在乱序核内扩展 runahead 状态机 + SIMD 向量化（checkpoint/recovery 保证架构状态安全）；DVR 增加一个解耦的轻量子线程执行引擎（~1139B 硬件开销）+ 循环推断与向量化单元；两者都需与主流水线深度耦合（Web 证据：VR 在剑桥大学仓库公开，DVR 见 MICRO 2023，DOI 10.1145/3613424.3614255）。使用场景：作为"投机执行类"不规则访存方案的 baseline，与缓存本地预取器（ICP、间接预取器）对比时突出复杂度-收益权衡；ICP 的关键论点即"ICP 用缓存本地机制达到接近 runahead 的覆盖，却不需要 runahead 级核内复杂度"。

涉及论文标题：
- ICP: Exploiting Instruction Correlation for Prefetching Irregular Memory Accesses
