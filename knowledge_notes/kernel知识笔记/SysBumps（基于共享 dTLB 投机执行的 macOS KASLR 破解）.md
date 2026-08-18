## SysBumps（基于共享 dTLB 投机执行的 macOS KASLR 破解）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SysBumps（Jang et al., CCS 2024）是破解 macOS 13.1–15.1 KASLR 的攻击：虽然 Apple 的 Double Map 隔离了用户页表与内核页表，但 data TLB（dTLB）跨特权级共享。当对已映射物理地址触发投机执行时，架构效应会在共享 dTLB 中驱逐 priming 的条目（投机结果本身被隔离，但 TLB 效应可见）。攻击用 Prime+Probe 技术 + counting-thread 定时器测量 probe 时间，遍历 32,768 个候选槽位，判定每个地址是否有有效物理映射，从而去随机化内核基址。
- 在本文中：SysBumps 作为 TIDE 去噪原语的验证载体——原版 SysBumps 在 idle 下成功率 92%（与论文 95.7%–98.8% 相当），在 Chrome v130 播放 YouTube 短视频造成的中断噪声下降到 54%；用 TIDE 增强定时器丢弃/重测被中断污染的 Prime+Probe 测量后提升到 81%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Prime+Probe 于共享 dTLB 的测量流程：
```
# 对每个候选内核地址 slot（共 32768 个）：
for slot in candidates:
    Prime:    访问 priming 地址集，把 dTLB 条目填满
    Spectre:  在系统调用内触发对 slot 的投机访问（对有效映射会产生 TLB 驱逐）
    Probe:    重访 priming 地址集并计时（counting-thread），被驱逐则重访慢
    判定:     probe 慢 ⇒ slot 有有效物理映射 ⇒ 内核基址候选
```
- 中断噪声破坏点：中断可能发生在 Prime 阶段（priming 条目被打乱）或投机执行阶段（内核上下文切换本身做大量 dTLB 访问），导致 probe 时间失真。TIDE 去噪：在测量前后各一次 x18 写/读，若 x18 被清零（本核被中断）则丢弃该 slot 测量并重做。剩余差距来自无中断的 dTLB 级噪声（视频播放本身的内存活动），无法由中断过滤消除。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：SysBumps 开源代码（论文原样部署、不做优化）；本论文在其上叠加 TIDE 增强 counting-thread 定时器。使用：KASLR 去随机化 → 为进一步内核攻击铺路。指标：成功率（100 runs 下 92% idle / 54% 噪声 / 81% TIDE 增强）。限制：噪声源不止中断（dTLB 干扰无法完全消除）；依赖共享 dTLB 的架构事实。

涉及论文标题：
- Towards Practical Interrupt Side-Channel Attacks on macOS for Apple Silicon
