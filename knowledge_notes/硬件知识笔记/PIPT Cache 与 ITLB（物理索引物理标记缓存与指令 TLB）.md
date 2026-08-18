## PIPT Cache 与 ITLB（物理索引物理标记缓存与指令 TLB）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PIPT（Physically Indexed, Physically Tagged）缓存用物理地址做索引与 tag 比较，彻底避免 VIPT 的同义词/别名（synonym/alias）问题，但代价是每次访问前必须完成地址翻译。Bumper 的关键约束即由此而来：L2C 是 PIPT 结构，而指令退休时通常只有 VA 没有 PA（PA 需要查页表）——因此不能从 retire 阶段直接向 L2C 发 hint。Bumper 的解法：retire 把 VA 送回 IFU（Fig.9 ②），IFU 把 Hint Request 存入 8-entry Hint Lookup Queue（HL1Q，③），在 FTQ 请求未占满 ITLB 带宽时机会式访问 ITLB 完成 VA→PA 翻译（④）。基线 ITLB 256-entry、DTLB 224-entry、L2TLB 4096-entry。实测 <0.1% 的 hint 请求在 ITLB/L1I/L2C miss（这些结构足够大，能覆盖"取指到首条指令退休"之间的信息存活期）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Bumper hint 请求的地址转换流程：retire VA 入 HL1Q → 机会式占用 ITLB 空闲带宽查页表项得 PA（miss 则正常页表 walk，几乎不发生）→ 以 PA 访问 L1I tag（命中且 send_hint=1 则清除并继续）→ 入 8-entry HL2Q（⑦）与 LSU/数据预取请求仲裁 → PA 访问 L2C 提升 RRPV=0（⑧）。机会式访问保证 hint 翻译不抢占 FTQ 需求请求的关键带宽；HL1Q/HL2Q 各 8 项（43B×2）即可发挥 Bumper 全部潜力。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：PIPT 常见于大容量末级缓存（L2/LLC），小容量 L1 常用 VIPT 换取并行性；ITLB/DTLB 缓存页表项（支持 4K/2M 页），miss 时由硬件页表 walker 遍历多级页表（Bumper 基线：5 级 radix tree 页表 + MMU cache + 硬件 walker）。Bumper 的扩展点：L1I tag 的 send_hint 位与 L2C 填充响应的 l2_vulnerable_fill 标志（1 bit，不计入 422B 总存储）。

涉及论文标题：
- Bumper: Hinting Instruction Usefulness for Robust Unified Caches
