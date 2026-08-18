## Page Retirement（页面退役，DRAM 运行故障的 host 侧处理策略）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Page Retirement 是操作系统/驱动层把物理内存页标记为不可用、从可用内存池移除的故障处理机制（与 PPR/内存控制器重映射、模块替换并列）。在可靠 bank-PIM 中，它用于防止可检测的多比特运行故障升级为 SDC：bank 级 detect-only CRC 检测弱于 rank 级 ECC，一旦观测到某些多比特运行故障，reliable bank-PIM 就比传统系统更激进地退役页面——具体规则：退役任何与超过一个故障逻辑行或列重叠的页面（即使该故障 rank 级仍可容忍）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
系统架构运转流程：rank 级纠错重试仍失败 → 系统在 host 端模拟执行 PIM 命令并触发内存退役 → 退役与 >1 个故障逻辑行/列重叠的页面 → 后续访问避开该页。论文的两个观察支撑该策略：① 值得退役的情形用 detect-only on-die CRC 很容易识别（这类故障在出现 SDC 前会反复触发 rank 级纠错）；② 退役仍然罕见——即便按保守策略退役整个 DRAM 模块而非单页，系统级 PIM 吞吐 5 年只下降 <2%（DDR5 预期故障率约 45 FIT/chip，数据来自 Jung & Erez MICRO'23 组件级故障模型）。退役由 host 处理，是"rank 级纠错 + Codeword Flip + 页面退役"三级可靠性流程中的最后防线。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：OS 内存子系统维护坏页表（bad page list），故障页从页分配器移除；reliable bank-PIM 中由 PIM 驱动/内存控制器把退役决策上抛 host 执行。使用要点：退役阈值要在"防 SDC"与"过度退役浪费容量"间权衡——论文取"重叠 >1 个故障行/列"（运行故障而非 VRT，VRT 由 Codeword Flip 掩蔽）；与 rank 级 ECC 配合时，重试 + 单比特纠错模式解码 + host 模拟执行构成完整升级路径（见 知识库_硬件架构.md"Two-tier ECC"与"Codeword Flip"条目）。

涉及论文标题：
- ECC Enabled Reliable and Performant Processing-in-Memory
