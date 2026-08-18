## CAM 匹配与 overlap filter（RST 并行替换电路）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- RST 压缩器 substitution module 的三个逻辑步骤（Fig.12）：① CAM（Content-Addressable Memory）匹配——并行找出最新选中序列在页数据中的所有出现位置，产生 match bitvector；② 替换——把这些出现替换为选中序列的字典索引（替换位点 substitution sites）；③ compaction——用 scatter-gather 把被消耗的符号移除。CAM 匹配让"整页找全出现"并行完成，是 RST 每轮压缩的基础。
- 从硬件架构角度拆解术语：当两个选中序列出现位置过近（重复周期 < 序列长度的 run，如 XXXXXX 或 XYZXYZXYZ）时，并行匹配电路会误报出中间的额外假匹配（overlapping-match problem，Fig.13a，类似流水线处理器的数据冒险）。正确性要求丢弃部分候选匹配，且"该丢哪些"依赖邻居、依赖可沿整个输入传播——必须组合式解析这些长距离依赖而不串行扫描。论文的关键观察：run 内重叠匹配的距离重复出现，序列长度仅 2–5、重叠只发生在小于序列长度的距离上，所以只有 10 种重叠 pattern；overlap filter 放在 CAM 与替换之间：识别 run 起止、检查 run 起始处的 pattern、在整段 run 上应用对应的重复 mask（Fig.13b），过滤出正确的不重叠位置，即使输入全是 run 也保持单周期吞吐。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 电路实现：CAM 由多行比较器组成（3-symbol 情形见 Fig.12；推广到其他长度只需加比较器行）；filter 在 run 起始处判定 pattern 并广播 mask。综合配置以 256-symbol 硬件 chunk 处理（匹配/compaction 宽度 4096→256，跨 chunk 边界序列少且少为 top-utility，压缩率仅损失 2.32%），合成后 substitution 路径流水化，流水线填满后每周期一个完整输入。与 LZ 硬件（IBM/CDPU 的串行匹配引擎）不同，RST 需要全并行匹配以支撑每页 >3×10^5 次操作的迭代压缩。
涉及论文标题：
- Random-Access Hardware Sequence Compression
