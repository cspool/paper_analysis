## Early Line Fetch（提前取行，tagged store 微架构优化）

术语解释
AmpereOne core 为缓解 MTE SYNC 模式 store 开销的微架构优化：对 tagged store，cache-line（含其 memory tag）的取回在地址翻译阶段就发起（如同普通 load），使 tag 校验远早于 commit 完成，并可与其他工作乱序重叠。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
背景：SYNC 模式下 store 必须先取回并校验目标 cache line 的 memory tag 才能 commit（不匹配要 fault、访问不完成），这是 core 内 MTE 主导成本；若取回等到 store 执行才发起，会串行阻塞 store 提交。其根源（内核社区 Jessica Clarke 的分析）：store 无法在知道目标位置 tag 前 retire，必须完成完整 cache lookup（甚至 miss 取回）才知道是否 trap，任何 write-allocate/write-around 方案都无法绕过"必须提前取 line（及其 tag）"。AmpereOne 的 early line fetch 把 line+tag 取回提前到地址翻译时发起（与普通 load 的取回时机相同），允许 out-of-order 重叠——tag 校验可远早于 commit 完成，多数 store 无额外延迟。load 侧几乎无开销：tag 校验并入现有 cache-lookup 路径、与地址翻译/权限检查并行，不引入额外 pipeline stage/stall。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
流水时序例子（store 指令）：译码 → 地址翻译（此处即发起目标 cache line 含 tag 的取回）→ 取回期间乱序执行其它指令（overlap）→ line+tag 到达、tag 校验完成（早于 commit 阶段）→ store 提交。相比"store 执行时才取回"的串行路径，提前取回把 tag 校验延迟从关键路径移出；当 tag 未命中 DRAM 时，提前量等于乱序窗口。注意该机制与 store-to-load forwarding 协同（见下条）：转发需 address tag 匹配，tag 校验结果未定时靠"同验同错"不变量保正确。Web 佐证（ARM MTE Performance in Practice）：Ampere SYNC 下 store 因需提前取 tag，行为类似 load——执行期把数据+tag 都取进扩展 cache line；但首代 Ampere 的 store-to-load forwarding 与 MTE 交互仍有实现缺口（SPEC 456.hmmer 最高 1.43×），已在新一代硅片修复。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在 load/store 单元的地址翻译阶段为 tagged store 注入 line 取回请求（带 tag），复用现有 miss 处理与 MSHR；store buffer 记录 allocation tag 支持转发规则。首代 AmpereOne 的剩余硬件开销：store-to-load forwarding 机会减少 + tag 读流量增加 L1D 读端口压力（结构冲突），两者已在新一代核心修复；对比 ARM 参考实现（Pixel 8/9 Cortex-X）SYNC store 串行化最高 6.64× 悬崖，early line fetch 使 Ampere 数据中心负载 C/B 中个位数百分比。内核侧配套（Ampere 2025 补丁）：Linux 7.0 收紧 PSTATE.TCO 处理、减少内核空间多余 tag 检查，memcached 25–50% 内核开销降到可忽略。

涉及论文标题：
- Optimized Memory Tagging on AmpereOne® Processors
