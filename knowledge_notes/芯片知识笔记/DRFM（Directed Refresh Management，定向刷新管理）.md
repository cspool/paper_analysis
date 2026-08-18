## DRFM（Directed Refresh Management，定向刷新管理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- DRFM 是 DDR5（JEDEC JESD79-5C）引入的 Rowhammer 缓解命令，首次让内存控制器把"刷哪些 victim 行"委托给 DRAM 芯片（芯片内部行拓扑保密）：控制器先经 bank 级命令（precharge-per-bank PREPB、带 auto-precharge 的 read/write 等）的专用位"capture"aggressor 行身份，再发 DRFMab（all-bank）或 DRFMsb（same-bank），DRAM 随后刷新该 aggressor 行 blast radius 内的 victim 行。DDR5 同时定义 BRC（Bounded/Blast Radius Refresh Configuration）：对 DRFM 引发的 victim 行做部分概率性刷新（some rows always、some rows probabilistically），以抑制"半双倍"式传递攻击（DRFM 自身成为 R±2 行的锤击源）。Sigries 论文补充：DDR5 规定 DRFM 命令受速率限制——平均对同一 bank/row 地址不得少于 7.8µs 一次；发布 DRFM 前控制器必须先跟踪或采样 row activations 识别 aggressor。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（Sigries 视角）：内存控制器每 sub-bank 维护计数器表 → 某行计数器达 Rowhammer 阈值 → 控制器对该行地址发 DRFM → DRAM 内部按 BRC 配置刷新该 aggressor 的 victim 邻行（Sigries 同时把计数器复位并置 lock bit）→ 控制器用 per-bank DRFM 地址表跟踪近期发过的地址以遵守 7.8µs 速率限制。Sigries 评估把"DRFM 造成的 DRAM 带宽开销"作为核心指标：采样类防御（PARA/PRA/DREAM-R）对 commodity workload 也持续发 DRFM 产生恒定带宽开销，而 Sigries/Graphene/PRAC 无攻击时零 DRFM。DREAM（ISCA 2025）量测：DRFMsb ~240ns、DRFMab ~280ns，会停顿多个 bank（RLP 8–32），DRFMab 路径可致 49–82% 减速。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：JEDEC DDR5 标准命令，控制器侧需 aggressor 识别（跟踪/采样）+ capture + 速率限制簿记；DRAM 侧按 BRC 做概率化 victim 刷新。Web 来源：Stefan Saroiu 的 DRFM 笔记 https://stefan.t8k2.com/rh/DRFM/index.html （capture/DRFMab/DRFMsb 语义、BRC 歧义、无强制发布时机、传递攻击风险）；RAMPART（ISCA 2024，BRC-VL 变体，https://arxiv.org/abs/2310.16354）；DREAM（ISCA 2025，https://dl.acm.org/doi/10.1145/3695053.3731117）。相关研究把 DRFM 与 RFM（PRAC 的刷新管理命令）区分：DRFM 延迟更高（240ns vs RFM 190ns），消耗更多带宽。

涉及论文标题：
- From Lab to Fleet: Building and Deploying a Practical Rowhammer Defense in Cloud SoCs
