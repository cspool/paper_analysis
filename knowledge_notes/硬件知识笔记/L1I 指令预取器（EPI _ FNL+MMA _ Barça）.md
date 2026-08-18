## L1I 指令预取器（EPI / FNL+MMA / Barça）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
L1I 预取器是缓解前端指令取指瓶颈的微架构机制：预测未来会被取指的指令块并提前取入 L1I 缓存，隐藏取指延迟。论文将其作为 IP-CaT 的评估基座，采用三种 SOTA 方案（均配置跨页预取）：①EPI（Entangling Prefetcher for Instructions，ISCA 2021，Ros & Jimborean）——"entangling"：把源行（触发行）与目标行（被预取行）配对存入 Entangled Table（60-bit 压缩编码、2-bit 置信度），命中源行时预取其 entangle 目标；用 timing table 记录 fill 时间戳计算 miss 延迟指导预取时机，用历史缓冲识别 entangle 对，缓存扩展处理 timely/wrong 预取；约 40KB 存储，IPC1 冠军衍生。②FNL+MMA（Seznec，IPC1/ISCA 2020）——FNL（Footprint Next Line）预测下一行近期是否会被用并预取（最多 5 行），MMA（Multiple Miss Ahead）用 I-Shadow 缓存（tag-only）预测第 n 个未来 miss；约 96KB 存储，IPC1 上 28.7% speedup。③Barça（Branch Agnostic Region Searching Algorithm，Jiménez et al.，IPC1 2020）——分支无关的区域搜索预取器。论文中三者在 server workload 上精度/覆盖率：EPI 74.1%/85.8%、FNL+MMA 72.9%/85.0%、Barça 67.7%/83.7%。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
L1I 预取器在硬件中的运转（以 EPI 为例）：取指产生 L1I 访问 → 命中则置 access bit；miss 时记录 fill 时间戳，计算取指延迟；缓存行 eviction 时若 access bit 未置 → wrong/early 预取 → 降置信度，否则 timely → 升置信度；fill 时在历史缓冲识别 source-entangled 行，把 (source,dest) 对存入 Entangled Table；每次访问命中 Entangled Table 中的行 → 预取整个基本块 + 置信度>0 的 entangle 目标块；基本块大小按需更新，准连续块合并（ABCECD→ABCD）。L1I 预取器在虚拟地址域工作，因此能发起跨页预取并间接预取翻译到 TLB 层次。评估平台：ChampSim（类 Intel Sunny Cove 前端）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：均为指令预取锦标赛（IPC-1）衍生的开源实现，部署在 ChampSim 的 branch/prefetch 模块；EPI 源码公开（作者主页）、FNL+MMA 见 HAL 文档（https://inria.hal.science/hal-02884880v1）、Barça 见 IPC1 程序页（https://research.ece.ncsu.edu/ipc/）。使用方式：作为 L1I 预取器基线测试新前端优化（如 IP-CaT 评估三种预取器以证明普适性）；配置跨页预取（permit page cross）以放大预取覆盖。局限：三种预取器均受益于更低的翻译延迟与更优的 L2C 预取行管理——这正是 IP-CaT 提供的改进空间。

涉及论文标题：
- Enhancing Instruction Prefetching via Cache and TLB Management
