## LH-cache（Loh & Hill 大块 die-stacked DRAM cache 组织）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LH-cache 是 Loh & Hill 在 MICRO-44'11（"Efficiently Enabling Conventional Block Sizes for Very Large Die-stacked DRAM Caches"，DOI 10.1145/2155620.2155673）提出的 die-stacked DRAM cache 组织，解决"超大 DRAM cache 的 tag 存储"问题：1GB DRAM cache 用 64B 块需约 96MB 片上 SRAM tag，不现实；把 tag 放 DRAM 则每次命中要两次完整 DRAM 访问。LH-cache 的关键是 tag 与 data 存在同一个物理 DRAM row：2KB row buffer 容纳 32 个 64B cache line，partition 为 29 个 data blocks + 3 个 tag blocks（约 48bit/entry）；配合 (1) Compound Access Scheduling——把 tag 与 data 访问安排成 compound access，数据访问恒为 row buffer hit（约 1.5× 而非 2× 延迟）；(2) MissMap——约 2MB 片上 SRAM 精确回答"该块是否存在"（36-bit tag + 64-bit vector，约 16.7 万项，精确无假阳/假阴），miss 时免读 DRAM tag。结果显示 1GB stacked DRAM cache + 2MB MissMap 比朴素 tag+data-in-DRAM 快 29–67%，达到理想 SRAM-tag 方案的 88–97%（addendum 修正后为 50–60% 区间）。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- TDMSim 论文直接采用 LH-cache 作为 DRAM cache 的基准组织：2D DRAM cache 16-way、16 bank，单 DRAM row 构成 cache set，cell 分区为多个 way（若干 tag ways + 若干 data ways）。运转流程：tag 访问时控制器先发 activation+read 把整 row 载入 row buffer → 读 tag ways 判定命中 → 命中则保留 row buffer 防止被其他请求关闭 → data ways 命中即 row buffer hit 直接返回；2D 材料低泄漏使 refresh 周期长达 0.5s（2D-1T1C），干扰率近零。Retention-aware 策略还在 LH-cache 的 way 粒度上做文章：把 retention 弱的边缘 tier 合并为 cyclic ways（仅存 clean block）、tag 严格映射到非循环 way。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：DRAM 阵列内按 row 组织 tag+data 分区，片上 SRAM MissMap 过滤 miss 的 tag 访问，compound access scheduling 保证命中路径 row buffer hit；row buffer 命中保留策略由控制器维护。使用/评估：多在 gem5/模拟器中建模（TDMSim 在 gem5 MI300X 模型里扩展 DRAM cache 组件按 LH-cache 组织运行）；评估指标为命中延迟、带宽、能耗。论文原文 https://research.cs.wisc.edu/multifacet/papers/micro11_missmap_talk.pdf，addendum 见 https://research.cs.wisc.edu/multifacet/papers/micro11_missmap_addendum.pdf。
涉及论文标题：
- TDMSim: Enabling High-Density and Energy-Efficient GPU DRAM Caches with 2D-Materials for Data-Intensive Applications
