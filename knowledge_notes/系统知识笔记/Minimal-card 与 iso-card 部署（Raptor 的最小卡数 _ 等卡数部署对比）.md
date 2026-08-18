## Minimal-card 与 iso-card 部署（Raptor 的最小卡数 / 等卡数部署对比）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Raptor（ISCA'26）评估内存基板（XPU+SRAM / XPU+HBM / RP+3D-DRAM）的两种部署口径：minimal-card 用能同时装下模型权重与 KV cache 的最少卡数（在给定 TP/PP/DP 及 MoE 的 EP 配置下），暴露各基板在物理部署边界的固有容量限制与性能；iso-card 把所有配置的卡数固定为 3D-DRAM 在 minimal-card 下的卡数，做容量无关的公平比较。动机：minimal-card 下 HBM 常因容量大用更少卡、造成 tok/s/card 偏低（卡数不同无法公平比 batch 效应），因此引入 iso-card 隔离卡数影响。dense 模型用 unified 部署（TP≤8、PP 仅单卡装不下时加卡、DP 增加并发序列），MoE 用 disaggregated 部署（TP=4 注意力组 + EP 专家池），每模型-基板配置见论文 Table II（格式 ⟨Attn-TP|FFN-TP|EP|SE|PP, mode, memGB⟩）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 Raptor 评估流程中：对每个模型（Llama3.1-70B、GPT_OSS 20B/120B、DeepSeekV3-671B、Kimi K2 1T、Canary 1B、Whisper）与每个内存基板（SRAM 150TB/s/4GB、HBM 18TB/s/192GB、3D-DRAM 100TB/s/32GB 及 2×/4× BW、2×/4× Full）确定最小卡数与并行配置（例：Llama3.1-70B 在 SRAM 需 8 卡 TP=8（128GB）、HBM 1 卡 TP=1（192GB）、3D-DRAM 1 卡 TP=1（32GB）；Kimi K2 1T 在 SRAM 需 1,584 卡（6336GB）、3D-DRAM 4 卡（1728GB））→ 扫 batch 得 tok/s/card vs interactivity(1/TPOT) 曲线（图 14，minimal-card；图 17，iso-card）→ 结果：minimal-card 下 HBM 用更少卡但带宽低拖累 tok/s/card；iso-card 下 HBM 用更大 batch 改善 tok/s/card，但中小 batch 仍受带宽限制、3D-DRAM 领先直到计算饱和。iso-card 排除 SRAM（容量无法扩展到所需卡数）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：部署规划阶段按模型权重+KV cache 足迹与每卡容量选最小卡数，用 TP/PP/EP/DP 组合覆盖；iso-card 只是把对比配置强制对齐到 3D-DRAM 的 minimal-card 卡数（batch 上限随之变化）。使用方式：minimal-card 回答"该内存基板物理上能支撑什么性能"，iso-card 回答"同卡数下带宽/容量谁占优"；网络延迟/带宽敏感性分析（0.01-10µs / 32GB/s-4TB/s，现实点 0.5µs/1TB/s）在 minimal-card 下进行。价值：把"容量→卡数→并行度→collective 量→网络敏感性"的因果链显式化——3D-DRAM 每卡 32GB 使 Llama-70B 单卡 TP=1，比 SRAM 的 8 卡 TP=8 collective 小得多，故对网络延迟/带宽不敏感（4K 上下文现实网络 4.38× vs HBM、3.15× vs SRAM）。
