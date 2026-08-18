## RowPress（长时激活行扰动 / Long Activation-Time Disturbance）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- RowPress（ISCA 2023，Luo et al.，ETH Zürich，获 ISCA 杰出 artifact 奖与 IEEE Micro Top Pick 2024）是与 RowHammer 不同的 DRAM 读干扰现象：把攻击行**保持开启长时间**（长激活时间 tAggON）即可诱导相邻 victim 行位翻转，而非像 RowHammer 那样反复激活-预充电。物理机制：开启的 wordline（"passing gate effect"）持续吸引 victim 单元电荷。实测（164 颗 DDR4、三家主要厂商）：诱发翻转所需激活数 ACmin 比 RowHammer 降 1–2 个数量级——极端情况下**单次激活**（ACmin=1）只要把行保持约 30ms 就能翻转；tAggON 从 36ns 升到 7.8µs 时 ACmin 降约 21×（部分芯片 191×）；温度升高加剧（80°C vs 50°C 时所需激活约少 50%）。Sigries 论文（ISCA'26）表述：RowPress 在行被保持开启很长时间时发生，需要的行激活远少于 RowHammer——"即使单次激活、把行保持开启数十毫秒（远超 DDR5 协议限制）也可能触发"。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 芯片层面：攻击者（用户态程序，可经访问同一行不同 cache block 保持行开启）对 aggressor 行发 ACT 并长时间不 PRE → 该行 wordline 持续高电平 → 相邻行单元经电荷耦合/漏电逐渐翻转。运转例子：Sigries 给出的实际缓解是**closed-page 策略**——每访问后立即预充电关闭行，限制行保持开启时长（tMRO，max row open time 限制在内存控制器实现）；这比按"激活次数"计数的 RowHammer 防御更本质地切掉 RowPress 的攻击向量，因为 RowPress 的核心是"时长"而非"次数"。TRR 等按激活计数的防御对单次激活+长保持的 RowPress 天然盲区。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 缓解实现：(1) 内存控制器限制最大行开启时间 tMRO（closed-page 策略，即 Sigries 所述方法）；(2) 把现有 RowHammer 防御按 RowPress 降低后的 ACmin 重新配置阈值。开源实现与数据：https://github.com/CMU-SAFARI/RowPress（artifact 可复现）、扩展版 arXiv:2306.17061、Zenodo artifact https://zenodo.org/records/7768005 。评估沿用 RowHammer 的方法论（ACmin 表征、Hammer 模式、刷新窗口内计数），但关注参数从"激活次数"扩展为"激活次数 × 保持时长"二维。

涉及论文标题：
- From Lab to Fleet: Building and Deploying a Practical Rowhammer Defense in Cloud SoCs
