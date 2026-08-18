## TRR（Target Row Refresh，目标行刷新）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TRR 是厂商内置于 DRAM 芯片的 Rowhammer 缓解：芯片内部跟踪被反复激活的 aggressor 行，并触发对相邻 victim 行的刷新。实现细节专有且不公开（"security by obscurity"），仅靠其保证系统可靠性——没有它，良性负载（如 Intel 因在 DRAM 存 cache coherency 目录位产生的非恶意高激活率）也会诱发位翻转。Sigries 论文（ISCA'26）指出 TRR 的局限：研究反复证明其**多孔**——只能挡住某些 Rowhammer 攻击模式（TRRespass、Blacksmith、ZenHammer、SMASH 等持续破解），且 JEDEC 白皮书（JEP300-1 近端/JEP301-1 系统级）承认现有防御无法消除全部攻击，但不披露具体局限。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 芯片内部运转：DRAM 维护少量 aggressor 行候选（各厂商算法不同，常见基于组内行激活计数或概率），当某行激活频率超内部阈值时，在后续 REF 周期对候选 aggressor 的相邻行执行额外刷新。失效例子（为什么云厂商在控制器侧另加防御）：DDR4 TRR 有可被利用的盲区（TRRespass 2020 用 many-sided 模式；Blacksmith 2022 用非均匀模式）；DDR5 TRR 盲区更少但 Phoenix（IEEE S&P 2026）用自校正同步模式在 15/15 颗 DDR5（SK Hynix）上触发位翻转、109 秒提权（CVE-2025-6202）。ProTRR（ETH）提出带形式化安全保证的 in-DRAM TRR（Proactive Misra-Gries 计数），与 DDR5 RFM 兼容。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：DRAM 厂商 RTL 内部逻辑，无公开规范；系统侧无法配置或观察其阈值，只能以攻击实验逆向（U-TRR 用 retention 失效做侧信道，https://people.inf.ethz.ch/omutlu/pub/onur-Meta-RowHammer-28-September-2023.pdf）。使用场景：作为"无防御"到"控制器级防御"之间的最低层缓解；Sigries 等控制器级防御假设 TRR 不可信（威胁模型里攻击者知道 Sigries 配置、可通过 side-channel 推断 REF/DRFM 时机），在控制器侧提供可分析、可配置的叠加保护。

PrISM 视角（ISCA'26，Loaded Dice 论文）：概率缓解把 TRR 当作"免费"的默认缓解机会——每次周期性 refresh 期间（论文默认每 2 个 tREFI 提供一次 TRR 机会，TRR 借用刷新时间 tRFC，不额外停通道）对 PMQ 中最高计数行做缓解。TRR 频率是性能敏感参数（Fig.9，TRH-D=500）：每 tREFI 一次 TRR 时 PrISM slowdown <0.1%；无 TRR 时 PrISM 只能靠 proactive RFM（每 72 激活一次）→ slowdown 升到 3.2%（MINT 对应 1.2%→7.2%），而 PRAC 因开销主导在计时膨胀上保持 ~14% 不变。TRR 的有限容量（4–28 项 per-bank trackers）也是其被绕过（TRRespass/Blacksmith/ZenHammer/Phoenix）的原因，PrISM 在 TRR 之上叠加概率采样历史而非依赖其精确追踪。

涉及论文标题：
- From Lab to Fleet: Building and Deploying a Practical Rowhammer Defense in Cloud SoCs
- Loaded Dice: Solving the Non-Selection Problem for Scalable Probabilistic RowHammer Defense
