## Variable Retention Time（VRT，可变保持时间弱单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VRT 是 DRAM 单元保持时间在高低两个状态间随机来回切换的现象：单元进入亚标称驻留状态后，在调度刷新前泄漏掉电荷产生保持性错误。机理：门致漏电（GIDL）受硅中亚稳态陷阱态波动驱动，随工艺缩微（sub-20nm）与高温（车载 125°C）加剧。关键特征：(1) 单向性——电荷泄漏使充电单元 1→0 翻转（DDR4 实测 0→1 翻转仅占 0.005%），即 retention 失效是单向的；(2) 随机且持续——现代芯片平均 2GB 内 15 分钟窗口有 350–500 个 Active-VRT 单元、约每 15 分钟出现 1 个新单元，无法在生产测试中完全筛除；(3) 概率随 scaling 上升——现代工艺单元进入亚标称状态概率 >10⁻⁸ 且持续增加。论文用 Samsung DDR5 on-die ECC 实测数据推导 VRT 率下界：启用 on-die ECC 使刷新时间延长 >4×、观察位错误率降 10⁻⁶×，据此解方程得每 bit VRT 驻留错误概率 X≈1.4×10⁻⁸（无 ECC 单比特失败概率 ×10⁻⁶ = 双比特失败概率），并在评估中向 10⁻⁵ 扫描以覆盖 scaling 趋势。Web 佐证：Wikipedia 的 variable retention time 条目、AVATAR（DSN'15）把 VRT 与多速率刷新 + ECC 结合提升可靠性 ~100×。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
VRT 在芯片设计中的影响：迫使 on-die ECC 与刷新策略协同（DDR5 的 136 位码字 SEC、refresh 周期由最坏单元决定）。芯片设计运转流程例子（论文的 VRT 率推导）：写 128 位数据块 → 刷新窗口内某弱单元以概率 X 泄漏翻转（无 ECC 时失败概率 P_1bit = C(128,1)·X·(1−X)^127）；启用 on-die SEC 后失败需同码字内 ≥2 位失败（P_2bit = C(136,2)·X²·(1−X)^134），按工业报告"错误率降 10⁻⁶×"解出 X≈1.4×10⁻⁸。对 PIM 的影响（论文核心动机）：bank-PIM 的 PIM 访问走 detect-only CRC（不做本地纠错），VRT 错误会频繁触发 rank 级纠错 → 需 Codeword Flip 掩蔽（见 知识库_硬件架构.md"Codeword Flip"）；若不处理，VRT 10⁻⁵ 下 LLM 推理（Llama2 权重/KV-cache 注入 VRT 错误）精度接近随机猜测，而 reliable bank-PIM 全程保持无错误精度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：靠 on-die ECC（DDR5 SEC(136,128)、HBM3 RS(19,17)）在芯片内吸收单比特 VRT 错误；AVATAR 类方案做 VRT 感知多速率刷新（故障行升频刷新 + ECC）；生产上通过测试与 row/column 重映射筛除稳定弱单元但无法覆盖 VRT 动态性。论文做法：把 VRT 建模为每 bit 概率的独立均匀位失败（沿用 Patel et al. 经验方法、二项统计），在可靠性模拟器中按真实 DDR5 FIT 率与错误模式注入（含 VRT 与运行故障重叠），评估 SDC/DUE 与 Codeword Flip 的掩蔽效果。注意事项：VRT 率厂商不公开，需像论文一样用 on-die ECC 效果反推下界并做范围扫描。

涉及论文标题：
- ECC Enabled Reliable and Performant Processing-in-Memory
