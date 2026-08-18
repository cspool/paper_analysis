## NIST SP 800-22 统计测试套件（NIST Statistical Test Suite，RNG 统计评估）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NIST SP 800-22 是由美国国家标准与技术研究院发布的"随机与伪随机数发生器密码学应用统计测试套件"，包含 15 项统计测试，从多个分析视角评估位序列的统计质量与不可预测性：Monobit、Frequency Within a Block、Runs、Longest-Run-of-Ones in a Block、Binary Matrix Rank、Discrete Fourier Transform、Non-Overlapping/Overlapping Template Matching、Universal Statistical、Linear Complexity、Serial、Approximate Entropy、Cumulative Sums、Random Excursions（及变体）。没有单一测试能完全量化随机性（看似非确定可能源于理解不足，真随机过程也可能因偶然而显示低熵），因此需要多视角组合。行业标准要求 RNG 通过 NIST 认证——但认证分析的是长连续序列，这恰好掩盖间歇系统短上电窗口内的熵缺陷。其他常用套件：Diehard/Dieharder、TestU01。本论文指出：这些测试的诊断可靠性强烈依赖数据如何采集、切分、拼接；µRNG 的核心观点是"仅在连续采集模式下通过 NIST 不足以保证安全"。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
µRNG 把 NIST SP 800-22 作为 5 阶段框架的第 2、3 阶段评估工具：阶段 2 连续采集——设备持续上电收集长位流过 15 项测试（名义条件）；阶段 3 间歇采集——每次电源循环取 RNG 首次输出高 16 bit、65,536 次循环拼接 128 KB 位流再过 NIST（模拟 boot-time entropy hole 下的密钥生成）。运转流程：DUT 供电 → 采集固件读 RNG 寄存器 → 按设备选存储策略（片上 NVM 行写入/MSPM0 shadow 地址规避 ECC/小 NVM 设备 UART 外传，2^16 样本 × 3s 循环最长 55 小时）→ 拼接 → NIST 15 项判定 pass/fail。关键实证：所有被测设备在连续模式下都通过 NIST（第一阶段全过），但阶段 3-5 暴露真实弱点——SAM L10/L11 在 +215°C 时 15 项中 8 项失败；这证明传统测试方法在间歇环境产生虚假安全感。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：NIST 官方 STS 工具（https://doi.org/10.6028/NIST.SP.800-22r1a）或开源实现（如 Python 的 randomness 库、R 的 randtoolbox）；输入为位流文件，输出每测试的 p-value 与 pass/fail 判定（通常按显著性水平与比例判定）。使用：认证流程、RNG 出厂测试、本论文式研究评估。局限（本论文明确）：依赖数据采集/切分/拼接方式；连续长流无法暴露间歇熵损失；对环境条件不敏感。因此 µRNG 用"采样条件与数据组织方式最极端化"补充传统测试——变环境、间歇采集、弱点发现测试（碰撞/熵/Moran's I）——揭示传统方法遗漏的缺陷。

涉及论文标题：
- μRNG: A Framework for Assessing Randomness in Intermittent Computing Devices
