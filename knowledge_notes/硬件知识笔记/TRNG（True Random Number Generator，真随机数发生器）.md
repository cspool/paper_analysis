## TRNG（True Random Number Generator，真随机数发生器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TRNG 是从非确定性物理过程（熵源）采样生成不可预测位序列的硬件模块，是密码学随机数的源头。理想 TRNG 输出满足两个性质：位相互独立（看到全部历史也无法猜测下一位）且均匀分布。其核心链条是：物理噪声源（模拟域）→ 采样/数字化（转化为数字位）→ 可选后处理/条件化（压缩去偏）→ 输出位流。数字电路天然确定性，因此 TRNG 必须从模拟域噪声（RO 抖动、SRAM 上电态、热噪声）或量子/混沌过程取熵；由于噪声源对环境（温度、电压、电磁干扰）敏感，攻击者可操控环境影响输出质量。本论文 µRNG 强调：TRNG 的条件化（post-processing）只是确定性地掩盖底层熵缺陷，因此评估必须捕获后处理之前的原始 TRNG 输出（ground-truth configuration）才能反映真实强度。本论文评估的片上 TRNG 包括：MSPM0 L-Series（Johnson-Nyquist 热噪声 + delta-sigma 调制器 + 内部 LDO + 流密码条件化，Class 5 推荐）、SAM L10/L11（32-bit/84 cycles 内存映射寄存器 TRNG_DATA，Class 2）、Apollo4（BSI AIS-31 / NIST SP 800-90B 合规黑盒，实际为 TRNG 种子 + AES-CTR PRNG 的混合结构，Class 3/4）；片外 TRNG 如 SAM L10/L11 Xplained Pro 的 ATECC508A 加密加速器（FIPS 合规 RNG，但总线可被嗅探）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
TRNG 在 SoC/MCU 中作为外设存在，运转流程（以 MSPM0 L-Series 为例）：① 模拟噪声源（电阻的 Johnson-Nyquist 热噪声电压波动）→ ② delta-sigma 调制器数字化 → ③ 内部 LDO（Low-Dropout Regulator）为噪声源单独供电以抵抗电源操控攻击（外部 buck 转换器/电感在攻击者物理可及的路径上可被旁路或插值）→ ④ 流密码条件化模块（decimation 1-8，bitwise XOR，论文以 rate=1 分析最弱条件化输出）→ ⑤ 输出 32-bit 存入内存映射寄存器 trng->DATA_CAPTURE，全部硬件实现无软件介入 → ⑥ 启动/持续健康测试（NIST SP 800-90B 风格），失败触发 ERROR 状态停机。重要硬件细节：启动时预置测试模式使首个 32-bit 字确定化必须丢弃，间歇供电下吞吐量减半；内部 LDO 阻止了对电源相关效应的直接测试。相比之下 SAM L10/L11 的 TRNG 用 84 个周期产 32-bit 随机数写入 TRNG_DATA，无文档说明熵源/条件化，其对环境的敏感性（温度↑碰撞↑，+215°C 时 15 项 NIST 失败 8 项）暴露了缺少电压调理与条件化的后果。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：模拟噪声源（热噪声/RO jitter/SRAM 上电态）→ 采样电路（比较器/调制器/振荡器计数）→ 数字条件化（流密码、XOR 树、哈希）→ 内存映射寄存器或 API 暴露。硬件厂商实现多样：TI MSPM0 用热噪声+调制器+LDO，SAM L10/L11 用 RO（推测，文档未披露），Ambiq Apollo4 用黑盒 TRNG（仅声明 BSI AIS-31 / NIST SP 800-90B 合规，无架构细节）。使用方式：固件通过厂商 RNG API 或直接读寄存器取随机数；安全设计者需验证 TRNG 在间歇供电与环境极端下仍满足熵需求——µRNG 框架要求捕获原始输出、连续与间歇两种采集、多环境 corner 重测、弱点发现测试（碰撞/熵/Moran's I）后才能分级（Class 1-5）。论文结论性建议：TRNG 必须片上（防总线嗅探）、稳压必须全片上 LDO（防攻击者插值）、熵累积与蒸馏应硬件实现、软件请求应阻塞直到累积 ≥128 bit 真随机。

涉及论文标题：
- μRNG: A Framework for Assessing Randomness in Intermittent Computing Devices
