## μRNG: A Framework for Assessing Randomness in Intermittent Computing Devices

- baseline方法是什么？
  - baseline 是传统 RNG 安全性评估方法：用行业统计测试套件（NIST SP 800-22 的 15 项测试、Diehard/Dieharder、TestU01）在**连续供电、攻击者访问受限**的假设下，对 RNG 输出的一段长位流做统计分析，全部通过即视为合格（对应 FIPS 140-3 / NIST SP 800-90 / BSI AIS-31 等认证流程）。其内在假设是：设备持续上电、熵源有足够时间累积、采样条件与攻击者影响无关，因此只要位流统计上均匀就代表安全。早期/现有工作（Santoro、Rojas-Munoz 的在线监测，Fujdiak 的 MSP430 时钟抖动 RNG 表征，Bhattacharjee 等的 PRNG 对比）都停留在这一阶段。
  - baseline 全栈执行例子：
    ```
    算法pipeline层：论文未明确说明（无推理算法模型；RNG 输出为随机位流，非算法推理加速）；
    系统框架层：论文未明确说明（无 serving 框架；被测设备无 OS 或仅有裸机运行环境）；
    编译框架层：论文未明确说明（无编译框架；固件用厂商工具链编译后烧录）；
    kernel调度层：论文未明确说明（无 GPU kernel；RNG 是片上外设寄存器访问，非计算 kernel 调度）；
    硬件架构层：设备持续上电 → 熵源（RO jitter / SRAM 上电态 / 热噪声）持续累积 → 每收集足够熵后
               条件化/抽取输出 → 采集一段连续位流（如 NIST 要求的百万 bit 级）→ NIST SP 800-22 15 项
               测试全过 → 判定 RNG 合格。此流程忽略：间歇设备上电时间 <100ms、密钥生成发生在每次
               电源循环后熵尚未累积的 boot-time entropy hole、以及攻击者可操控供电与温湿度。
    ```
- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 方法：µRNG 是一个面向间歇计算场景的 5 阶段 RNG 评估框架，在传统统计测试之外显式建模间歇供电与环境攻击者影响。(1) ground-truth 配置发现——反汇编/打补丁获取硬件/软件后处理之前的**原始** RNG 输出（对应"TRNG 后处理只是确定性掩盖熵缺陷"的洞察），确保测的是真熵而非被条件化掩蔽的序列；(2) 连续模式 NIST 测试（保留 baseline 首阶段作对照）；(3) 间歇模式采集——每次电源循环后取 RNG **首次输出**高 16 bit，65536 次循环拼接 128 KB 位流再测试，直接复刻 boot-time entropy hole 下密钥生成的真实时机（对应"连续长位流掩盖间歇熵损失"的缺陷）；(4) 环境极端 corner——在攻击者可控制的温度（-68°C~+215°C）/电压/压摆率范围内逐 corner 重跑阶段 3（对应"传统测试不考虑环境对模拟域噪声源的影响"的缺陷）；(5) 弱点发现测试——NIST 之外补碰撞测试（生日悖论期望 vs 实测偏差）、min-entropy/Shannon 熵、Moran's I 空间自相关，检测环境-输出关联的非二元弱点（对应"pass/fail 二元判定掩盖可被强攻击者利用的弱相关"的缺陷）。据此把 RNG 分为 Insecure / Class 1-5，并给出硬件设计建议（片上 TRNG、全片上 LDO 稳压、硬件熵累积与蒸馏、128 bit 累积门限、混合 RO+SRAM 熵源等）。
  - 论文方法全栈执行例子（以 MSPM0 L-Series 设备评估为例）：
    ```
    算法pipeline层：论文未明确说明（无推理算法模型；框架是测试方法学而非推理加速）；
    系统框架层：论文未明确说明（无 serving 框架；被测设备无 OS）；
    编译框架层：论文未明确说明（无编译框架；仅需厂商工具链编译采集固件）；
    kernel调度层：论文未明确说明（无 GPU kernel；采集固件直接读 TRNG 内存映射寄存器 trng->DATA_CAPTURE，
               高 16 bit 写入 flash 行（ECC 下用 shadow NVM 地址空间），非计算 kernel 调度）；
    硬件架构层：电源控制器（TLP222AF 继电器 + MCP4725 DAC + Raspberry Pi 3B）每循环 2s 断电 + 1s 上电，
               共 65536 次；每次上电读 TRNG 首次输出高 16 bit（丢弃启动确定性首字）→ 拼接 128 KB 位流 →
               NIST SP 800-22 15 项测试 + 碰撞/熵/Moran's I 弱点测试；TE-123H 温箱逐 corner（-68°C/+25°C/
               +85°C）+ DAC 逐电压点重跑；输出分类 Class 5（唯一推荐：热噪声源 + delta-sigma 数字化 +
               内部 LDO 抗电源攻击 + 流密码条件化 + 健康测试，全部 corner NIST 通过且无弱点）。对比
               baseline：同一设备在连续模式同样全过 NIST，但 baseline 无法区分——SAM L10/L11 的 +215°C
               8/15 NIST 失败、MSP430FR59x/69x 的固定种子状态泄露、SAM D21 的 uptime 种子重复序列
               （121 个重复 DH 共享密钥）、Apollo4 的 TRNG 种子经 SRAM 截获，全部依赖阶段 3-5 才暴露。
    ```
