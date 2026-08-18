## Johnson-Nyquist 热噪声熵源（Thermal / Electrical Noise TRNG）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Johnson-Nyquist 热噪声（thermal noise）是导体中载流子因黑体辐射引起的随机热运动产生的电压/电流波动，幅度与温度、电阻正相关（电阻上的噪声电压 PSD = 4kTR，k 玻尔兹曼常数、T 温度、R 电阻），是嵌入式 TRNG 捕获的主要电噪声（另有 shot noise、flicker noise）。TRNG 通过跨阻性元件捕获该电压波动并数字化来产生随机数；该噪声也存在于其他模拟域过程（如 RO 的时序 jitter）。由于热噪声主要取决于温度与元件电阻，环境温度变化直接改变可用熵，但也提供稳定且难以外部强制的物理随机性来源——前提是电压/温度扰动不破坏采样电路。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
热噪声 TRNG 硬件运转流程（MSPM0 L-Series 即此类型）：① 电阻/结产生的 Johnson-Nyquist 噪声电压 → ② delta-sigma 调制器数字化（高过采样把微弱噪声累积成数字流）→ ③ 内部专用 LDO 为噪声源与采样电路供电（隔离电源操控，抵抗攻击者电压注入）→ ④ 条件化模块（论文称"reportedly stream-cipher-based"）+ 可配置 decimation（1-8，XOR 抽取，论文用 rate=1 分析最弱条件化输出）→ ⑤ 32-bit 结果入 trng->DATA_CAPTURE。硬件级健康测试（启动预置模式 + 持续监测）检测熵损失/模块故障，失败进入 ERROR 停机。µRNG 实测：该 TRNG 在全部温度（-68°C~+85°C）与电压 corner 下连续与间歇输出均过全部 NIST 测试且无统计弱点——归因于热噪声对环境的鲁棒性 + LDO 隔离 + 条件化；但内部 LDO 使供电相关效应无法直接测试，且启动确定性首字使间歇吞吐减半。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：片上电阻/二极管结 + 放大器 + ADC（delta-sigma 或 SAR）+ 数字条件化 + 健康测试；关键设计点：专用稳压（LDO）隔离电源扰动、条件化压缩去偏、decimation 权衡吞吐与熵（decimation=4 时 ~1ms/2048 bit @8MHz，论文推荐）。使用：作为密码学随机数源头连续/间歇供熵；评估需覆盖环境 corner（论文验证其全 corner 通过）与间歇采集模式。相较 RO 与 SRAM 熵源，热噪声熵源对本论文测试的环境操控最鲁棒，MSPM0 L-Series 因此是 189 台中唯一 Class 5（推荐）平台。

涉及论文标题：
- μRNG: A Framework for Assessing Randomness in Intermittent Computing Devices
