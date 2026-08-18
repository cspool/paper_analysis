## PRNG（Pseudo Random Number Generator，伪随机数发生器）与 CTR-DRBG（计数器模式确定性随机位发生器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PRNG 是用确定性算法（哈希、流密码、线性同余等）把小的高质量种子扩展为长"看似随机"序列的模块，其安全完全依赖种子的保密性与不可预测性。实践上 TRNG 与 PRNG 结合（如 Linux /dev/urandom）：TRNG 提供少量熵，PRNG 拉伸出大量密钥。CTR-DRBG（Counter Mode Deterministic Random Bit Generator）是 NIST SP 800-90A 标准定义的 DRBG 构造之一（§10.2.1），以 AES 为底层分组密码、用计数器模式（对递增计数器加密）生成伪随机输出；参数上 AES-256 实例安全强度 256 bit、种子 384 bit（32B key + 16B counter/V）、重播种间隔 2^48、单次最大请求 2^19 bit。本论文评估的 TI MSP430FR59x/69x 家族用硬件 AES 加速器实现 CTR-DRBG（应用报告 SLAA725），作为 PRNG 为密钥生成服务。注意"PRNG"在论文中指代两类：① 无熵源的纯软件 PRNG（C 库 rand() 的 LCG、SAM D21 的 Yasmarang），种子固定或来自 system uptime，间歇下必然重复（Class 1）；② 有固定设备级种子 + AES-CTR 的 CTR-DRBG（Class 3）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
以 MSP430FR59x/69x 的 CTR-DRBG 为例，其硬件-软件协同运转流程：① API 调用时检查 .infoD FRAM 中的 8-bit instantiated_flag（0xAA 表示已实例化）；② 未实例化则用设备描述符 TLV 区（0x1A30 固定 128-bit 种子 + 0x1A0A 64-bit nonce，生产时烧写）调用 AES 加速器，扩展出 128-bit 密钥与 128-bit 数据块构成 256-bit 工作状态；③ ctr_drbg_update() 对数据块两次增量加密刷新工作状态并写回 .infoD（FRAM 非易失，掉电保持）；④ 生成随机字节 = 对递增数据块用密钥增量加密直至达到请求长度，再 update 刷新。硬件架构上的安全缺陷：TLV 种子/ nonce 对共驻软件与外部接口（debugger/DMA）可读；.infoD 工作状态与 flag 可读可写——攻击者可重建或操控 PRNG 状态，甚至打补丁重置 flag 强制重新实例化使设备重复相同序列（论文用此构造 Diffie-Hellman 共享密钥恢复攻击）。PRNG 若持续被 TRNG 高熵输入重新播种则具备预测抵抗力，但本论文中的固定种子实现不具备。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：软件 LCG（rand()、Yasmarang）零硬件；MSP430 类用片内 AES 加速器（HW AES）做 CTR 模式流密码；通用嵌入式/服务端实现为 mbedtls ctr_drbg（默认 AES-256 + 派生函数，熵长度 SHA-512 48B/SHA-256 32B，默认重播种间隔 10000）、Go crypto/internal/fips140/drbg（AES-256 无派生、384-bit 附加输入）等开源库。使用方式：初始化（instantiate，熵+nonce+personalization）→ 按需生成（generate，输出并 update 状态）→ 定期重播种（reseed，混入新熵）；NIST 建议重播种间隔内生成量受限。对本论文场景（间歇计算）：纯软件 PRNG 因种子随电源循环重置必然重复（SAM D21 uptime 种子误差约 100ns 产生 121 个重复 DH 密钥）；FRAM 保持状态的 CTR-DRBG 掉电续存但固定种子可被读取/篡改。论文建议：用 TRNG 持续重新播种维持跨电源循环的不可预测性，且 PRNG 状态/种子必须受内存隔离保护。

涉及论文标题：
- μRNG: A Framework for Assessing Randomness in Intermittent Computing Devices
