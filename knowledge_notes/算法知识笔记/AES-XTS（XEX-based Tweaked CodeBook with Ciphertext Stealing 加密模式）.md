## AES-XTS（XEX-based Tweaked CodeBook with Ciphertext Stealing 加密模式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- AES-XTS 是 NIST SP 800-38E 标准化的可微调块密码模式（tweakable block cipher mode），广泛用于存储与内存加密（AMD SME、Intel TDX、Microsoft BitLocker 等）：用两个 AES 密钥（Key1 加密 tweak、Key2 加密数据块），tweak 取数据地址/扇区号，使相同明文在不同位置产生不同密文。MANATEE 用它加密 NVM 页（64B 页 = 4 个 16B XTS 块）。
- MANATEE 选 AES-XTS 而非 AES-CTR 的原因：AES-CTR 依赖 counter 新鲜度保证保密性，需完整性树（integrity tree，如 Bonsai Merkle Tree）防止 replay 攻击，而完整性树在 EHS 上能量开销巨大（论文指出 CME + 完整性验证相对 MANATEE 约 50× 慢）；AES-XTS 天然免 replay（每个 tweak/位置独立）——论文强调它"无需完整性树即保证数据保密性"，且比 AES-CTR 更强的抗单比特翻转等攻击（虽然不能防篡改，因无完整性验证）。
- MC-ORAM 语境（TEE 内存加密）：Intel TDX/AMD SEV-SNP/ARM CCA 用 TME（Total Memory Encryption）引擎对 DRAM 做确定性 AES-XTS 加密，C = AES_XTS(addr_128, D_128)，tweak=物理地址、无任何 nonce 元数据——高效但确定性：同一物理地址写同一明文产生同一密文，形成**密文侧信道**（攻击者经内存总线嗅探 DRAM 密文可检测"值重复或变化"）。该泄漏破坏 ORAM 的不可区分性（树/暂存内容不变→密文不变，暴露暂存占用与 dummy 位置，区分优势可达 1/4）。MC-ORAM 用 112 位随机掩码+16 位计数器保证同一物理地址两次访问的加密前 128 位值不同（同掩码周期概率 1、跨周期 1−2^−112），从而让确定性 AES-XTS 密文每次访问都变化。
从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程（加密一个 16B 块 P_i，位于页偏移 i，tweak = 页地址派生）：
```
T_i = AES_Key1(tweak) ・ alpha^i        # GF(2^128) 中乘 alpha 派生 tweak
C_i = AES_Key2(P_i XOR T_i) XOR T_i     # XEX 结构
# 64B 页 = 4 个块：连续加密 4 个 16B 块，SPM 内缓冲凑齐后一次原子写 NVM
```
- 例子：页 P3 被驱逐/断电，page manager 取页基地址作 tweak，逐块 AES_Key2 加密，4 块凑 64B 原子 flush；读回时同 tweak 解密。MSP430FR5994 提供 AES 加速器/库，MANATEE 用其实现 AES-XTS。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：利用 MSP430FR5994 的 AES 加速器/库（先于本文的方案 [55,57] 同样使用）；页粒度加密（64B）摊销加解密开销，使加解密频率远低于字级全加密。性能分解显示 CRC32 这类写密集负载加解密占 ~74% 执行时间。论文未给出公开代码，无法确认是否开源。
涉及论文标题：
- Intermittence-aware Speculative Page Coloring for Secure NVM
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs
