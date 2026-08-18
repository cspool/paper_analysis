## AES-GCM（Galois/Counter Mode，认证加密算法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- AES-GCM 是 NIST 标准（SP 800-38D）的认证加密（AEAD）模式：CTR 模式用 AES 分组加密生成密钥流（pad/keystream）做机密性，GHASH（GF(2^128) 上的 Galois 域乘法累加）生成认证标签做完整性。IroKnight 选用它作为加密与认证算法，理由：广泛部署（NVIDIA H100/H200 机密计算、AMD SEV、Intel AES-NI）、TEE 文献常用、NIST 认可。在 IroKnight 中 AES-GCM 承担两个角色：(1) 机密性——每周期产生 128-bit pad（=AES 加密 (地址,版本号,计数器) 的结果），与数据 XOR 完成加解密（见 Pad/PadGen）；(2) 完整性——GHASH 的 ⟨GF-multiply, XOR⟩ 认证（见 Mid-Execution Authentication）。AES-GCM 要求每个 ⟨地址,值⟩ 用唯一 pad（nonce/计数器 + 版本号），这正是 IroKnight 维护 per-tile/per-register 版本号的原因。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - IroKnight 中 AES-GCM 的硬件化：PadGen 把 (地址, key, 版本号) 推进 10 轮 AES 类变换（key addition 用 AES 密钥扩展出的 round key、S-box 替换、行移位、列混合）产出 128-bit pad，pad 直接连到 ALU 输入/输出 XOR，不落存储（明文与 pad 都只在组合逻辑内存在）；认证侧把 128×128 GHASH 乘法器拆成与操作数位宽匹配的 128×8 小乘法器沿脉动阵列分布，XOR 累加构成镜像认证路径。为防 AES-GCM 计算 key 泄露，计算用 key 与通信用 key 隔离：模型/数据 owner 经 RSA 私密通道发送的 AES-GCM key 只在通信时用，计算 key 由片上 key 生成单元直接写入 PadGens、物理上无观测路径；首列脉动阵列会把接收数据重加密为计算 key，结果输出前再重加密回 owner 的 key。版本号溢出前由 AES-GCM key 生成单元预生成新 key 保持 pad 唯一性。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 通用实现：软件（OpenSSL/libcrypto、Intel AES-NI + PCLMULQDQ、ARMv8 Crypto Extensions）、或硬件引擎（NVIDIA 机密计算、TEE 内存加密引擎在片外内存接口做 AES-XTS/GCM 加解密）。IroKnight 的用法特殊：不在内存接口集中加解密，而是把加解密铺进每个 ALU 的输入输出（同周期 XOR）+ 把 GHASH 认证铺进 MACC 镜像路径——这是"认证算术与 GEMM 同构"的直接产物。使用场景：需要加密所有权 + 篡改检测的 LLM 推理 serving（云上模型参数与用户数据都保密的场景）；IroKnight 用它达到 LLM 端到端 3.3% 运行时 / 15% 能量开销，对比 FHE（1110×/2872×）与 TEE（不保有所有权）。


LÆGIS 补充视角（ISCA'26，GPU CC 下 UVM 页迁移的 AES-GCM 使用）：LÆGIS 剖析 GPU-based CC 的 AES-GCM 用法——UVM 页迁移跨 CPU-GPU 信任边界时，每个 4 KB 页用 AES-GCM 加密（NVIDIA CC 对 CPU-GPU 互联的保护采用 CME 变体 AES-GCM），IV 由访问顺序隐式派生（IV_t ← increment(IV_{t-1})），不显式存 IV/MAC（避免完整性树开销），但要求 CPU 与 GPU 对称同步访问序、加密只能同步执行且与计算无法重叠。加密路径为 kernel-space：nvidia-uvm 的 CSL（Cryptography Services Library）→ libspdm → Linux Kernel Crypto API（crypto_aead_encrypt），单线程串行、实测 1.3 GB/s（2.98 µs/4KB 页）；GPU 侧 CE/GSP/SEC2 硬件 AES 引擎做解密。LÆGIS 改变 IV 管理：在 HBM 内显式存 per-VABlock 的 IV Bank Entry（19-bit ID + 77-bit RV），加密公式 AES(K_h2d, RV||ID||blkidx||0^15) 生成 OTP，解密侧 CE 构造 128-bit 输入经 AES 引擎产生 OTP 异或——OTP 生成与访问顺序解耦、可乱序/提前，且不再需要完整性树（HBM 可信假设）。

涉及论文标题：
- IroKnight: Ownership-Preserving Neural Acceleration for Inference Serving
- LÆGIS: Pinpointing and Addressing Performance Overheads of GPU-based Confidential Computing
