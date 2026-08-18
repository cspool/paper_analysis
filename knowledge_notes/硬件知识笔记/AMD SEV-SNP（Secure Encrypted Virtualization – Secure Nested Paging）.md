## AMD SEV-SNP（Secure Encrypted Virtualization – Secure Nested Paging）

术语解释
- AMD SEV-SNP 是 AMD EPYC 的 VM-based TEE：SEV 加密 VM 内存，SNP（Secure Nested Paging）增加完整性保护与 VM 间/管理程序隔离，与 Intel TDX 并列的机密虚拟机方案。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SEV 用 AMD 安全处理器管理密钥，对每个 VM 内存加密；SNP 引入 RMP（Reverse Map）表与嵌套页表完整性，阻止管理程序重映射/重放攻击，并支持远程认证（ATTESTATION 指令）。SEV-SNP 的内存加密同样为确定性 AES-XTS（物理地址 tweak），与 TDX 相同的密文侧信道问题。
- 本论文将 SEV-SNP 与 TDX 并列为设计目标："operate efficiently within TDX- and SNP-style TEEs"——MC-ORAM 机制只依赖 TME 式确定性加密，对厂商透明。实验仅在 TDX 上进行（论文未在 SEV-SNP 实测，为信息缺口）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
SEV-SNP 机密 VM 中一次 ORAM 块写回（设计兼容）：
CPU 计算掩码数据||计数器 → 写访存 → 内存控制器 AMD AES-XTS 加密
→ 加密 DRAM（攻击者从总线嗅探只见密文）
→ 读回解密，TEE 内处理掩码/计数器/刷新
```
- 例子：MC-ORAM 声称可部署于 SNP 式 TEE 的 PathORAM/RingORAM，但论文未给出 SEV-SNP 实测数据；LÆGIS 论文引用 AMD 官方白皮书描述 SNP 的 VM 隔离与完整性保护。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：AMD EPYC 安全处理器（密钥管理）+ 内存控制器 AES-XTS 引擎 + 固件（AMD 白皮书《SEV-SNP: Strengthening VM Isolation with Integrity Protection and More》）；配合 KVM/管理程序支持。
- 使用：云机密 VM、可信数据库、ORAM 等；本文为设计目标平台之一（未实测）。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs
