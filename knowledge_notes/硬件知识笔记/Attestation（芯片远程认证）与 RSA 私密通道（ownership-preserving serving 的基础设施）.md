## Attestation（芯片远程认证）与 RSA 私密通道（ownership-preserving serving 的基础设施）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Attestation（远程认证/证明）是让远程方（Model Provider、Data Owner）确信某芯片是真实、可信 IroKnight 实例的机制：验证方发随机 nonce 给芯片，芯片用出厂内置的私有 attestation key 签名返回，验证方用制造商公钥验签并核对 nonce 一致。RSA 私密通道是 IroKnight 在完全不可信基础设施（CPU、网络、系统软件均不可信）上安全分发 AES-GCM 计算 key 的方式：芯片与 Model Provider、Data Owner 各建立一条独立 RSA 公/私钥加密通道（私钥仅本地持有、绝不上云），通过通道交换 AES-GCM key，使模型参数与用户数据在传输与存储中始终以"私密 RSA 加密过的 AES-GCM key"加密。芯片间也以同样方式建立 pairwise RSA 私密通道交换 AES-GCM key，支持多 NPU 协同（tensor/expert parallel）交换加密中间数据。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - 硬件组件与流程：片上集成 RSA 与 AES-GCM key 生成单元（AES-GCM key 单元在初始与版本号溢出前生成新 key 保证 pad 唯一性）；attestation 用一台轻量 in-order CPU（无分支预测器/无 cache）创建并签名 attestation report，执行期关闭以降低攻击面。端到端流程（多芯片 serving）：逐芯片 attestation（Model Provider 发 nonce → 芯片签名 → 验签）→ Model Provider 发其 RSA 公钥 → 芯片用该公钥加密自己生成的 AES-GCM key 发回 → Model Provider 用 RSA 私钥解出 AES-GCM key → 用其加密模型参数传给各 NPU；Data Owner 走独立通道；NPU 两两间 attestation 后互换 RSA 公钥、互相用对方公钥加密自己的 AES-GCM key 完成 pairwise 交换。此后 IroKnight 像普通 NPU 一样工作（加密参数/数据流入、加密中间结果流出）。隔离计算 key：计算时不用通信 key，而是首列脉动阵列把接收数据用"从未离开过芯片"的计算 key 重加密，计算 key 由 key 生成单元直接写入 PadGens（物理上不可观测）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：RSA/AES-GCM key 生成单元、attestation 专用轻量 CPU、片上 key 存储/写入路径（无观测路径）均为 RTL 的一部分（IroKnight SystemVerilog + 综合验证）。通用背景：远程认证是 TEE/机密计算的标准组成（Intel SGX ECDSA attestation、AMD SEV-SNP、NVIDIA H100 CC attestation、TLS 的远程证明延伸）；密钥交换用非对称加密（RSA/ECDH）+ 对称会话密钥是标准做法。IroKnight 的特别处：所有基础设施（含 host CPU、网络）都在威胁模型内，因此 RSA 私钥绝不进入基础设施、AES-GCM key 经 RSA 私密通道分发且计算 key 与通信 key 物理隔离、attestation 每芯片重复。使用场景：让 IroKnight 成为 conventional NPU 的 drop-in 替换（attest + 分发 key 后即正常 serving）。

涉及论文标题：
- IroKnight: Ownership-Preserving Neural Acceleration for Inference Serving
