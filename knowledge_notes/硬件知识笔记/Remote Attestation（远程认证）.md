## Remote Attestation（远程认证）

术语解释
- 远程认证是 TEE 支持的安全协议：远程方在向 enclave/机密 VM 提供机密或敏感数据前，验证其软件与配置的完整性（证明运行的是预期代码），是"云上可信客户端"部署的前提。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 流程：TEE 启动装载代码 → 硬件（CPU/安全处理器）对代码与配置计算度量并签名 → 远程验证方获得签名报告 → 验证通过后 provision 密钥/数据。TDX（QUOTE/SEV-SNP ATTESTATION、Intel/AMD 签名）与 SGX（EINIT/Quote）均支持。本论文威胁模型（II）将其列为前提："允许远程方在提供机密前验证 enclave 内部软件与配置的完整性"。
- 与 ORAM：ORAM 客户端驻留云端 TEE 后，终端用户必须确信客户端实现（位置图/stash/掩码/计数器/刷新逻辑）未被篡改，才把敏感数据与块地址交给它。MC-ORAM 的 TCB 论证（VII-D）：新增可信代码 <200 行，相对 CVM 内 Linux 内核（~30M LoC）极小，因此远程认证的验证面可负担。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
部署时序（MC-ORAM 场景）：
1) TDX/SNP 机密 VM 启动，装载 ORAM 客户端（<1000 行 + <200 行 MC-ORAM）
2) 终端用户发起远程认证请求
3) 硬件对 VM 配置/代码度量并签名（TDX QUOTE / SNP ATTESTATION）
4) 用户验证报告 → 通过后 provision 位置图根/密钥/数据
5) 开始 ORAM 访问（位置图查询+路径读+全暂存扫描+驱逐）
```
- 例子：本论文假设 TEE 支持远程认证（论文 II），使 ORAM 客户端逻辑可安全驻留云端；认证后用户的每次逻辑访问只发块地址、只收目标块明文。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：硬件签名（CPU 安全扩展+密钥）+ 度量基础设施（TDX QUOTE、SEV-SNP ATTESTATION、SGX Quote）+ 验证服务；相关研究如 VRASED（验证的软硬件协同远程认证）、Sanctum 的 secure boot+remote attestation。
- 使用：机密计算部署的信任建立环节；本论文把它作为威胁模型前提（未深入实现细节，论文未明确说明其具体协议）。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs
