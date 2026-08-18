## TEE（Trusted Execution Environment，可信执行环境）

术语解释
- TEE 是处理器内硬件隔离的安全执行区域，运行时保护代码和数据免受主机 OS/管理程序/特权软件访问；实现分 enclave 级（Intel SGX）与 VM 级（Intel TDX、AMD SEV-SNP、ARM CCA），以及移动端 TrustZone 与 GPU 侧 NVIDIA Confidential Computing。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TEE 通过 CPU 硬件强制隔离（页表/安全状态/加密引擎）保证：即便 OS/管理程序被攻破，TEE 内代码与数据仍机密且不可篡改。本论文采用 VM-based TEE（TDX/SEV-SNP）：把 ORAM 客户端（位置图、stash、全部隐私逻辑）装进机密虚拟机，ORAM 服务器进程同机，两者之间流量不跨 WAN。TEE 的 TME 引擎对 DRAM 全加密，因此 ORAM 树/暂存可直接放 TEE 内存，省去客户端逐块重加密。
- 三方可信模型：①终端用户（数据所有者，边缘设备）②TEE 内 ORAM 客户端 ③不可信 ORAM 服务器。用户只发块地址、只收目标块明文。威胁模型（论文 II）：CPU 包/缓存/TEE 特性可信；敌手可观察 DRAM 访问模式与密文（stash/位置图/树）。局限：TEE 不能完全阻止微架构侧信道（用 oblivious 全暂存扫描缓解）与确定性加密导致的密文侧信道（MC-ORAM 解决）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
用户(边缘) --块地址--> [TEE 内 ORAM 客户端(位置图/stash/掩码计数器)]
                           |  oblivious 路径请求（整路径形状固定）
                           v
                    [ORAM 服务器进程(不可信, 同机)]
                           |  DRAM 密文存取
                           v
              [内存控制器 TME 引擎: AES-XTS(addr, 128位块)]
                           |  TEE 加密 DRAM（树/暂存/位置图）
```
- 例子（本论文实验平台）：双路 Intel Xeon 6548Y+、512GB DDR5，host/guest 均 Ubuntu 22.04.5，guest 为 Intel TDX 机密 VM；ORAM 客户端与服务器同机运行，每次逻辑访问在 TEE 内完成位置图查询（递归 ORAM）+路径读+全暂存扫描+驱逐。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：主流包括 ARM TrustZone+OP-TEE（移动/嵌入式）、Intel SGX（应用级 enclave）、Intel TDX/AMD SEV-SNP/ARM CCA（VM 级）、NVIDIA CC（GPU TEE）。MC-ORAM 面向 TDX/SNP 式 VM-based TEE，TCB 增加 <200 行（掩码/计数器/刷新管理）。
- 使用：云端安全计算（ORAM/可信数据库/机密 LLM 推理等）；本论文用于承载 ORAM 客户端与加密存储。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs
