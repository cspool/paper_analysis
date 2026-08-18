## Intel TDX（Trust Domain Extensions，信任域扩展）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Intel TDX 是 Intel 的 CPU 机密计算技术：实现 VM 级隔离的 TEE，把整个虚拟机（含 OS/驱动）封装为信任域（Trust Domain, TD），由轻量软件组件 TDX Module（可视为精简 hypervisor）管理，实现 insecure world 到 secure world 的转换而无需修改客户代码。TD 私密内存由内存控制器中的 TME-MK 引擎用 AES-XTS 加密（TME-MK 提供 per-key 保护，多密钥），未授权访问返回零。LÆGIS 中 TDX 是 GPU CC 的 CPU 侧信任基板：nvidia-uvm 驱动与 GPU runtime 都运行在 TD 内，TDX 页在 UVM 迁移加密前先被 TME-MK 解密。
MC-ORAM 中 TDX 是 ORAM 的承载平台：TDX 隔离 VM（Ubuntu 22.04.5 guest）内运行 PathORAM/RingORAM 客户端与服务器，ORAM 树/暂存/位置图全部驻留 TD 私密内存（TME-MK 确定性 AES-XTS 加密）。TDX 的确定性加密"同一物理地址同一明文→同一密文"是密文侧信道根源，MC-ORAM 以 112 位掩码+16 位计数器恢复非确定性（带宽仅 1.125×），实测端到端最高 1.82× 加速；评估平台为双路 Intel Xeon 6548Y+、512GB DDR5。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
TDX 在 LÆGIS 平台中的运转：TDX 1.5 + QEMU 7.2.0（TDX patched）→ TD 内运行 Linux 6.2 + NVIDIA driver（550.163.01）→ GPU 经 passthrough 附属于 TD → UVM 页迁移时，CPU 侧 driver 读取 TDX 私密内存页（TME-MK 硬件 AES-XTS 自动解密，仅约 40 cycle 延迟）→ 再按 CC 要求做 AES-GCM 加密 → 经 PCIe 传给 GPU。TME-MK 引擎在 GPU kernel 执行期间大多空闲，LÆGIS 的 X-LÆGIS 变体建议复用该引擎（其 AES-XTS 与 AES-GCM 同 Galois 域，重配置即可支持 AES-GCM，40-cycle fill、16 B/cycle）作为 CPU 侧硬件 AES 加速。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Intel 处理器（Sapphire Rapids 起）BIOS 开启 TME/TME-MT/TDX + SEAM Loader；软件栈含 TDX Module、QEMU/KVM 支持、tdx-tools（Intel 开源，https://github.com/intel/tdx-tools）；客户内核需 TDX 补丁。使用：confidential VM 承载敏感工作负载（AI 推理/数据库），与 CC GPU 组合（如 H100）做 GPU 机密计算。Web 证据：Canonical TDX 仓库文档化 TDX + NVIDIA H100 passthrough 配置（https://github.com/canonical/tdx）。

涉及论文标题：
- LÆGIS: Pinpointing and Addressing Performance Overheads of GPU-based Confidential Computing
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs
