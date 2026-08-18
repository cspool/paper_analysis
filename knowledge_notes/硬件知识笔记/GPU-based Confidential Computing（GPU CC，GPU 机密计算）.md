## GPU-based Confidential Computing（GPU CC，GPU 机密计算）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GPU-based CC 是把 CPU 可信执行环境（TEE，如 Intel TDX、AMD SEV-SNP）与 CC-capable GPU（如 NVIDIA H100/H200）结合的机密计算范式，保护"使用中的数据"（data in use）：CPU 侧 VM 级隔离（TD/CVM）包住整个 GPU runtime 与 kernel driver，GPU 进入 CC 模式后与 TD 通过加密通道通信。通信保护：CVM 与 GPU 之间共享内存区域用 AES-GCM 会话密钥保护（密钥经远程认证后建立），H100 在 CC 模式下阻断直接 DMA 访问并禁用性能计数器（侧信道防御）；GPU 侧 HBM 采用"可信"假设、不运行时加密。代价是显著性能开销（论文 [5] ISPASS'25、[6] PipeLLM ASPLOS'25 均研究），LÆGIS 定位其 UVM 路径的加密关键路径开销。相关概念：CPU 侧 TD 私密内存由 TME-MK 引擎加密（AES-XTS），GPU 侧 CE/GSP/SEC2 集成硬件 AES 引擎，片上 fuse 存安全密钥。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
LÆGIS 的 GPU CC 平台架构（TDX + H100）：CPU 端 TDX 信任域（TDX Module 作为轻量 hypervisor）→ 用户 CUDA 运行时调用 /dev/nvidia-uvm 驱动 → GPU 端：SM 执行 → GMMU 页表走查 → 缺页触发 fault → GMMU 聚合 fault buffer → 中断 CPU → driver 服务批次：对每个 4 KB 页做 AES-GCM 加密（TDX 页先被 TME-MK AES-XTS 解密）→ CE copy 命令入 push buffer → 经未受信 PCIe 通道（64 GB/s）DMA → GPU CE 用硬件 AES 解密 → HBM 明文存储。密钥建立（1）后即进入该数据面流程；控制面（provisioning/CC 服务）与数据面（CPU-GPU 数据移动加密）分离。加密引擎：GSP/SEC2 是 RISC-V 微控制器集成硬件 AES；GPU 解密可流水隐藏，CPU 侧加密是软件瓶颈（1.3 GB/s）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：NVIDIA Confidential Computing（H100/H200，CUDA 12.2+）+ Intel TDX/AMD SEV-SNP 宿主，经 TEE-IO/PCIe IDE 等安全通道（论文不假设新协议，仅用现有平台资源）。使用：云上敏感 AI 推理/训练（CC 容器 + GPU passthrough），经远程认证（attestation）建立信任后分发密钥；研究界以 GPGPU-Sim 等模拟 CC 行为。Web 证据：NVIDIA/Edgeless Systems 文档确认 H100 是首款全面 CC 特性 GPU、CVM-GPU 共享内存经 AES-GCM 会话密钥保护且 H100 不运行时加密显存（https://www.edgeless.systems/wiki/hardware/nvidia-hopper-h100 ；Intel TDX+H100 白皮书 https://cdrdv2.intel.com/v1/dl/getContent/917058 ）。

涉及论文标题：
- LÆGIS: Pinpointing and Addressing Performance Overheads of GPU-based Confidential Computing
