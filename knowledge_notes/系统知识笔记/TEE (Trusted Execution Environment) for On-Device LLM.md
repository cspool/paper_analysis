## TEE (Trusted Execution Environment) for On-Device LLM

术语是什么？通过联网搜索让回答具体和精准。

TEE（Trusted Execution Environment，可信执行环境）是处理器内的安全执行区域，通过硬件隔离确保代码和数据在运行时免受主机操作系统或其他应用程序的访问和篡改。在移动端LLM场景中，TEE保护模型权重（专有IP）、用户数据和推理中间结果（activation、KV cache）的机密性。ARM TrustZone是最广泛部署的移动TEE硬件基础，将ARM处理器划分为Secure World（运行TEE OS和TA）和Normal World（运行REE OS如Android/Linux）。与纯软件加密（仅加密模型文件，inference时明文参数在内存中出现）不同，TEE通过硬件级隔离确保明文参数仅出现在TEE保护的secure memory中。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

TEE在TZ-LLM中的系统架构层次：

```
┌────────────────── Normal World (REE) ──────────────────┐
│  OpenHarmony v4.1 / Linux v5.10                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ REE NPU Driver (Control Plane)                    │  │
│  │  - Job scheduling / Power-Freq management         │  │
│  │  - Shadow job queue processing                    │  │
│  ├──────────────────────────────────────────────────┤  │
│  │ TZ Driver (CMA allocation/deallocation, +197 LoC) │  │
│  ├──────────────────────────────────────────────────┤  │
│  │ Encrypted Model Storage (NVMe SSD)                │  │
│  └──────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────┘
                        │ SMC (Secure Monitor Call)
┌───────────────────────┴─────────────────────────────────┐
│ Secure World (TEE)                                      │
│  OpenHarmony TEE OS (~17K LoC + ~112 LoC ext)          │
│  ┌──────────────────────────────────────────────────┐  │
│  │ LLM TA (llama.cpp + extensions, ~2.2K LoC)        │  │
│  │  - Pipelined parameter restoration scheduler      │  │
│  │  - OpenSSL AES decryption                         │  │
│  │  - NPU data plane driver (~1K LoC)                 │  │
│  ├──────────────────────────────────────────────────┤  │
│  │ TZASC/TZPC/GIC Configuration (~50 LoC)            │  │
│  ├──────────────────────────────────────────────────┤  │
│  │ Secure Memory (CMA + TZASC-protected)              │  │
│  │  Decrypted parameters / activations / KV cache     │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

TZ-LLM执行流程中TEE的关键作用：
1. **启动**：REE应用触发LLM请求→通过TEE Client API发送请求→LLM TA在Secure World启动
2. **参数加载**：TA通知REE从NVMe SSD读取加密参数→DMA直接到allocated未保护内存→TA调用extend_protected→TEE OS配置TZASC→内存成为secure memory→TA在此解密
3. **推理执行**：TA在secure memory中执行CPU/NPU推理，所有明文参数、activation、KV cache受TZASC硬件保护
4. **结果返回**：仅推理结果文本从TEE传回REE，明文模型参数永不离开Secure World

TEE在mobile LLM场景的特殊挑战：(1)动态大容量内存——LLM参数可达数GB，传统TEE面向KB-MB级TA；(2)加速器共享——NPU需低开销低TCB的time-sharing；(3)冷启动延迟——TZ-LLM的pipelined restoration和co-driver NPU分别解决(1)(3)和(2)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

主流TEE实现：(1) ARM TrustZone + OP-TEE/QTEE/OpenHarmony TEE——移动和嵌入式设备；(2) Intel SGX/TDX——x86服务器，SGX保护应用级enclave，TDX保护完整VM；(3) AMD SEV/SEV-SNP——保护VM；(4) NVIDIA Confidential Computing——GPU TEE。TZ-LLM基于TrustZone+OpenHarmony TEE。安全模型局限（论文明确列出）：不防护物理DRAM攻击、side-channel、密码分析攻击、DoS。参数tensor size和NPU job执行时间向REE泄漏模型结构级信息。

涉及论文标题：
- TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone
