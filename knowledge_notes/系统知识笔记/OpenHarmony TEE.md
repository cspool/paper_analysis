## OpenHarmony TEE

术语是什么？通过联网搜索让回答具体和精准。

OpenHarmony TEE是OpenHarmony（华为发起、开放原子开源基金会管理的开源分布式操作系统）中的TEE子系统，为TA（Trusted Application）提供基于ARM TrustZone硬件隔离的安全运行时环境。提供TA生命周期管理、安全存储、加密服务、安全驱动框架等。TZ-LLM选择OpenHarmony TEE作为TEE OS平台进行原型开发，主要因为其与论文使用的硬件平台（RK3588开发板）和REE OS（OpenHarmony v4.1 / Linux v5.10）有更好的集成支持。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

OpenHarmony TEE在TZ-LLM中的核心交互流程：

1. **TEE OS Kernel**（~17K LoC原始，+112 LoC扩展）：提供TA调度、secure memory管理、IPC、加密服务。扩展包括CMA page memory mapping（~62 LoC，将REE CMA分配的物理页映射入TEE地址空间）和动态TZASC/TZPC配置（~50 LoC）。

2. **TA运行时**：LLM TA基于llama.cpp，运行在TEE user-mode，经TEE OS授权访问安全资源。

3. **REE-TEE交互**：
   - REE侧TZ driver（+197 LoC）：处理CMA allocation/deallocation，响应TEE的secure memory扩展/收缩
   - 编程接口：GlobalPlatform TEE Client API（TEEC_InitializeContext→TEEC_OpenSession→TEEC_InvokeCommand）
   - CMA分配路径：TA调用extend_allocated→TEE OS通过SMC转发请求到REE→REE TZ driver调用Linux CMA API→返回物理地址给TEE→TEE OS映射入TA地址空间

4. **安全服务**：OpenHarmony TEE提供OpenSSL（AES参数解密）、安全随机数、安全时钟等标准TEE服务。

TZ-LLM在TEE OS内核侧修改极小（~112 LoC），大部分代码放在TA user-mode（~2.2K LoC llama.cpp扩展 + ~1K LoC NPU data plane driver），遵循最小化TCB的安全原则。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

OpenHarmony由开放原子开源基金会管理，TEE是其安全子系统之一。与其他移动TEE OS（OP-TEE开源、Qualcomm QTEE闭源）相比，OpenHarmony TEE与OpenHarmony生态深度集成。TZ-LLM基于OpenHarmony v4.1及其TEE系统实现原型。

涉及论文标题：
- TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone
