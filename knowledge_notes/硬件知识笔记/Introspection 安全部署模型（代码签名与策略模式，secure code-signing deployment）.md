## Introspection 安全部署模型（代码签名与策略模式，secure code-signing deployment）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
IPU 的安全部署模型是让"app-store 式内省程序分发"成为可能的机制：证书式代码签名 + 公/私钥基础设施（类似 Android/iOS app 商店），保证只有可信 introspection 程序（芯片设计者、研究人员、开发者撰写）能部署到 IPU 上执行。背景逻辑链：IPU 能读芯片内部微架构信号、运行可编程分析，若被滥用会泄露未公开的 HIT 微架构细节（与 PICS/性能计数器的已知隐私问题同类，甚至更强），且新能力需要受控分发；同时架构上 IPU 被设计为"只能读不能写"——不能向 HIT 注入信号，只接受签名二进制并输出内省数据，从物理上限制了恶意利用面。运行时整套系统呈现为单个 PCIe 设备（每 IPU 独立内存映射区域），主机 API 下载二进制并配置触发逻辑。三种策略模式在签名时强制执行：closed（禁止第三方内省，芯片设计者独占）、restrictive（第三方程序需源码审查）、permissive（仅凭证开发者）。未来方向包括程序验证（Bouncer 类静态分析）、trace wringing、本地差分隐私、安全多方计算等更强保证。隐私层面：只流式输出内省数据、不访问主机状态（如 IP/MAC），与现有性能计数器风险相当；SKU 变体或 boot 配置可完全禁用 IPU（对齐行业对 JTAG 的处理）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
安全部署在 IPU 系统架构中的运转流程：开发者用 RISC-V 工具链写 introspection 程序 → 用其私钥签名二进制（绑定设备 ID 与策略模式）→ 发布到内省程序"应用商店" → 主机侧运行用户程序，调用 IPU_CONFIG_IMAGE(image) 把二进制+触发元数据经 OCN/PCIe 发给指定硬件设备 ID 的 IPU → IPU 侧验证签名与策略（closed 模式拒绝第三方；restrictive 要求源码审查通过）→ 通过后加载二进制到指令内存（8KB，2048 条指令）→ 用户程序执行期间 IPU 按触发逻辑运行分析，只输出内省数据（FIFO 到主机内存区域），不能读主机状态也不能写回 HIT。部署后策略模式固定于签名，无法绕过；P&R 时 IPU flatten 进 HIT 层级，安全边界由物理层级+签名共同保证。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：证书链代码签名（论文明确类比 iOS/Android app 签名机制，参考 Apple 平台安全指南与 Android app signing 文档），IPU 内含签名验证逻辑，策略模式编码进签名证书；作为对照，现代 CPU/GPU 的类似机制包括安全飞地/机密模式（如 Intel SGX、NVIDIA 机密计算）与 Intel 签名固件。使用方式：closed 策略最常用于芯片设计者（如论文 D 节的 CPU 核设计者自己写 PICS 程序，ABI Spec 可不公开）；restrictive/permissive 面向生态（研究人员/第三方开发者经审查或凭证部署）；不可信部署中 IPU 被完全禁用（SKU 变体或 boot 配置），符合行业规范。局限：论文承认侧信道风险与形式化保证（程序验证、差分隐私、MPC）留作未来工作，当前风险与现有性能计数器相当；供应商-用户信任问题非 IPU 独有。

涉及论文标题：
- Enabling Continuous, In-Field Introspection: The Programmable IPU Architecture
