## Pointer Authentication（PA/PAC，指针认证码）

术语解释
Armv8.3-A 引入的指针完整性机制：用轻量分组密码对"指针值 + modifier + 域密钥"生成截断 MAC（PAC），嵌入 64-bit 指针未用高位（通常 7–16 bit，Apple M1 为 16 bit），解引用/返回时硬件校验；LIPPEN 以它作为 baseline 并保持其 ISA/编译器/ABI 兼容。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PAC = Pointer Authentication Code：把指针当作消息，用密钥 K（存于系统寄存器、用户态不可见、按进程随机化）与 modifier（上下文，如栈指针 SP，无需保密）经轻量分组密码（Arm 用 QARMA 类，Armv8.1-M PACBTI 用 32-bit PAC）生成短 MAC，塞进 64-bit 指针的未用高位；PACIA/PACIB 签名（A-key/B-key）、AUTIA/AUTIB 认证，认证失败在 Armv8.3 破坏指针高位使其不可用、Armv8.6 抛同步异常。Web 证据：Aalto SSG 的 ARMv8.3-A PA 资料（https://ssg.aalto.fi/wp-content/uploads/2019/09/HARP_2019.pdf）与 Arm 文档（https://developer.arm.com/documentation/109576/0100/Pointer-Authentication-Code/Instructions）。
- 关键性质：零额外内存开销（复用指针本身的高位）、NOP 兼容 hint 空间（旧硬件可跑）、编译/ABI 无缝（Clang/LLVM `-msign-return-address` 自动生成 PACIASP/AUTIASP，Apple arm64e、Android、Windows on Arm、Linux 内核均已部署）。用途：保护返回地址（RETAA/RETAB 融合）、函数指针、vtable、间接分支目标（PLT 用 PACIA1716/AUTIA1716）。
- 局限（LIPPEN 的动机）：① 认证码须塞进未用地址位，有效熵 <24 bit（Apple M1 16 bit），暴力破解可行——PACMAN 用投机执行 PAC oracle 在 M1 上约 2.94 分钟穷举出合法 PAC；② 为保留原始地址位参与分支预测与调试，牺牲了加密空间；③ 截断 MAC 在 Spectre 类投机路径下可充当 oracle（PACMAN、Speculative ROP）。相关研究堆栈：PARTS（PAC-CFI）、PACStack、PTAuth、PACMem、AOS、PacTight、RSTI 等在其上扩展类型/时空安全。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件流程（一次 arm64e 函数调用与返回）：PACIASP 在 prologue 取返回地址（LR）与 modifier（SP），经硬件认证单元（密钥存系统寄存器，Apple M1 按 boot/EL/VM 派生密钥）生成 16-bit PAC 嵌入高位写回栈 → 攻击者栈溢出改返回地址 → epilogue AUTIASP 用 SP 重算 PAC 比对，不匹配则高位破坏/异常（程序崩溃而非劫持）；RETAA 把认证与返回融合（认证通过才跳转）。数据指针用 LDRAA/LDRAB（认证+加载融合）。Web 证据：PAC 生成/指令细节见 Aalto SSG 与 Arm 文档。
- 与 LIPPEN 的对照：PAC 保留原始地址位（认证码只占高位），LIPPEN 用全部 64 bit 做加密——PAC 的暴力空间 2^16 vs LIPPEN 2^64。LIPPEN 的 ISA（PTR_SEAL/PTR_UNSEAL/SET_KEY/SET_M_SIZE）语义对齐 PAC*/AUT*，故 PacTight 等 PAC 编译器基础设施可直接复用（移植 <50 行）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 硬件实现：Armv8.3-A 处理器内认证单元（分组密码 datapath + 密钥寄存器），指令在 NOP/hint 空间编码；研究原型用 FPGA 在 RoCC 上挂 QARMA 引擎实现 PAC baseline（LIPPEN 的 Rocket-PAC/BOOM-PAC 配置，RoCC 2,071 LUT、Fmax 89 MHz）。
- 使用方式：编译器插桩（Clang/LLVM/GCC `-msign-return-address`，arm64e 默认）或 LLVM IR pass（如 PacTight）插入签名/认证指令；系统侧配置密钥（按进程/域）；Apple 用户态需禁用 SIP 才能使用 PAC 指令（artifact 注意事项）。

涉及论文标题：
- LIPPEN: A Lightweight In-Place Pointer Encryption Architecture for Pointer Integrity
