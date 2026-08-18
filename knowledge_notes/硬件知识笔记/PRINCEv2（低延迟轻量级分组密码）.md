## PRINCEv2（低延迟轻量级分组密码）

术语解释
PRINCE 族第 2 代 64-bit 分组密码（SAC 2020）：128-bit 密钥、FR-MR-BR 结构、12 轮（5+2+5）、为单周期 unrolled 硬件 datapath 设计，是 LIPPEN 指针全加密的底层密码（加密=解密结构、超短关键路径）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PRINCEv2 由 PRINCE 改进而来（Božilov et al., SAC 2020）：通过改轮常数、密钥调度与轮密钥插入方式，在不增加轮数与面积/延迟的前提下显著提升安全性——FR/MR/BR 三函数结构（MR 2 轮、FR/BR 各 1 轮，S-box 层计数 12），SP 型轮函数（4-bit S-box 并行 + AES 式 ShiftRows + 2 进制 4×4 MixColumns）。Web 证据：CASA/RUB 页面（https://casa.rub.de/en/research/publications/detail/princev2-more-security-for-almost-no-overhead）与 SAC 2020（DOI 10.1007/978-3-030-81652-0_19）。
- 安全性：PRINCE 存在全轮 reflection attack；PRINCEv2 声称数据复杂度 ≤2^50 时时间复杂度 ≥2^112（符合 NIST 轻量密码要求），无 <2^47 对 / 2^112 时间攻击。面积/延迟（NanGate 15 nm）：14,181 GE / 404 ps（PRINCE 13,468 GE / 401 ps），几乎零额外开销。
- 在 LIPPEN 中为何选它：指针在解引用前解密，解密延迟直接进入 load-use 关键路径，须单周期 unrolled 实现；经典轻量密码（PRESENT、SIMON、KATAN 等）轮数太多不适合 unroll；QARMA 有 64-bit tweak（可编码上下文）但面积/逻辑深度更大；PRINCEv2+mod（XOR 注入 m1/m2 的变体）1,522 LUT/1cyc/42 MHz 优于 QARMA-unrolled 1,794 LUT/40 MHz。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 LIPPEN 硬件中的流程：PTR_SEAL/PTR_UNSEAL 指令 → RoCC 请求队列 → PRINCEv2 unrolled datapath 单周期完成 Enc_{k⊕m2}(ptr⊕m1)（加密与解密共用 datapath，因 PRINCE 的 α 反射性质）→ 结果经 RoCC 响应队列回核 → 校验 m1 位为 0。modifier 通过 XOR 注入 plaintext（m1）与密钥（m2），避免采用更贵的 tweakable cipher（QARMA）。
- 硬件评估：Table III 对比 QARMA-vhd（2 cyc/67 MHz）、PRINCE-vhd（2 cyc/84 MHz）、PRINCEv2-unrolled（1 cyc/44 MHz）、PRINCEv2-unrolled+mod（1,522 LUT/1 cyc/42 MHz）；Table VI 整机数据：Rocket-LIPPEN RoCC 1,034 LUT vs Rocket-PAC（QARMA）2,071 LUT——PRINCEv2 数据通路更小、Fmax 更高（99 vs 89 MHz）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：公开 HDL 基线来自 TU Graz memsec（qarma.vhd/prince.vhd，https://github.com/isec-tugraz/memsec/tree/develop/hdl/crypto）与 PRINCE 论文实现，LIPPEN 修改加入 m1/m2 XOR 逻辑（+mod）；Verilator 仿真 + Vivado 2021.2 综合到 VCU118。
- 使用场景：低延迟硬件安全原语——指针认证/加密、内存加密、IoT/感知计算单周期加解密；LIPPEN 中作为指针完整性原语在 RoCC 上以 100 MHz 运行（Rocket/BOOM）。

涉及论文标题：
- LIPPEN: A Lightweight In-Place Pointer Encryption Architecture for Pointer Integrity
