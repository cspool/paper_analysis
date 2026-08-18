## Decoupled Read Protection（解耦读保护）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Decoupled Read Protection（解耦读保护）是 HBM-CASO 的读路径技术：把 ECC 解码过程分离成 error detection（检错）与 correction（纠错）两个阶段，HBM 片内只做检测（因为片上资源不足以解码 advanced SysECC 或被压缩的 regional 码字），检测到错误后由处理器侧内存控制器取回全部奇偶校验并完成纠错。关键观察：ECC 仅用于检测时，编码逻辑可以直接复用为解码——重新生成 parity 并与存储的 parity 比较即可，无需完整解码器。具体流程：从内存阵列读出 regional codeword → ODECC 与 CRC 单元做 error detection（复用 Merging Unit 重生成 parity 比较）→ 无错：控制器用 global parity 做附加验证（同时覆盖传输错误）→ 有错：HBM 不尝试纠正、发 alarm 信号给处理器（无 alarm 通道时用特殊数据模式 "catchword"，同 XED [52]）→ 控制器发起特殊访问、通过加长 burst 从数据通道取回全部 parity（regional ECC + CRC parity，类似 DUO [18]）→ 控制器做 tiered error correction。这保证 HBM-CASO 与片内 error scrubbing 兼容（scrub 扫描读到错误时同样只检测、上报，不就地纠正）。
从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（G-mode 读路径）：读请求 → HBM 阵列返回 regional RS(34,32) 码字 → 片内 Merging Unit 重生成 regional parity 与存储 parity 比较（error detection，复用编码逻辑、延迟被 full correction path 管线覆盖）→ 检测无错 → 码字经数据通道传出 → 控制器用 global RS(68,64) parity 复核（覆盖存储+传输错误）→ 检测到错误（或收到 alarm）→ 控制器发起特殊读、加长 burst 取回 regional + CRC parity → tiered correction：(1) 把两个 regional RS(34,32) 合并为 RS(66,64)、再补 global 扩展为 RS(70,64)（6 check 符号），走标准 Berlekamp-Massey 纠 ≤3 符号错（若恰好 3 错或 DUE 则弃用该结果，防 miscorrection）；(2) ≥4 符号错时 brute-force 枚举错误位置（排除 4 错全在同一 32B 区域的不可纠组合，共 C(72,4)=956,870 组合），CRC 校验通过才接受。延迟分级：≤2 错硬件快速处理；多错（罕见）卸载到软件避免额外硬件。正常（无错）路径只增加 tCL 0.25/0.51ns（R/G-mode）。片上代价：ODECC+CRC 只需检测逻辑（复用编码），故片内新增逻辑极小（+61 cells）；完整的 global/regional 编解码在控制器侧（G-mode 5910μm²/4.56ns）。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：HBM 侧"编码即检测"逻辑（Merging Unit 重生成 parity + 比较器）+ alarm/catchword 上报通道；控制器侧 global/regional RS 编解码器 + 特殊访问（加长 burst 取 parity）+ 分层纠错调度。使用场景：(1) 片上资源不足、无法解码大码字的内存（HBM-CASO 的核心动机之一）；(2) 需保留片内 scrubbing 的场景（检测 + 上报代替就地纠正）；(3) 大码字 SysECC（G-mode RS(68,64)）覆盖传输 + 存储错误，比 baseline 的 CRC-only 检测强得多（Table IV：G-mode UE% 全 0）。设计权衡：检测不增加读关键路径（大多数访问无错），纠错路径延迟随错误数增长（≤2 硬件、≥3 软件）；32B 细粒度访问下 G-mode 用 regional parity 单独检测即可（global parity 只在检测到错误后才读），避免读放大——与"更大码字=更强保护但更粗粒度"的经典权衡（NVM RMW 问题）对比，HBM-CASO 用"分层检测/纠错"解耦了读路径的粒度约束。此思路源自 [79]（Virtualized and Flexible ECC，解码分离检测/纠错两阶段）。论文未明确说明 alarm 通道的物理实现。

涉及论文标题：
- HBM-CASO: A Coordinated Approach to HBM System-Level and On-Die ECC
