## PACMAN 攻击（投机执行 PAC oracle）

术语解释
MIT CSAIL 于 ISCA'22 提出的攻击（Ravichandran et al.）：利用投机执行把 PAC 认证结果通过微架构侧信道（TLB）变成 oracle，在 Apple M1 上约 2.94 分钟暴力破解 16-bit PAC，突破 PAC 防护实现控制流劫持；是 LIPPEN 低熵动机的核心证据。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PACMAN 核心：构造"PACMAN gadget"（错推测分支 + 对被签名指针的认证指令 + 触发侧信道的解引用），在投机路径上认证失败不会抛架构异常，但认证结果影响后续访存是否发生 → TLB/缓存侧信道可区分"猜测 PAC 是否正确"；由此建立 PAC oracle，无需崩溃即可逐位穷举 16-bit PAC。Web 证据：官方站 http://pacmanattack.com/ 、paper.pdf、MIT news（https://news.mit.edu/2022/researchers-discover-hardware-vulnerability-apple-m1-0610）。
- 影响：Apple M1 16-bit PAC 在 ~2.94 分钟被暴力破解；跨特权级可泄露内核指针签名（PACMAN I/II）；Neoverse V1/N2、Cortex-A78C/A710 等多款 Arm 核也受影响；硬件层面不可完全打补丁。后续缓解：XPAC 立即清 PAC、FEAT_FPAC_SPEC 使认证失败/成功在架构状态上不可区分、MTE 组合——但投机路径下伪造指针仍可被暂时使用（Speculative ROP 等）。
- 与 LIPPEN 的关系：PACMAN 证明低熵截断 MAC 的根本弱点——这是 LIPPEN 用全 64-bit 加密（2^64 暴力空间）替代 PAC 的直接动机；LIPPEN 的威胁模型明确包含投机执行期间的指针伪造。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：攻击者命中可被错推测的分支 → 投机执行路径上的 AUTIA 用猜测的 PAC 认证 → 认证结果决定后续 load 是否访问某地址 → 该 load 的 TLB/cache 状态可测（计时）→ 用二分/穷举推断正确 PAC → 得到合法签名后构造 ROP 链劫持控制流。整个过程中无架构异常、无需崩溃重试。
- 对硬件的影响：PAC 认证的延迟/结果本应隐藏，却被投机执行放大为可观测侧信道；LIPPEN 的 2^64 空间使此类 oracle 攻击在计算上不可行（暴力或需破解 PRINCEv2，且无低熵 PAC 可猜）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：公开 PoC（https://pacmanattack.com/ ），需要受害者程序存在可利用内存破坏 + 可用 PACMAN gadget；演示于 Apple M1 与多款 Arm 核（DEF CON 30）。缓解：PAC 清除（XPAC）、FPAC_SPEC、更大认证空间（LIPPEN 思路）、硬件侧认证失败立即抑制投机访存。

涉及论文标题：
- LIPPEN: A Lightweight In-Place Pointer Encryption Architecture for Pointer Integrity
