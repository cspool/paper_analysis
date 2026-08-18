## 全指针加密（Full-Pointer Encryption）

术语解释
LIPPEN 的核心设计：把 64-bit 指针整体当作密文块，用轻量低延迟分组密码（PRINCEv2）加密，解引用时透明解密并校验，实现完整性+机密性、零元数据开销，暴力空间从 PAC 的 2^16 提升到 2^64。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 与 PAC 的截断 MAC 不同，全指针加密不保留原始地址位，而是把指针当作 64-bit 明文块整体加密（seal = Enc_{k⊕m2}(ptr ⊕ m1)，unseal = Dec_{k⊕m2}(cipher) ⊕ m1），解密后校验未用位（m1 所在位）必须恢复为 0，否则判定伪造并抛异常标志（检测概率 1-2^-|m1|）。攻击者伪造指针须在 2^64 空间内猜中合法密文或破解底层密码（形式化：Adv_LIPPEN ≤ Adv_E + ε(q)，归约到对分组密码的选择密文攻击）。
- 先例：PointGuard/CTR 模式指针加密存在定向位翻转弱点；C3 做部分指针加密但只有 1/16 旁路概率。LIPPEN 首次评估"全 64-bit 指针加密"的安全与性能。
- 关键设计权衡：加密会序列化"解密-解引用"（解密在 load-use 关键路径上，PARTS 估 4 cycle/dereference，nbench 数据指针约 20% 开销），但代码指针依赖分支预测而非裸地址（BTB/RAS 用当前 PC 预测，不因指针加密改变预测率，仅 mispredict 时多 1 cycle），故整体开销与 PAC 相当（SPEC geo mean 2.9% vs PAC 3.6% on Rocket）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件流程（一次被保护的函数调用）：LLVM RISC-V backend 在 call 前插 PTR_SEAL（modifier=SP）→ 返回地址经 RoCC 紧耦合队列送 PRINCEv2 引擎单周期加密得 Enc_{k⊕SP}(ra) 写回栈 → 返回时 PTR_UNSEAL 送 RoCC 解密并校验高 16 位为 0 → 恢复真实返回地址供 RAS/分支预测，失败置异常标志。数据指针：load 前插 PTR_UNSEAL，认证位于 load-use 关键路径（pointer-chasing 微基准实测，Loop 单依赖 load 与 Unrolled 32 依赖 load 变体）。
- 硬件代价：PRINCEv2 单周期 unrolled datapath（1,522 LUT/1cyc/42 MHz post-implementation），Rocket-LIPPEN 总 58,137 LUT/99 MHz/4.035 W vs Rocket-base 56,311 LUT/150 MHz/3.935 W——面积/功耗增幅 <4%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Chipyard v1.8 中扩展 Rocket/BOOM 核，RoCC 挂 PRINCEv2 加密引擎；LLVM v18.1 RISC-V backend 新增 PTR_SEAL/PTR_UNSEAL 插桩；SET_KEY 配置 128-bit 密钥、SET_M_SIZE 配置 modifier 长度。开源：https://github.com/bearhw/LIPPEN （GPL-3.0），artifact https://doi.org/10.5281/zenodo.19901476。
- 使用/部署：与 PAC 相同的编译器插桩与 ABI；密钥管理与 PAC 一致（按域/EL/boot 派生，可防跨域 key-collision）；指针算术场景可编译器优化"先解密后算术"的顺序（RSTI/AOS 表明算术非瓶颈）。

涉及论文标题：
- LIPPEN: A Lightweight In-Place Pointer Encryption Architecture for Pointer Integrity
