## k-induction 半形式验证（Semiformal Verification / 有界模型检验 BMC 扩展）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
k-induction（k 步归纳）是模型检验/形式化验证中证明时序性质的一种归纳技术，建立在 Bounded Model Checking（BMC，有界模型检验）之上：BMC 只证明性质在 k 步内成立（有界、不完备）；k-induction 额外做归纳步——假设"任意 k 步状态序列若满足性质则第 k+1 步也满足"，从而把有界证明推广为无界证明，同时避免完整可达状态空间搜索（LTL model checking 随状态元素个数指数膨胀，工业级设计不可行）。因归纳步通常缺少可达性约束（可达状态只是全部状态子集），归纳证明可能失败或需要更强不变量（如 k 需增大或加状态子集约束）；这是"半形式验证"（semiformal）：以严格完备性为代价换取工业可扩展性——实践表明 k 取 2-5 即可识别绝大多数优化机会且运行时可接受。经典出处：Mishchenko et al. ICCAD 2008 "Scalable and scalably-verifiable sequential synthesis"（abc 中实现，scorr/dsec 的基础）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
æSIP 中 k-induction 的运转流程（§IV-C3）：输入 = 注入 EDC（SVA assume）的 baseline netlist + 裁剪候选 → ① scorr：用 BMC 在 k 步窗口内验证候选节点对在 EDC 下所有可达状态功能等价（等价则合并/替换，即 sequential resubstitution）；② dsec：对候选节点注入 stuck-at-0/1 故障，用 k-induction 验证 faulted 电路与原始电路在 EDC 下顺序等价（等价则说明该节点在所有可达状态下冗余，即 sequential redundancy removal）；③ k=2..5 实践足够 → 输出可安全裁剪的门。若归纳不变量深度 >k 则性质不可判定（不完备性），但经验上剩余收益小。论文报告 scorr 引擎 22.5-213.8s（中位 25.0s）——硬件裁剪阶段主导整个框架运行时（对比：程序重写 saturation 1.2-32.4s、extraction 0.3-78.4s）。该流程替代完整 LTL model checking（PDAG 用的方法在 RV32I baseline 上 5 小时超时）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：abc（https://github.com/berkeley-abc/abc）内建命令——scorr（sequential resubstitution：检测并合并顺序等价节点）、dsec（sequential verification / 顺序冗余移除：k-induction 验证故障注入后等价性），底层为 SAT 求解器驱动的 BMC；可配 k 步（2 步常见）。使用：在逻辑综合/验证流程中配合 assume 约束（EDC）裁剪冗余逻辑、或验证优化前后设计顺序等价（dsec 的验证用途）；商用 FPV 工具（Cadence JasperGold）同样支持 k-induction 类证明模式。文献佐证：IEEE "Scalable Sequential Logic Synthesis Using Observability Don't Care Conditions"——abc 的 dsec 2-step induction 验证各 benchmark <13s（最大 74s）；sequential redundancy removal 平均降 3.40% NAND2 门、加 resubstitution 3.65%、2-step induction 3.97%；配合顺序 SAT-sweeping 可再降 18.3% 面积。

涉及论文标题：
- æSIP μArch-aware ASIP-ISA Co-Design via Program Synthesis, Equality Saturation, and External Don't Cares
