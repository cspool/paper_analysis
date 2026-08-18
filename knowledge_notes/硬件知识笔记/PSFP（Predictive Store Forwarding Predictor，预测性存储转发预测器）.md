## PSFP（Predictive Store Forwarding Predictor，预测性存储转发预测器）

术语解释
AMD 乱序核中与 MDP 协同的另一个预测器：预测 store 数据是否可转发给后续依赖 load（store-to-load forwarding 的结果预测）。在 AMD Zen3 上 PSFP 与 SL-S 型 MDP（z3-mdp1）重叠，是 MDP 逆向与侧信道分析时必须隔离/校正的干扰源。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
store forwarding（存储转发）指依赖 load 直接从更老 store 的未写回数据取数；PSFP 预测"该 store-load 对是否会转发"，从而让 load 提前执行。SSBench 论文中 PSFP 的作用：① 干扰源——若 PSFP 预测转发但在数据依赖上误预测，load 仍会乱序执行，使 T2 中出现 B 与 S 混合（本应只有 B/R），需校正：把 D_P 样本中的 S 调为 B、N_P 样本中的 R 调为 B；② 与 MDP 表重叠——AMD Zen3 的 z3-mdp1（SL-S 型 MDP）与 PSFP 表重叠，SSBench 无法直接表征该表项（Table II 注 z3-mdp1 overlaps with AMD's PSFP and cannot be characterized directly through SSBench），只能表征 z3-mdp2（L-S 型）。这解释了 AMD Zen3 上 MDP 状态的复杂性（此前 [37] 人工逆向出 Zen3 五计数器/十转移函数的状态机即同时涉及 MDP 与 PSFP）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。
在 SSBench 的识别流程中：固定 IP 执行 store-load 对 → 测 T 序列 → 若 T2 同时观测到 B 与 S（即有数据依赖时 load 仍乱序执行），判定存在 PSFP/内存重命名等转发预测机制 → 后续分析把受污染的时序样本校正为纯 MDP 行为（S→B、R→B）→ 否则状态机与组织参数会被 PSFP 干扰而失真。在攻击侧：MDP-CF（AMD）利用的是"无前置 store 的 load 也能更新 MDP"的 z3 表项（非 PSFP 部分），而 z3-mdp1 因与 PSFP 重叠不可直接探测。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于 AMD 乱序核的 load-store 单元（具体微架构未公开；识别依据是行为：数据依赖时 load 仍乱序执行）。使用方式（逆向/安全研究）：在 MDP 时序分析中通过 T2 的 B/S 混合检测其存在并校正数据；在自动化工具（SSBench）中将其标记为 out-of-scope/重叠组件，对重叠表项报告诊断信息而非强行表征。论文未明确说明 PSFP 的独立开源实现或完整硬件细节。

涉及论文标题：
- SSBench: Automated Characterization of Memory Dependence Predictors on Modern CPUs
