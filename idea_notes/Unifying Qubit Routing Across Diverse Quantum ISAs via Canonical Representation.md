## Unifying Qubit Routing Across Diverse Quantum ISAs via Canonical Representation

- baseline方法是什么？
  - baseline 是三类现有量子编译/路由方法：(1) CX-centric 可扩展路由器（SABRE、TOQM 等）——基于简化路由模型，把 SWAP 按教科书分解固定展开为 3 个 CX（SWAP = CX·CX·CX），电路代价用 CX 门数与深度量化，完全忽略后端 native 门（如 √iSWAP、iSWAP、ZZ(θ) 族）的合成能力与物理时长；SABRE 是双向迭代启发式（front layer F + extended set E 平均最短路径距离 + 经验衰减因子），TOQM 是 A* 全局搜索深度驱动、宣称 QFT 深度最优。两者把路由开销系统性高估 2-5×。(2) BQSKIT 类数值优化/暴力近似合成范式——针对目标 ISA 做结构搜索+数值优化端到端 rebase，指数计算复杂度（216 个 case 平均每电路 18 分钟 vs SABRE 17 秒），且不保证合成最优，其评估曾得出"高级 ISA 不比 CX 好"的结论（Kalloor 等 roofline 工作）。(3) 仅限 CX 的交换性优化（CX 对共享控制位或共享目标位时可交换），模式有限且需跟踪 control/target 位置。
  - baseline 全栈执行例子（SABRE 在 CX ISA 上路由含远程 2Q 门的电路，如 1D chain 上的 QFT）：
    ```
    算法pipeline层：电路以 CX 表达（QFT 用 CPhase 块、算术/化学模拟用 CX 序列），逻辑级优化用 CX 专用
               模板/peephole（仅限 CX 的交换律与 Clifford 等价）
    系统框架层：论文未明确说明（量子编译无 serving 调度层；路由后电路直接在 QPU 上执行）
    编译框架层：SABRE 分层提取 front layer F → 对每个候选 SWAP 按绝对平均最短路径距离 Avg{dist} 启发式
               评分 → 插入 SWAP 并按固定 3×CX 成本计费 → 输出 CX 电路；TOQM 用 A* 全局搜索最小化深度
               （但 qft_6 在 1D chain 上只找到 #Can 16 / Depth2Q 10，非最优 15/9）
    kernel调度层：论文未明确说明（CX 电路由 QPU 微码/调度执行，无 kernel 抽象）
    硬件架构层：SWAP 在 CX-only 设备上按 3 CX 合成执行；即便设备原生支持 CZ/iSWAP（如 Sycamore 的
               CZ+iSWAP，SWAP 甚至可原生实现且脉冲时长仅 1.5×CZ），CX-centric 模型也不感知，
               仍按 3×CX 计费 → 路由决策与真实执行成本脱节，优化空间被系统性浪费
    ```
- 论文方法是什么？如何对应解决Baseline的缺陷？
  - CANOPUS 用 "规范 2Q 表示 + monodromy 多胞形合成成本模型 + 统一启发式" 在路由阶段直接建模后端 ISA 的合成能力，实现 routing-synthesis 深度协同优化：①SWAP mirroring——SWAP 与前驱 2Q 门合并为复合酉 U'=SWAP·U，边际成本 c_g=COST(SWAP·U)−COST(U) 可低至 0 甚至为负（CX 基下 SWAP·iSWAP~Can(1/2,0,0) 只需 1 个 CX，c_g=−c_cx；√iSWAP 基下 c_g=0），把"固定 3 CX 的 SWAP"变成"按上下文动态计费的 SWAP 吸收"；②广义交换性（Theorem 1）——任意共享 1 qubit 的规范门满足 b=b'=c=c'=0（纯 XX）即可交换，捕获 CX 编译器看不到的交换模式（算术、QFT、化学模拟常见，Fig.7(b) 四种模式），扩大可重排门池以找到更低成本 SWAP 插入；③统一启发式 H 同时最小化 Ccount 与 Cdepth（D 记录加权关键路径，取消 SABRE 衰减因子），并用一致 basis 门成本表（式3）做跨 ISA 公平评估；④保持 SABRE 的多项式复杂度（所有计算 O(1)/线性，1-2× SABRE 编译时间），把 BQSKIT 的指数复杂度从竞争路径上淘汰。
  - 对应解决 baseline 缺陷：①"3×CX 假设" → monodromy 多胞形精确合成成本（SWAP 实际可 1CX+1iSWAP、3√iSWAP 或原生实现）；②"ISA 盲" → ISA-aware SWAP 选择主动利用 √iSWAP/ZZ(θ)/pSWAP 族合成能力（CX→ZZPhase_ 时 chain 上 Ccount overhead 1.88×→1.39×，-26%）；③"交换性仅限 CX" → 规范形式统一捕获全部 2Q 交换模式（消融：开启后平均再降 2-11% 门数、2-10% 深度，峰值 37-48%）；④"BQSKIT 指数复杂度" → 多项式复杂度可扩展方案；⑤"Tower of Babel 碎片化"（每个后端一套专用编译器）→ LLVM 风格统一框架，换 ISA 只需配置 basis 门成本表。
  - 论文方法全栈执行例子（CANOPUS 在 ZZPhase_ ISA + 1D chain 上路由同一 QFT/电路）：
    ```
    算法pipeline层：TKET 逻辑级优化后 rebase 到 {Can, U3}（KAK 分解得规范形式）；QFT 的 CPhase 块直接
               对应规范系数
    系统框架层：论文未明确说明（无 serving 层；真机验证在 IBM Quantum Cloud ibm_marrakesh 上执行
               QFT 核，Hellinger fidelity 测量，CZ 门数降 52.9%、错误降 26.89%/34.98%）
    编译框架层：DAG 分层 → 对每个候选 SWAP 计算 c_g（与 L 层前驱门 SWAP mirroring 的边际合成成本，
               经 monodromy 多胞形定位 canonical 坐标查成本表，如 SWAP 吸收进 ZZ(π/4) 前驱形成镜像门
               Can(1/2,1/2,1/2-1/4) 等低成本复合）+ Δdepth（D 记录 + 加权重关键路径）
               + 差分拓扑距离 → 选 H 最小的 SWAP 插入 → Algorithm 1/2 更新 L/D/C（O(1)）
               → 输出 {Can,U3}+SWAP 电路 → 后端 synthesizer 按 ISA 成本表最优 rebase
               （pSWAP/ZZ 族低成本实现 SWAP，如 pSWAP(π/2)≡iSWAP 成本 1.5）
    kernel调度层：论文未明确说明（QPU 原生微码执行，无 kernel 抽象）
    硬件架构层：SWAP 以 ISA 原生方式合成执行（Heron-R2 的 CZ+√X+Z(θ)+ZZ(θ)），路由选路时已按真实
               门物理时长权衡深度；qft_6 在 1D chain 上达到理论最优路由（#Can 15/Depth2Q 9，比 TOQM
               的 16/10 优，超越 Maslov 手工最优方案），qft_12 同样最优（66/21）
    ```
