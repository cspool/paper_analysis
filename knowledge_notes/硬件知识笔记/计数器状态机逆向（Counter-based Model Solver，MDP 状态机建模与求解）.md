## 计数器状态机逆向（Counter-based Model Solver，MDP 状态机建模与求解）

术语解释
把 MDP 的预测置信度状态机建模为硬件计数器（计数器的值 + 阈值谓词决定预测，实际依赖结果驱动计数器更新），再用自动化边界搜索 + 不等式组求解恢复状态机参数的逆向方法。SSBench 的核心贡献之一（Algorithm 1），解决 AMD 五计数器/十转移函数这类人工逆向爆炸问题。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MDP 与分支预测器不同：误预测代价不对称（独立误预测=回滚 vs 依赖误预测=阻塞），导致状态机非对称；且有冷启动属性（初始迁移需多次误预测）。SSBench 的单计数器模型用 7 个参数定义：(upd_{0t}, upd_{1f}, upd_{0f}, upd_{1t})（0/1=非依赖/依赖预测、t/f=预测正确/错误时的计数器更新值）、上界 bnd、上溢/下溢行为（ovf/unf）、阈值 ths（谓词 c≤ths）。简化假设：初始态 0 预测独立、unf=0；足够长独立序列重置到 0（upd_{1f}=−1）；单阈值。Algorithm 1 流程：自动搜索最小重复次数 x1（D_P 使计数器越过 ths，建立 upd_{0f} 与 ths 关系）→ x2（x1 次 upd_{0f} 后 N_P 使计数器回到 ths，建立 upd_{1f} 与 ths 关系）→ 测试 upd_{0t}>0 分支（x3=∞ 表示依赖正确时无增）→ x4/x4'/x4'' 建立 upd_{1t}、bnd、ovf 关系 → x5/x6 建立 upd_{1t}、upd_{0f} 与 ths 关系 → 用 pulp 解不等式组 EQ1..EQ7。实测恢复的参数（Table II）：Intel L-S 状态机 [0,-1,15,14,15,1,0,0]、AMD Zen3 z3-mdp2 [0,-1,15,16,62,1,0,30]（ths=30>0 体现需多次误预测激活）、Cortex-A76 [0,-1,14,0,14,1,0,0]、Apple SL-S [0,-1,7,1,7,1,0,0]（3-bit 计数器）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。
运转流程（单计数器，以 Cortex-A76 为例）：初始 c=0 预测独立（S）→ 首个 D_P 使计数器初始化到 14 且依赖预测正确时 upd_{0t}=0（不更新）→ 依赖预测时计数器停在 14（B 持续）→ 每次独立对（N_P）使 upd_{1f}=−1 递减（B→S 转移所需 N_P 数=x2）→ 直到 c≤ths 恢复独立预测。SSBench 的 sm.py 用固定/变化 IP 的 store-load 对执行 Algorithm 1：每次 D_P 后按 x1/x2/... 模式测 T 序列，从 S/B/R 转移点反推边界值，代入不等式组求解。双计数器模型（Fig.8，三谓词更多参数）用于覆盖更复杂设计；若 1 计数器模型无解（如 ≥2 计数器），SSBench 终止并提示偏离假设的组件，转人工分析。同核双 MDP（AMD Zen3 的 MDP-1 L-S 与 MDP-2 SL-S）隔离技巧：MDP-1 用共享 load IP 不同 store IP 的 Pair_0^x 切换，MDP-2 用 Pair_0^0/Pair_0^1 且每次 D_P 后足够 N_P 重置 MDP-1 计数器。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SSBench 中 sm.py + utils/ 求解器（pulp 线性规划求解不等式组），微基准用乘法延迟 store 地址放大 S/B/R 时序。使用方式：对目标 CPU 跑状态机阶段 → 输出 8 参数数组（store_sm/load_sm 格式见 characterization.json 示例）；对 SL 型需分别恢复 store 侧与 load 侧状态机。用途：理解 MDP 预测行为（激活阈值、更新步长、重置行为），支撑攻击构造（如 AMD ths=30 意味着需 30 次依赖才能稳定预测依赖——MDP-CF 攻击中初始化计数器到 32 后按受害进程分支路径探测）与缓解评估。

涉及论文标题：
- SSBench: Automated Characterization of Memory Dependence Predictors on Modern CPUs
