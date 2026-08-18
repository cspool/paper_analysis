## Patch 操作（Patch Rotation / Deformation / Movement，patch 旋转/变形/移动）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Patch 操作是表面码 patch 在 board 上的基本几何操作，是晶格手术执行的最小动作集（O3LS Fig.3 规则表）：①初始化——单 qubit patch 可初始化为 |+⟩ 或 |0⟩，双 qubit patch 可初始化为 |+⟩⊗|+⟩ 或 |0⟩⊗|0⟩，成本 0；②Patch Deformation——patch 可扩展到覆盖更多 tile（1 时间步）或收缩到更少 tile（0 时间步），扩展+收缩组合实现 patch 移动到相邻 tile（movement，1 时间步）；③Patch Rotation——patch 可经角移动+平移组合旋转（3 时间步，拆成变形/角移动/移动三片）；④Measurement——相关边缘邻接 ancilla 路径时测量 Pauli 算子（1 时间步）。这些规则源自 Litinski [34] 的 tile-based 协议。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 调度视角：每个 patch 操作是调度器可选动作。在 O3LS 松散调度 Algorithm 2 中，当目标 PPM 不可执行时，从候选操作集 O_B（board 上全部可能的 rotation/deformation/movement）中选奖励最高者（r(o_b,P_i) 最大化可执行 patch 数、保持连通、同奖励取低时间开销）。Rotation 是最昂贵操作（3 时间步），在紧凑布局中可占 >50% 时间步（Fig.7），因此 O3LS 用布局搜索（多边缘 patch 减少旋转）+ Y-synthesis（减少需要旋转的操作数）+ 边感知映射（把高频旋转 qubit 放双边缘 patch）三个模块联合削减旋转。例子：Fig.4 中只移动 q_0 并旋转暴露不同边缘（1+3 时间步）替代整 patch 旋转（3 时间步×多 patch），省掉多轮旋转。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译器生成的指令（lattice surgery instruction set，Triage 论文称 LLI），物理上由表面码稳定子测量电路执行；LER 评估中 rotation 错误率分解为三片分别用 STIM 仿真（d=9、p=10⁻³）。O3LS 在固定布局（compact/sparse/standard）与自动生成的 squeezed 布局上都以 patch 操作序列为执行目标，输出时间步数与 LER。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling
