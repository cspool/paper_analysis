## Measurement Nativization & Masking（测量原生化与掩码：逻辑 qubit 映射 + idle-qubit Z 掩码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 把协议要求的逻辑 Pauli 测量/旋转编译成目标架构"单次原生测量"的过程。背景：BB 码的原生测量集有限——LPU 只有 15 种双逻辑 qubit 测量，乘 36 个 shift automorphism 得 540 个 native 多体测量，仅占 $4^{12}$ Pauli 群的一小部分；非原生 Pauli 需经 Clifford 共轭（[31]）或多段 automorphism 才可测，代价高。原生化 = 通过 (i) 逻辑 qubit 映射（把 m 个协议 qubit 放进 k 个逻辑 qubit 的哪个子集 S，使多数所需旋转 native）与 (ii) masking（m<k 时用空闲 qubit 补 Z 因子把非 native P 变成 native Q）最大化"一次 automorphism 序列 + 一次 LPU 测量"即可实现的旋转比例。masking 合法性：空闲 qubit 初始化 |0⟩ 时 Z 作用不变量（Z|0⟩=|0⟩），故 $Q=P\cdot\prod_{j\in\mathcal{M}}Z_j$ 与 P 在活动 qubit 上等价，且零深度。
- 从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 编译流程（本论文 Sec. IV-A/B）：
  ```
  输入: BB 码 k=12 逻辑 qubit、协议 m 个 qubit 的旋转标签 {P_c=Z^⊗S_c}
  ① Mapping: 枚举 m 元素子集 S⊆[k]，选择使 native 旋转数最大的分配
     （小协议可暴力枚举；平局按到 pivot 的路由距离打破）
  ② Masking: for 每个仍非 native 的 P_c:
       找 native Q 使 Q 在 S 上与 P_c 一致、只在 idle qubit 上差 Z 因子
       → 替换 P_c ← Q = P_c·Π_{j∈M}Z_j（M=掩码 qubit 集，初始化 |0⟩）
  ③ 残余非 native 旋转用 Clifford 共轭兜底
  输出: 每个旋转 = 1 个 automorphism 序列 + 1 次 LPU 测量（+byproduct Pauli 跟踪）
  ```
  效果：15-to-1 在 gross 码上 15 个旋转全部 native（原 4 个非 native 被 masking 消掉）；8-to-CCZ 在两码上全 native。Fig 9(b) 显示 native 测量占比随掩码 qubit 数增长（full-Pauli 与 I/Z-only 两算子集分别统计）；编译开销 ~10 min（mapping+masking，two-gross+49-to-1 最大实例）。收益传导：减少 Clifford 共轭 → 缩短深度 → 降低累积逻辑错误。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 论文代码：https://github.com/kunliu7/bb-code-magic-state-distillation 的 prj_msd/10mapping/。相关外部实现：QLDPC 逻辑映射的两阶段流水线（超图划分聚 in-module 簇 + 优先级算法分配簇到硬件）；GeneCS（Zhou, Javadi-Abhari, Li）做任意稳定子码的 resource-efficient surgery 合成；extractors 架构（arXiv:2503.10390）与 parallel Pauli product measurements（arXiv:2407.18490）把"原生测量集"扩展到任意逻辑 Pauli——masking 思想可迁移到"块未占满时扩展有效测量集"的通用场景（论文称同样惠及其他 BB 电路）。
- 涉及论文标题：
- Distilling Magic States in the Bicycle Architecture
