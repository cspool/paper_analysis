## Monodromy 多胞形（Monodromy Polytope，2Q 门合成成本建模）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Monodromy polytope 是量子编译中用于精确刻画"给定 basis 门集（如 {√iSWAP, ECP}）能用 N 个 2Q 门合成出的所有 2Q 门集合"的理论工具（Zhang 等提出的 Weyl chamber 内多胞形理论）。它基于 2Q 门规范表示：Weyl chamber 内被固定数量 basis 门（任意夹心 1Q 门）可达的区域恰好是一个凸多胞形（polytope）。例如 {√iSWAP} 基下 2 个 √iSWAP 可达区域是四面体 {1/2 ≥ a ≥ b+|c|}；{√iSWAP, ECP} 基下不同门数对应不同多胞形（Fig.5：2√iSWAP 区域、1√iSWAP+1ECP 区域、3 门区域等）。给定目标 2Q 门的规范坐标，查它落在哪个多胞形，即得所需的最小 2Q 门数与具体组合方案——这是对任意量子 ISA 合成能力的统一、量化描述。
- 论文用途：用 monodromy 多胞形为路由中的每个 2Q 门/SWAP 组合计算 ISA 特定的合成成本（SWAP ~ Can(1/2,1/2,1/2) 在 {√iSWAP,ECP} 下 √iSWAP 与 ECP 等成本时优先 "1√iSWAP+1ECP"，ECP 成本 >2×√iSWAP 时改用 "3√iSWAP"），从而做 routing-synthesis 协同优化。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 编译框架内运转流程：① 预计算：对目标 ISA 的 basis 门集，把各 basis 门的规范坐标及其"镜像/逆"做合成闭包，枚举可组合的门数 k=1,2,3... 得到 Weyl chamber 内 k 门可达多胞形族（覆盖集 coverage set）；② 查询：路由中遇到 2Q 门或候选 SWAP，先 KAK 分解得其规范坐标，再定位包含该坐标的最小 k 多胞形 → 得合成成本（k 与具体组合）；③ 成本归一：basis 门各有预定义单位成本（式3：CX:1、ZZ(π/t):2/t、√iSWAP:0.75、iSWAP:1.5、ECP:1.25、pSWAP(π/t):2−1/t），多胞形族帮助按总成本选最优合成方案；④ 缓存：CANOPUS 缓存已计算的规范门成本，避免重复计算。论文实现中成本计算是定位凸多胞形，线性时间复杂度；monodromy 库负责 Weyl chamber coverage 计算。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：开源库 monodromy（https://github.com/Youngcius/monodromy，CANOPUS 作者维护，依赖 lrslib 的 lrs 二进制做多胞形枚举，需手动安装），`pip install git+https://github.com/Youngcius/monodromy`。CANOPUS 在安装时即依赖该库生成各 ISA 的 coverage set（experiments 目录 Makefile 中 `make` 会 prepare coverage sets）。理论来源：Zhang 等 monodromy polytope 论文（CANOPUS 引 [56]）。
- 与朴素方法的区别：朴素 rebase（如 KAK+固定模板）逐门合成、不感知 ISA 合成能力的上下文差异；monodromy 多胞形给出"该 ISA 下此 2Q 门最便宜要几个 basis 门"的精确答案，使路由器能在插入 SWAP 前就量化其真实代价，是 CANOPUS 相对 CX-centric 路由器的核心差异。

涉及论文标题：
- Unifying Qubit Routing Across Diverse Quantum ISAs via Canonical Representation
