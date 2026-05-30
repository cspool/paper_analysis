## ILP-based Expert Affinity Placement (基于整数线性规划的专家亲和放置)

术语解释
ILP-based Expert Affinity Placement 是 ExFlow 论文提出的一种离线优化方法：将 MoE 模型中各层 expert 到 GPU/node 的放置问题形式化为整数线性规划（Integer Linear Programming, ILP），目标是最小化 token 的跨 GPU/跨节点路由次数，约束为保证每 GPU/node 负载均衡。与 METRO 的 MIN-EXP-ROUTING（ILP 最小化 activated expert replicas）不同，ExFlow 的 ILP 目标是最小化 cross-layer token re-routing。

术语是什么？
给定 E 个 expert/层、P 个 GPU（或节点），每个 GPU 容量为 C=E/P 个 expert/层。目标是找到 binary assignment matrix $x_{i,j}^p$（expert $E_i$ 在 layer $j$ 是否分配给 GPU $p$），使得所有 token 在跨层遍历时的 re-routing 次数最小。

ExFlow 通过 Lagrange 对偶将原始的"最大化 combined affinity"问题转化为等价的"最小化 token re-routing cost"问题，从而得到可高效求解的 ILP 形式。

从编译框架角度拆解术语：
ILP Formulation 完整形式：

```
Minimize:  sum_{k=1..N} sum_{j=1..L-1} R_{k,j}          (8)

Variables:
  x_{i,j}^p ∈ {0,1}  — expert i at layer j on GPU/node p
  R_{k,j}   ∈ {0,1}  — token k at layer j requires cross-GPU/node routing

Subject to:
  (9)  sum_{i=1..E} x_{i,j}^p = E/P      load balance: each GPU gets E/P experts per layer
       for all j ∈ {1..L}, p ∈ {1..P}

  (10) sum_{p=1..P} x_{i,j}^p = 1        exclusivity: each expert on exactly one GPU
       for all j ∈ {1..L}, i ∈ {1..E}

  (11) R_{k,j} >= x_{i,j}^p - x_{i,j+1}^p   routing indicator: token k changes GPU
  (12) R_{k,j} >= x_{i,j+1}^p - x_{i,j}^p   (symmetric constraint)
```

**两阶段求解（Staged Optimization）**：
- Stage 1: 将 P 设为节点数，最小化 inter-node routing（利用 NVLINK >> InfiniBand 带宽差异）
- Stage 2: 基于 Stage 1 结果，在每个节点内将 P 设为 GPU 数，最小化 intra-node routing

**Lagrange 对偶转化思路**：
原始问题：maximize aggregate affinity P(E_{p,j+1}|E_{i,j}) → 等价 dual: minimize routing disruptions。Dual function:
$g(\lambda, E_{i,j}) = \inf_{E_{p,j+1}} [P(E_{p,j+1}|E_{i,j}) - \lambda G(E_{p,j+1}, E_{i,j})]$
其中 $G(\cdot)$ 为 re-routing cost function，$\lambda$ 为正则化项。

术语一般如何实现？如何使用？
- **Offline 一次性求解**：profiling N=1000-3000 tokens → 构建 routing log → 求解 ILP → placement 用于模型加载
- **求解效率**：因仅需数千 token 的 routing log，变量规模可控（如 MoE-64, L=24, 8 GPUs: $x$ 变量约 64×24×8=12,288 个），可用标准 ILP solver（CPLEX、Gurobi、OR-Tools）快速求解
- **拓扑适应性**：placement 参数化于 $P$（GPU/node 数），对任意拓扑无需 retrain；硬件拓扑改变时重新求解 ILP 即可
- **与 METRO MIN-EXP-ROUTING 的区分**：
  - METRO: ILP 最小化每 GPU activated expert replicas 的最大数量（makespan minimization），解决 scheduling problem
  - ExFlow: ILP 最小化 cross-layer token re-routing 总量，最大化 intra-GPU/intra-node affinity
- **无需 expert replicas**：与 Lina（local replica of popular experts）不同，ExFlow 的 ILP 通过全局优化 placement 避免 replicas，在极端情况（每 GPU 仅 1 expert/层）仍可通过 intra-node affinity 提供加速

涉及论文标题：
- Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference
