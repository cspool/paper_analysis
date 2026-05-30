## ILP-Based Context-Tiling and Kernel Scheduling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

ILP-Based Context-Tiling and Kernel Scheduling是UltraAttn使用的整数线性规划（Integer Linear Programming）方法，用于优化分布式attention的workload分配和kernel执行顺序。三个ILP环节：(1) Device-Level Context-Tiling ILP——在$P \times P$ grid上分配block $B_{r,c}$到设备$U_g$，最小化MCV；(2) Node-Level Context-Tiling ILP——同一formulation但以node为分配单位；(3) Runtime Kernel Scheduling ILP——在parallel dependency graph中确定per-stream kernel执行顺序，最小化$End\_Time$。求解器使用Gurobi（商业），求解时间在大部分配置下为毫秒级。

从编译框架角度拆解，Device-Level Context-Tiling ILP formulation：
```
# Variables: x_{r,c,g}(binary: B_{r,c}→device g), H_{g,r}(need Q_r?),
#           V_{g,c}(need KV_c?), A_g(inbound Q), B_g(inbound KV),
#           C_g(outbound Q), D_g(outbound KV), Cin_g, Cout_g, MCV

# Constraints:
Allocate Uniqueness: ∑_g x_{r,c,g} = 1, ∀(r,c)∉EB
H definition: H_{g,r} ≥ x_{r,c,g}; V definition: V_{g,c} ≥ x_{r,c,g}
A_g = ∑_{r|Cmap(r)≠g} H_{g,r}  # remote Q needed
B_g = ∑_{c|Cmap(c)≠g} V_{g,c}   # remote KV needed
C_g = ∑_{r|Cmap(r)=g} ∑_{k≠g} H_{k,r}  # local Q→remote
D_g = ∑_{c|Cmap(c)=g} ∑_{k≠g} V_{k,c}  # local KV→remote
In/Out traffic: Cin_g = A×1 + B×2 + C×1; Cout_g = A×1 + C×1 + D×2
Comp Balance: ∑_{FB}x×1 + ∑_{CB}x×0.5 ≤ τ, τ = ⌈COMP/CP⌉
Objective: minimize MCV, where MCV ≥ max{Cin_g, Cout_g}
```
**Annotations**: 权重Q:KV:O=1:2:1的物理含义——每个token的Q/O维度=D，K+V合并维度=2D，因此KV通信量为Q/O的两倍。Cmap来自context remap。等价于number-filling问题：在$P \times P$ grid中填入$[0,CP)$数字，最小化每数字占据的行/列投影。

Runtime Kernel Scheduling ILP formulation：
```
# Variables: S_v(start time, real), Order_{uv}(exec order, bool), End_Time
# Stream Exclusivity (同stream不重叠):
S_u + D_u ≤ S_v + (1-Order_{uv})Ub ∧ S_v + D_v ≤ S_u + Order_{uv}Ub
# Dependency: S_u + D_u ≤ S_v for ∀(u,v)∈E
# End Time: S_v + D_v ≤ End_Time for ∀v∈V
# Objective: minimize End_Time
```
**Annotations**: $D_v$由profiling获取。Ub为所有$D_v$之和（安全上界）。Stream exclusivity通过$Order_{uv}$控制：(u→v)时第一个clause有效，(v→u)时第二个clause有效——因Ub大于End_Time，被disable的clause自动满足。

术语一般如何实现？如何使用？Gurobi Optimizer求解（学术许可）。ILP时间受CP、P和pattern密度影响——dense pattern (causal, P=8, CP=64)最重（3672ms）。开源替代：SCIP、OR-Tools CP-SAT、HiGHS。可优化方向：启发式warm-start或approximation加速dense pattern下的ILP。使用场景：分布式训练/推理前的offline planning——ILP结果缓存为execution plan，运行时直接使用。

涉及论文标题：
- UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling
