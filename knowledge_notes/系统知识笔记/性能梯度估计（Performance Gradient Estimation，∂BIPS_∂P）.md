## 性能梯度估计（Performance Gradient Estimation，∂BIPS/∂P）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 性能梯度估计是 PowerGrad（ISCA'26，UIUC/UNC/IBM/AMD）提出的运行时功率管理核心机制：动态估计每个运行中工作负载的"性能对功率的梯度" ∂perf/∂power（论文用 ∂BIPS/∂P 作为可计算形式），即单位功率变化带来的指令吞吐变化量，表征工作负载当前对功率的性能敏感度。思路是：若把功率从低梯度（功率不敏感）工作负载转移到高梯度（功率敏感）工作负载，前者失去的性能少、后者获得的性能多，总功率不变下净性能增益——这正好解决严重功率受限环境（所有节点需求>分配）下"仅凭功率观测无法确定节点间优先级"的问题。它需要可微的性能/功率模型：性能模型 CPI(f)=CCPI+MCPI×f/f^(t)（线性于频率），功率模型 P(V)=P_idle(V)+P_active(V,E)（idle 为 V 的三次多项式、active 为计数器加权×(V^γ+V)），系数由硬件性能计数器与运行时电压/频率值在线生成（数据驱动、免领域知识、无需预 profile 工作负载）。
- 梯度计算逻辑链（论文 §III-D，公式 (8)–(13)）：∂BIPS/∂P = (∂P_active/∂BIPS + ∂P_idle/∂BIPS)^(-1)；由于 V、f、计数器相互耦合无法直接微分，做三个近似：① V 是 f 的二次多项式 [3]；② 频率变化时性能事件计数 E_i 与 BIPS 线性成比例（E_i=e_i·BIPS）；③ 非空闲时长∝1/f、空闲时长不变 → util(f)=util^(t)/(util^(t)+(1−util^(t))·f/f^(t))。据此用链式法则 ∂V/∂BIPS=∂V/∂f·(∂BIPS/∂f)^(-1)（∂BIPS/∂f(f^(t))=(util·CCPI)/CPI²−util(1−util)/CPI）算出 ∂P_active/∂BIPS 与 ∂P_idle/∂BIPS。最后用链式法则把每核心梯度聚合为处理器梯度 G=Σ_i(∂BIPS_i/∂P_i·P_i/P)（近似每核功率随处理器功率比例变化）。实测梯度估计精度：Legacy 上 R²≈0.501（迭代优化只需近似正确梯度即可收敛）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 PowerGrad 三层组件中的运转流程（每 100ms 一个周期）：
```
输入: 每核心硬件计数器 BIPS, util, ldm_stalls; 频率 f^(t); 离线回归系数 a_i, w_i, γ
BCPS ← f^(t) · util;  CPI ← BCPS/BIPS;  MCPI ← ldm_stalls/BIPS;  CCPI ← CPI − MCPI
CPI(f) ← CCPI + MCPI · f/f^(t)                     # 线性性能模型
P(V) ← a3V³+a2V²+a1V+a0 + Σ_i(w_i E_i^(t))·(V^γ+V) # 多项式功率模型
∂BIPS/∂P ← (∂P_active/∂BIPS + ∂P_idle/∂BIPS)^(-1)  # (8), 链式法则+三个近似
G ← Σ_i(∂BIPS_i/∂P_i · P_i/P)                       # (13) 聚合为处理器梯度
输出: G 传给 Local Controller / Hierarchical Controller
```
- 例子（Legacy 双 CPU 节点，Llama-high 与 Llama-low 各占一个处理器）：Llama-high 的 prefill 阶段（计算密集）测出高梯度，decode 阶段（内存密集）梯度骤降；Llama-low 因小 batch 内存密集梯度恒低。控制器看到 G_high>G_low 就把功率上限从 Llama-low 处理器转移到 Llama-high 处理器——图 9 实测 PowerGrad 下功率在两类请求间动态转移，响应延迟下降。梯度是"功率-性能敏感度"而非简单计算/内存二分：核心频率已很高时，即使计算密集也可能低梯度（升频吃功率多、性能增益少）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Gradient Estimator 为用户级 Java 线程，每 100ms 被唤醒，从 Linux perf/RAPL 侧读取性能计数器与频率/电压，重算模型系数并微分；无需应用配合（软件透明）。回归系数 a_i、w_i、γ 离线训练：Legacy 用 PARSEC 3.0 应用、Accelerated 用 TorchBench（engage AMX 指令）按 PPEP 框架方法学拟合；六个计数器：instruction-count、cycle-count、uops.executed、cache-misses、branch-misses、ldm_stalls_pending（Accelerated 额外 exe.amx_busy、fp_arith_inst_retired.vector，1 个 AMX busy 周期计 16 条指令）。使用：作为 Local/Hierarchical Controller 的 Algorithm 1 输入（PL'[i]=PL[i]+lr×G[i]−α(PL[i]−P[i])），把梯度转成功率重分配。可移植性：新架构只需重训梯度估计器的回归系数即可重定向。局限：需要 kernel 执行期间动态读性能计数器的硬件支持，当前 GPU/加速器（如 NVIDIA kernel 执行中不可查询计数器）无法支持，故论文只在 CPU 集群上评估。

涉及论文标题：
- PowerGrad: Hierarchical Power Management for Power-Limited ML Inference Clusters
