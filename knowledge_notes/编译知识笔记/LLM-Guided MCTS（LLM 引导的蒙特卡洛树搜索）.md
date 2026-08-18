## LLM-Guided MCTS（LLM 引导的蒙特卡洛树搜索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM-Guided MCTS 是 QiMeng-Tensify（ISCA'26）的核心搜索算法：在标准蒙特卡洛树搜索（Selection/Expansion/Simulation/Backpropagation 四阶段）基础上，用 LLM（gpt-5-preview）生成"候选调度规则上的先验概率分布"注入搜索，引导树向高奖励的 program sketch 与参数配置方向生长。Selection 阶段不用传统 UCB，而最大化 Gumbel-augmented 分数 a* = argmax_a[g(s,a) + π(s,a) + σ·Q(s,a)]：g 为 Gumbel 噪声（Gumbel-max trick 保证探索）、π 为 LLM prior logit（注入语义知识，如"GEMM 后紧跟 elementwise 时推荐 compute_at"）、Q 为动作经验值、σ 平衡先验与经验。Expansion 用 LLM prior 排序扩展顺序；Simulation 做细粒度参数规格（XGBoost cost model + 邻域局部搜索 + 真机测量）；Backpropagation 把 R=flops(p*)/t_best 沿路径移动平均更新 Q(s,a)←Q·α+R·(1-α)，并可在发现显著更优程序时触发 LLM prior 更新。

从编译框架角度拆解术语，比如术语所在编译框架的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 1（论文）的迭代流程（500 迭代上限 + early stopping K=200）：
```
R*←0; t*←0; p*←∅
for t=1..N:
    if t-t* > early_stopping: break
    S←G; π_seq←∅
    while S 已访问且非终止:            # Selection：Gumbel-augmented 下钻
        A←SelectBest(S); π_seq+= (S,A); S←T(S,A)
    if S 非终止:                       # Expansion：LLM 生成规则先验
        logits←LLM(S, Prompt)
        A←Sample(S, logits); π_seq+=(S,A); S←T(S,A)
    Sketches←GenerateProgramSketches(G, π_seq)   # Simulation
    (p,R)←FineGrainedParamSpec(Sketches)          # XGBoost+局部搜索+真机
    for (s,a) in π_seq: Q(s,a)←Q(s,a)*α+R*(1-α)   # Backpropagation
    if R>R*: R*←R; t*←t; p*←p
return p*
```
例子（GatedMLP）：LLM 识别 SiLU 可融合（SiLU(x)=x/(1+e^-x)）→ 给 AutoInline 高先验；识别 GEMM1/GEMM2 共享输入 X → 给 MultiLevelTiling+ComputeAtLocation 高先验 → MCTS 沿这些路径扩展 → 全融合单 kernel（S0→S5）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：LLM 每 MCTS 迭代仅调用一次（生成结构引导的规则概率，inference 延迟占编译总时间可忽略）；使用 gpt-5-preview (2025-07-01) via OpenAI API、temperature 0.8、8192 token 预算、失败回退随机游走策略、所有查询串行。MCTS 控制开销用 JAX 并行实现（<8% 编译时间）。使用方式：作为编译管线前端的调度搜索引擎，输出最优 TensorIR 调度序列；消融显示 LLM prior（DeepSeek-V3.2/Qwen3-max/GPT 5.0）比统计 prior（MLP/Random Forest）高 20%-30%，比随机 prior 的裸 MCTS 收敛更快。搜索时间开销（A100）：GatedMLP 1.37h/SelfAtten 1.83h/LoRA 1.92h/QKNorm 1.17h/nTrans 1.69h。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
