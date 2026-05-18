## Cumulative Acceptance Probability（累计接受概率）

术语是什么？通过联网搜索让回答具体和精准。
Cumulative Acceptance Probability (H_t) 是SADDLE用于自适应控制draft sequence length的核心指标。定义为H_t = ∏_{i=1}^{t} p_i，其中p_i是DLM在第i步生成的draft token x_i的采样概率P_{DLM}(x_i | x_{<i})。H_t的本质是对当前draft sequence中所有t个token都被TLM接受的概率的下界估计——由于speculative decoding的rejection sampling机制，一个token被接受的概率为min(1, p_T / p_D)，其中p_D = P_{DLM}(token)和p_T = P_{TLM}(token)。当p_D较低时，该token被拒绝的概率更高，H_t快速下降→触发early stopping。H_t单调递减（每步乘以p_i ∈ (0,1]），天然适合阈值决策。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
H_t的更新和决策机制：
```
// Per draft step t:
H_t = H_{t-1} * p_t  // single FP16 multiplication

// Stopping rule:
if H_t < τ:
    stop_drafting()
    
// 直观理解（τ = 0.3为例）：
// Step 1: p_1 = 0.9 → H_1 = 0.9   ≥ 0.3 → continue
// Step 2: p_2 = 0.8 → H_2 = 0.72  ≥ 0.3 → continue  
// Step 3: p_3 = 0.5 → H_3 = 0.36  ≥ 0.3 → continue
// Step 4: p_4 = 0.6 → H_4 = 0.216 < 0.3 → stop (draft length = 4)
```
H_t的理论基础：当DLM对某个token的信心（p_i）低时，该token更可能与TLM的distribution不一致→被拒绝的概率高→继续在此token之后drafting大概率产生无效计算。H_t作为累积度量同时惩罚连续低概率token（如连续3个0.5概率token使H_t降为0.125）和单个极低概率token（如一个0.1概率token直接触发停止）。

术语一般如何实现？如何使用？
SADDLE在Controller硬件中实现H_t更新：1KB SRAM存储每请求的当前H_t值，multipliers执行FP16乘法更新，comparators与τ比较。仅需轻量级on-chip memory操作，延迟可忽略（<1% of prediction latency）。H_t的threshold τ通过离线校准确定，不同model pair和dataset可有不同τ。论文观察到OPT-66B+OPT-1.3B在Dolly上τ≈0.3效果最优，但具体值需根据workload调整。

涉及论文标题：

