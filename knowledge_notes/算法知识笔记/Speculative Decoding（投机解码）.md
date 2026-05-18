## Speculative Decoding（投机解码）

术语是什么？通过联网搜索让回答具体和精准。
Speculative Decoding（投机解码）是一种加速LLM自回归推理的算法范式。核心思想是"先小模型起草，后大模型验证"：使用一个轻量级draft model快速生成γ个候选token，再由完整的target model通过一次并行forward pass验证所有候选，通过概率接受机制(acceptance probability α_i = min(1, p(t_i)/q(t_i)))保证输出分布与target model原生自回归解码完全一致。理论加速比Speedup = c·γ / ((1-ρ)·c·γ + c·ρ + 1)，其中c为target/draft速度比、ρ为平均接受率、γ为draft长度。当c≫1且ρ→1时，加速比逼近γ。DFVG是ASPLOS'26上首个将speculative decoding完整映射到FPGA+GPU异构硬件的系统。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Speculative Decoding的单次迭代pipeline：
```
给定: prefix X_1:j, draft model M_q, target model M_p, draft length γ

// Step 1: Draft Generation (自回归，draft model逐token)
for i = 1 to γ:
    x̃_{j+i} ~ M_q(· | X_1:j+i-1)  // draft model逐token生成候选
    q_i = q(x̃_{j+i} | X_1:j+i-1)  // 记录draft概率

// Step 2: Target Verification (并行，target model一次forward)
p_1:γ = M_p(X_1:j+γ)  // target model并行计算所有位置的真实概率

// Step 3: Probabilistic Acceptance (逐token验证)
for i = 1 to γ:
    α_i = min(1, p_i(x̃_{j+i}) / q_i(x̃_{j+i}))  // 重要性采样接受概率
    if random() < α_i:
        accept x̃_{j+i}  // 接受此token
    else:
        // 拒绝，从修正分布重采样
        p'_i = norm(max(0, p_i - q_i))
        x_{j+i} ~ p'_i
        break  // 后续所有候选token丢弃
```
Tree-based变体：draft model每步生成k个分支候选（而非单token），形成token tree。验证时target model并行计算tree中所有节点概率，按top-down（SpecInfer的OT方法）或bottom-up（Traversal Verification）选择最长有效前缀。DFVG的ADAPT算法动态决定每层分支数k_i和tree深度D。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流实现：
- **SpecInfer** (ASPLOS'24)：多GPU tree-based speculative decoding，静态预定义branch配置，optimal transport tree verification
- **EAGLE** (ICML'24)：feature-level speculative decoding，利用draft model的feature uncertainty替代token probability
- **Medusa** (ICML'24)：在target model上加多个extra decoding heads并行预测多token
- **DuoDec** (2025)：CPU+GPU异构，hardware-aware draft budgeting
- **DFVG** (ASPLOS'26)：FPGA+GPU异构，ADAPT动态tree构建+TreeSort-Verify高效验证，开源https://github.com/ShaoqiangLu/DFVG
- **AdaServe** (2026)：构建于FlexFlow Serve之上，SLO-customized speculative decoding，将multi-SLO serving形式化为budget-constrained token tree构造，使用beam search speculation + two-phase selection (SLO-customized + throughput-optimized) + tree-based verification，开源https://github.com/zikun-li/AdaServe-Artifact-Evaluation
适用场景：所有LLM自回归解码，2×-4×加速比，数学等价于原生解码（无质量损失）。

AdaServe 的 SLO-customized speculative decoding：将 multi-SLO serving 与 speculative decoding 结合，形式化为带 budget 约束的 token tree 构造问题。小 draft model 用 beam search 生成 candidate token tree，CPU scheduler 按 SLO 需求选节点（SLO-customized selection），剩余 budget 按全局 path probability 分配（throughput-optimized selection），target LLM 并行做 tree-based verification。同一 batch 中不同 SLO 请求可在同一次 verification 中前进不同 token 数。

涉及论文标题：
- DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU
- Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems
- AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding

