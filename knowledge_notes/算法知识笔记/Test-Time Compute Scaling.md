## Test-Time Compute Scaling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Test-Time Compute Scaling指在推理阶段通过增加计算预算来提升模型准确率的策略族。两种主要形式：(1) Sample-based scaling——生成k个独立样本（使用非零temperature），通过majority voting（self-consistency, Wang et al. 2023）选出最终答案；(2) Length-based scaling——增加单个样本的最大生成长度（longer chain-of-thought），给模型更多"思考token"。M1论文的核心创新在于将Mamba的推理速度优势（3x faster throughput vs Transformer）转化为test-time compute scaling的准确率增益：同等wall-clock时间预算下，M1可生成更多样本或更长序列，从而获得更高的majority voting accuracy或long-CoT accuracy。评估使用pass@k指标（k个样本中至少一个正确的unbiased概率估计）和Maj@k（majority voting accuracy）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Pass@k unbiased estimation (Chen et al., 2021)
# N=64 total samples per problem, c=#correct, k=budget
pass_at_k = 1 - C(N-c, k) / C(N, k)  if N-c >= k else 1.0
# 使用numerically stable实现 (Chen et al. 2021)

# Majority Voting (Self-Consistency)
answers_per_question = []
for i in range(k):  # k in {1, 16, 32, 64}
  output = model.generate(question, temperature=0.7, max_len=8k)
  answer = extract_boxed(output)  # 从\boxed{...}提取最终答案
  answers_per_question.append(answer)
final_answer = majority_vote(answers_per_question)
accuracy = mean(final_answer == ground_truth)

# M1的速度→准确率转换 (核心创新)
# 最优吞吐量 (通过sweep batch size找到):
# M1-3B: 15169 tokens/s  → 每8k样本≈0.53s
# R1-1.5B-Qwen: 7263 tokens/s → 每8k样本≈1.10s
# 
# 16s wall-clock budget:
# M1可生成 16/0.53 ≈ 30 samples → Maj@30 accuracy
# R1可生成 16/1.10 ≈ 15 samples → Maj@15 accuracy
# M1用更多样本弥补单样本quality的微小差距

# Length-based scaling:
# 固定k=1, 变化max_len ∈ {2k,4k,8k,16k,24k}
# M1生成24k tokens需 24k/15169 ≈ 1.58s
# R1生成24k tokens需 24k/7263 ≈ 3.30s
# 同等时间下M1可生成更长CoT → 更高accuracy (Figure 4 right)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
M1使用VeRL evaluation tools进行评估：temperature=0.7, max_len=32k, pass@1 averaged over 64 runs, Maj@k repeated 100 times以减少统计方差。评估prompt统一为"Let's think step by step and output the final answer within \boxed{}"。最优吞吐量确定方法：从batch size=8开始逐步增加直到throughput decrease，记录峰值tokens/s。Test-time compute scaling适用于任何有高效推理的模型+可自动验证答案的任务（数学、代码），核心trade-off是compute budget vs accuracy gain。M1证明了对Mamba架构，将速度增益转化为准确率增益是可行的实践策略。

涉及论文标题：
- M1__Towards_Scalable_Test-Time_Compute_with_Mamba_Reasoning_Models
