## MagicPIG: LSH Sampling for Efficient LLM Generation

- 属于算法pipeline的实现是什么？实验比较什么？
  提出基于Locality Sensitive Hashing (LSH) 的采样方法来近似self-attention计算，替代传统的TopK稀疏注意力。将attention output视为从attention score分布w中采样value的期望值（o = E_{i~w}[v_i]），利用Self-normalized importance sampling + LSH SimHash从key分布中高效采样，实现对attention输出的无偏估计。实验比较了MagicPIG vs Full Attention、Quest（dynamic sparse attention）在lm-eval-harness (GSM8K-CoT, MMLU, COQA)、LongBench (QASPER, LCC, Repobench-P, TriviaQA等)和RULER (13个合成任务，16K-256K context)上的准确率，以及不同硬件下(A100, L20, RTX 4090)的解码吞吐量和延迟。

- 硬件平台是什么，配置是什么。
  GPU: NVIDIA A100-80GB (搭配CodeLlama-34B, 16K context)、NVIDIA L20-48GB (搭配CodeLlama-13B, 16K context)、模拟RTX 4090-24GB (L20带宽限制, 搭配Llama-3.1-8B-Instruct, 96K context)；CPU: Intel Platinum 8480+ (A100场景)、Intel 8563C (L20场景)。

- 模型是什么。数据集和bench分别是什么。
  模型: Llama-2-7b-chat、Llama-3.1-8B-Instruct、Code-Llama-13b-16K、Code-Llama-34b-16K、MegaBeam-Mistral-7B-512K、Llama3-8B-Prolong-512K、Llama-3.1-70B-Instruct。
  数据集/benchmark: lm-eval-harness (GSM8K-CoT, MMLU-Flan-Cot-Fewshot, COQA)、LongBench (QASPER, LCC, Repobench-P, TriviaQA, PRE, TREC)、RULER (13个合成任务含NIAH single/multi-key, CWE, FWE等)、infini_igsm (4K/8K close reasoning tasks)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接: https://github.com/Infini-AI-Lab/MagicPIG。算法pipeline核心流程：

  1. **预处理(centering)**：对K cache做中心化，K = K - mean(K)，解决q和k方向几乎相反导致LSH采样失效的问题。
  2. **LSH哈希表构建**：用K×L个随机投影向量W (SimHash)，将每个k_i投影到K-bit哈希码，构建L张哈希表。
  3. **解码时每步**：
     a. GPU上计算q的哈希码：q_code = Sign(q @ W)，得到K×L bit。
     b. CPU上查询L张哈希表，收集至少在2张表中与q碰撞的key索引集合S。
     c. CPU上计算采样概率u_i = 1 - (1-p_i^K)^L - L·p_i^K·(1-p_i^K)^{L-1}，其中p_i = 1 - arccos(q·k_i/(|q|·|k_i|))/π。
     d. 计算注意力输出估计：ō = Softmax(w_S/√d - log(u))·V_S，其中w_S = q·K_S^T。
  4. On-device cache: sink tokens和local tokens保留在GPU，不经过LSH采样，通过recursive attention合并GPU/CPU结果。
  张量计算维度：q∈R^{1×d}, K,V∈R^{n×d}, W∈R^{d×(K×L)}, 典型参数K=8~10, L=75~300, 计算量为全注意力的2%~5%。
