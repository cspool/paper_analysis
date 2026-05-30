## FineMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving

- 属于算法pipeline的实现是什么？实验比较什么？
  - FineMoE 的算法 pipeline 创新包含：
    1. **Expert Map 数据结构**：记录每个 inference iteration 中每层 gate network 输出的全概率分布 P_l^{(i)} ∈ R^J（而非 coarse-grained 的 binary 激活或 hit count）。expert map 可退化恢复 coarse-grained 信息（对概率分布取 top-K + 聚合迭代）。直观上，expert map 不仅识别哪些 experts 被选择，还捕获 gate network 对所有 experts 的 confidence/preference 分布。
    2. **Semantic-based Expert Map Search**：利用 MoE 模型的 embedding layer 输出作为 semantic embedding，与 Expert Map Store 中历史 semantic embeddings 计算 pairwise cosine similarity，选择最相似的 historical iteration 的 expert map 指导 expert prefetching。该方法基于"语义相似的 prompts 具有相似的 expert 选择模式"的假设。
    3. **Trajectory-based Expert Map Search**：对第 l ∈ [d+1, L] 层，收集前 (l-d) 层已观察到的 expert probability trajectory（即前序层的 P_1,...,P_{l-d}），与 Expert Map Store 中历史 expert maps 对应的前 l-d 层计算 cosine similarity，选择最匹配的 expert map 的 P_l 指导当前层 prefetching。
    4. **Similarity-aware Dynamic Expert Selection**：对每层 l 根据 search confidence（cosine similarity score）动态计算 selection threshold δ_l = Clip(1-score, 0, 1)。高 similarity → 低 δ → 选择 fewer high-probability experts；低 similarity → 高 δ → 选择 more experts 防止 miss。
    5. **Expert Map Deduplication**：通过 unified redundancy score RDY = (d/L)*score^{sem} + ((L-d)/L)*score^{traj} 评估新 iteration 与 Expert Map Store 中旧 iterations 的冗余度，达到 capacity C 时剔除最相似（冗余）的旧 map 以维持多样性。理论分析表明保持 2LJ expert maps 可保证 ≥75% similarity lower bound，保持 (1/2)LJ*ln(LJ) maps 可保证 ≥98% similarity。
  - 实验比较（算法层面消融）：
    - Expert pattern tracking 对比：1) Speculate（speculative prediction），2) Hit count（request-level, MoE-Infinity 方式），3) Map(T)（仅 trajectory similarity），4) Map(T+S)（trajectory + semantic 但 static top-K selection），5) Map(T+S+δ)（全部 feature 启用）。结果：随 feature 递增恢复，expert hit rate 逐步提升
    - Pearson correlation 验证：semantic similarity 和 trajectory similarity 均与 expert hit rate 呈正相关（所有模型/数据集组合 Pearson coefficient > 0）

- 硬件平台是什么，配置是什么。
  - 主测试台：6× NVIDIA GeForce RTX 3090 24GB, NVLink, PCIe 4.0 32GB/s, AMD Ryzen Threadripper PRO 3955WX 32C, 480GB CPU RAM
  - 高配测试台：NVIDIA A100 80GB HBM2e, 2 TB/s 峰值带宽

- 模型是什么。数据集和bench分别是什么。
  - **模型**：Mixtral-8×7B（12.9B active / 46.7B total, 2/8 experts/layer, 32 layers）、Qwen1.5-MoE（2.7B active / 14.3B total, 4/60 experts/layer, 24 layers）、Phi-3.5-MoE（6.6B active / 42B total, 2/16 experts/layer, 32 layers）
  - **数据集**：LMSYS-Chat-1M、ShareGPT；online 实验使用 Azure LLM 推理 traces 驱动请求
  - **Metrics**：TTFT（prefill）、TPOT（decode）、expert hit rate、CDF of end-to-end request latency

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：FineMoE 原型基于 MoE-Infinity 代码库（https://github.com/TorchMoE/MoE-Infinity），自身未发现独立开源仓库
  - **Expert Map Search 伪代码**：
  ```
  # Input: 新 prompt 的 semantic embedding sem_new ∈ R^{B×h}
  #        Expert Map Store: sem_old ∈ R^{C×h}, map_old ∈ R^{C×L×J}
  #        prefetch distance d, total layers L

  # 对每层 l ∈ [1, L]:
  for l in range(1, L+1):
      if l <= d:
          # Semantic-based search for initial layers
          score_sem = cosine_similarity(sem_new, sem_old)  # R^{B×C}
          best_iter = argmax(score_sem, dim=-1)  # 每 batch 选最相似 iteration
          # 使用 best_iter 的 expert map 中前 d 层指导 prefetch
          for target_l in range(1, d+1):
              P = map_old[best_iter, target_l, :]  # R^{B×J}
              prefetch_experts(P, scores=score_sem)
      else:
          # Trajectory-based search for later layers
          traj_new = concat([P_1, ..., P_{l-d}])  # R^{B×(l-d)J}
          traj_old = map_old[:, :(l-d), :].reshape(C, -1)  # R^{C×(l-d)J}
          score_traj = cosine_similarity(traj_new, traj_old)  # R^{B×C}
          best_iter = argmax(score_traj, dim=-1)
          P = map_old[best_iter, l, :]  # R^{B×J}
          prefetch_experts(P, scores=score_traj)
  ```
  - **Similarity-aware Expert Selection 伪代码**：
  ```
  # Input: searched P_l ∈ R^J (probability distribution), score ∈ [-1, 1]
  def select_experts_to_prefetch(P_l, score, K):
      δ = clip(1 - score, 0, 1)  # 相似度低时选更多 experts
      sorted_experts = argsort(P_l, descending=True)
      E_prefetch = []
      sum_p = 0.0
      for j in sorted_experts:
          E_prefetch.append(j)
          sum_p += P_l[j]
          if sum_p >= δ and len(E_prefetch) >= K:
              break
      return E_prefetch
  ```
  - **关键张量计算流**（Mixtral-8×7B, L=32, J=8, K=2, B=1）：
    1. Semantic embedding extraction: token_ids → embedding_layer → sem_new ∈ R^{1×4096}
    2. Semantic search: cos_sim(sem_new, sem_old[1..C]) → score_sem ∈ R^{1×C} → select best_iter
    3. Layer 1 (l ≤ d=3): P_1 = map_old[best_iter, 0, :] = {p_{1,1},...,p_{1,8}} → δ = clip(1-score_sem, 0, 1) → select experts until Σp ≥ δ and |E| ≥ 2 → prefetch E_prefetch
    4. Layer 4 (l > d): traj_new = concat(P_1, P_2, P_3) ∈ R^{1×24} → cos_sim(traj_new, traj_old[1..C]) → score_traj → best_iter → P_4 from map_old[best_iter] → select experts with δ = clip(1-score_traj, 0, 1)
    5. Repeat to Layer 32
