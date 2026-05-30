## FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - FedMoE 提出一个基于 MoE 架构的个性化联邦学习框架，包含两阶段训练流程：
    1. **Stage One: Coarse-grained Submodel Initialization**：首先通过 PEFT（LoRA）在客户端进行少量轮次（约5轮）内存高效微调，收集各 expert 的激活概率（$p_{i,j} = n_{i,j}/N$）。对于内存不足的客户端，云端基于同任务其他客户端的数据量加权平均估算其激活概率。然后云端对每个客户端执行启发式子模型搜索——建模为优化问题：在内存约束下最大化每层保留 expert 的激活概率阈值 θ。使用二分搜索在 [0,1] 范围内寻找最优 θ，对每个 θ 构建满足阈值的最小 expert 子集，验证是否超出内存限制 $\alpha \cdot M$，动态调整上下界。
    2. **Stage Two: Federated Training and Fine-grained Submodel Adjustment**：
       - **Modular Aggregation**：dense 层使用 FedAvg 聚合；sparse 层按模块粒度——未被任何客户端激活的 expert 保持不变，仅被单个客户端使用的 expert 直接更新，被多个客户端共享的 expert 使用 FedAvg 聚合。Router 对应的维度按相同模式更新。
       - **Expert Recommendation**：若客户端多轮性能无提升（达到瓶颈），云端基于各客户端 expert 激活概率的 cosine similarity（Eq. 4）找到 top-K 最相似客户端作为参考。若参考客户端平均 expert 数多于当前客户端，则推荐增加 expert（按估算激活概率 $\hat{p}_{expert}$ 排序）；否则推荐裁剪低效 expert。调整具有探索性，若性能未改善则回退并固定结构。
  - 实验比较：
    - End-to-end 性能：FedMoE vs randomMoE（随机选 expert 子集）、FedProx（正则化联邦优化）、SCAFFOLD（控制变量）在 4 种 FL 设置下对比 task performance + communication volume + memory usage
    - 收敛速度：FedMoE vs 三 baseline 的 99%/90% 相对目标性能加速比
    - 鲁棒性：各方法在不同设置下的 Coefficient of Variation (CV) 和 Composite Variation Index (CVI)
    - 消融实验：FedMoE vs w/o stage1 vs w/o stage2 对比任务性能和 expert 数量演变

- 硬件平台是什么，配置是什么。
  - 模拟 FL 环境：30 个客户端，每轮随机选择 5 个（Standard 设置）或强制选择 3 个不同任务类型的客户端（Enforced 设置）。
  - 客户端内存容量：18GB–24GB，典型高性能智能手机和边缘计算平台。
  - 云端服务器：维护全局 MoE 模型，执行子模型搜索、聚合和 expert 推荐。
  - 训练使用 Hugging Face Transformers 框架，模型权重从 Hugging Face 直接下载预训练 Switch Transformers。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：Switch Transformers 架构。FedMoE 和 randomMoE 的全局模型配置为每层 32 experts；FedProx 和 SCAFFOLD 由于边缘设备内存限制配置为每层 8 experts。预训练权重从 Hugging Face 下载。PEFT 阶段使用 LoRA 进行内存高效微调。
  - **数据集**：
    - AG News（文本分类 task-TC，评价指标 accuracy）
    - SQuAD（阅读理解 task-RC，评价指标 F1 score）
    - XSum（文本摘要 task-TS，评价指标 Rouge-2）
  - **FL 设置**（4 种模拟真实场景）：
    1. Standard-Hetero-T：30 客户端，异构任务，每轮随机选 5 个
    2. Standard-Hetero-TD：在 Hetero-T 基础上引入 label-skewed non-IID 数据分布（不均匀标签分配）
    3. Enforced-Hetero-T：强制每轮选 3 个不同任务类型客户端，制造更强冲突
    4. Enforced-Hetero-TD：Enforced 客户端选择 + label-skewed non-IID 数据
  - **Benchmark 指标**：task accuracy/F1/Rouge-2，communication volume (GB)，peak memory usage (GB)，convergence speedup，CV/CVI 鲁棒性指标。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：论文未明确说明开源代码仓库链接。
  - **算法 pipeline 解释（Stage One 启发式子模型搜索）**：

  ```
  # 输入: 每层所有 expert 的激活概率 p[i][j], 内存约束 M, 预留比例 α
  # 输出: 每层保留 expert 的二值指示 x[i][j]

  def HeuristicSubmodelSearch(p, M, α, w_d, w_expert):
      lo, hi = 0.0, 1.0
      while hi - lo > epsilon:
          θ = (lo + hi) / 2
          feasible = True
          for each layer i:
              sorted_experts = sort_by_p_desc(i)  # 按激活概率降序
              cum_prob = 0
              for expert j in sorted_experts:
                  x[i][j] = 1
                  cum_prob += p[i][j]
                  if cum_prob >= θ:
                      break
          # 计算子模型总内存
          mem = w_d + sum(x[i][j] * w_expert[i][j])
          if mem <= α * M:
              lo = θ  # 可行，尝试更大 θ
          else:
              hi = θ  # 不可行，减小 θ
      return x  # 最优 expert 映射
  ```

  **Stage Two Federated Training（Algorithm 1 核心流程）**：

  ```
  for each round r = 1..R:
      S ← sample subset of clients
      for client u_k in S (parallel):
          w_k = subsample(global_model, client_expert_map[u_k])
          w_k* = TRAIN(w_k, D_k_train)       # 本地微调
          p_all, acc = VALIDATE(w_k*, D_k_val)  # 收集激活概率和验证分数
          send (w_k*, p_all, acc) to server

      # Modular Aggregation
      for param in dense_layers:
          w_global[param] = FedAvg(w_k*[param] for k in S)
      for expert e in sparse_layers:
          clients_with_e = {k: e ∈ w_k for k in S}
          if len(clients_with_e) == 0: continue   # 未激活不变
          elif len(clients_with_e) == 1:           # 单客户端直接更新
              w_global[e] = w_k*[e]
          else:                                    # 多客户端 FedAvg
              w_global[e] = FedAvg(w_k*[e] for k in clients_with_e)

      # Expert Recommendation
      for client u_k in S:
          if acc not improved:
              # 计算与其他客户端的 cosine similarity
              sim(u_k, u_a) = cosine_sim(p_u_k, p_u_a)  # Eq. 4
              S' ← top K similar clients
              n = AVG(n_expert(S')) - n_expert(u_k)
              if n > 0:  # 增加 expert
                  for expert outside w_k:
                      p_hat = weighted_avg_p(sim, p_from_S')  # Eq. 6
                  E ← top n experts by p_hat
                  add E to w_k
              else:      # 裁剪 expert
                  for expert inside w_k:
                      p_hat = weighted_avg_p(sim, p_from_S')
                  E ← top |n| experts by lowest p_hat
                  remove E from w_k
              if adjusted model not improved:
                  revert and fix structure
  ```

  **Modular Aggregation 张量计算**（以单个 MoE 层为例）：
  ```
  # 全局模型: W_global = {W_dense, W_router, W_expert[0..E-1]}
  # 客户端 k 的子模型: W_k = {W_dense, W_router[kept], W_expert[kept]}

  # Dense 层聚合 (FedAvg)
  W_dense_new = (1/|S|) * Σ_k W_k_dense

  # Sparse 层聚合 (Modular)
  for expert j in 0..E-1:
      S_j = {k: expert j ∈ W_k}
      if len(S_j) == 0:
          W_expert_new[j] = W_expert_old[j]  # 不变
      elif len(S_j) == 1:
          W_expert_new[j] = W_k_expert[j]    # 直接更新
      else:
          # FedAvg 加权聚合
          n_total = Σ_{k∈S_j} |D_k|
          W_expert_new[j] = Σ_{k∈S_j} (|D_k|/n_total) * W_k_expert[j]

      # Router 对应维度同步
      W_router_new[j] = same_pattern_as_expert(S_j)
  ```

  **Cosine Similarity for Expert Recommendation (Eq. 4)**：
  ```
  # u_k, u_a: 两个客户端在全部 expert 上的激活概率向量
  # p_{i,j} 为第 i 层第 j 个 expert 的激活概率
  sim(u_k, u_a) = Σ_i Σ_j p_{i,j}(u_k) · p_{i,j}(u_a)
                  / (||p(u_k)|| · ||p(u_a)||)
  ```

  **关键性能数据**：
  - Standard-Hetero-T: FedMoE 94.76/86.64/16.92 (TC/RC/TS) vs FedProx 92.92/87.99/11.94（FedProx 在 RC 上略优但 TS 显著落后）
  - Communication volume: FedMoE 1.76GB vs FedProx 2.30GB（−23.5%）
  - Memory usage: FedMoE 13.44GB vs FedProx 24.71GB（−45.6%）
  - 收敛加速（Enforced-Hetero-T）：1.35×–2.92× vs baselines（90% target）
  - Ablation: w/o stage1 性能显著下降（TS: 14.50 vs 16.92），expert 数不减反增（96→104）；w/o stage2 expert 数保持 78 不变
