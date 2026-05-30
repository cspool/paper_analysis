## Mordal: Automated Pretrained Model Selection for Vision Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Mordal，一个自动化VLM预训练模型选择框架，包含三个核心算法组件：(1) Candidate Clustering——使用CKA (Centered Kernel Alignment) 计算vision encoder和language model的表示相似度，通过两步聚类（先聚类VE、再基于每个VE cluster的medoid聚类LLM）构建VLM候选聚类，每个cluster的候选具有相似性能；(2) Efficient Evaluation (Early Stopping)——采用Successive Halving Algorithm (SHA)在inter-cluster evaluation阶段早期淘汰表现差的cluster，每轮保留top 1/η候选，逐步增加评估budget；(3) Scaling Prediction——利用observational scaling law发现VLM alignment性能与训练数据量存在log-linear关系，通过对数线性回归从部分数据训练预测完整数据训练的最终性能，减少每个候选的评估时间。

  实验比较的baseline是grid search（穷举搜索），即对49个VLM候选（7个VE × 7个LLM）全部用full alignment data训练feature projector并评估。衡量指标：总搜索时间（GPU hours）、Top-1 model quality（accuracy）、Kendall's τ（Top-10候选排序一致性）。也对比了LLaVA-1.5-7B equivalent structure (CLIP-Vicuna) 的准确率。

- 硬件平台是什么，配置是什么。
  16× NVIDIA A40 GPU（每GPU 48 GB GDDR6），部署在cluster的一组VM上。软件：PyTorch（bfloat16精度）、HuggingFace Transformers、PEFT（LoRA）、Flash Attention-2。所有模型训练使用Adam optimizer（minibatch=4, initial lr=1e-4, linear schedule）。

- 模型是什么。数据集和bench分别是什么。
  模型：7个vision encoders（CLIP-ViT-L/14@336, SigLIP-so400m-patch14@384, DFN-CLIP-ViT-H/14@378, InternViT-300M/14@448, DINOv2-ViT-L/14@518, EVA-CLIP-02-ViT-L/14@336, ConvNeXt-L/14@256） + 7个LLMs（Vicuna-1.5-7B, Llama-2-7B, Llama-3-8B, Mistral-v0.2-7B, Qwen2-7B, Phi-3-Small-7B, Gemma-1.1-7B），共49个VLM候选。Feature projector: MLP (两层linear + GELU)。Alignment数据集：LLaVA-1.5-Instruction mixture。Benchmark数据集（3个domain 6个任务）：Visual QA — GQA, VizWiz；Doc QA — ChartQA, DocVQA；Knowledge — ScienceQA, AI2D。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/SymbioticLab/Mordal

  Mordal算法pipeline（以7个VE × 7个LLM = 49个candidate搜索GQA任务为例）：

  **阶段1: Candidate Clustering（候选聚类，Section 3.1 + Algorithm 2）：**
  ```
  # Step 1: Vision Encoder Clustering
  for each pair (VE_A, VE_B) in 7 vision encoders:
      # Pass calibration images through each VE, get output embeddings
      act_A = VE_A(images_from_target_task)   # shape: [N, D_ve_A]
      act_B = VE_B(images_from_target_task)   # shape: [N, D_ve_B]
      # Compute CKA similarity via MinibatchCKA
      cka_score = CKA(act_A, act_B)           # Equation (1)(2)
      dist = 1 - cka_score
      Dist_ve[VE_A][VE_B] = dist
  C_ve = HierarchicalClustering(Dist_ve, t_ve=0.7)
  # e.g., C_ve = [{CLIP, SigLIP, DFN-CLIP}, {InternViT, DINOv2}, {EVA-CLIP, ConvNeXt}]

  # Step 2: LLM Clustering (per VE cluster)
  for each VE_cluster in C_ve:
      medoid_ve = PickMedoidModel(VE_cluster)  # most central VE
      # Warm up a feature projector to match VE output to LLM input shape
      wp_projector = WarmupProjector(medoid_ve, alignment_data, rounds=10)
      fixed_ve_output = wp_projector(medoid_ve(images))

      for each pair (LLM_A, LLM_B) in 7 LLMs:
          # Feed fixed VE output to each LLM, get last hidden state
          rep_A = LLM_A.last_hidden_state(fixed_ve_output)
          rep_B = LLM_B.last_hidden_state(fixed_ve_output)
          cka_score = CKA(rep_A, rep_B)
          dist = 1 - cka_score
          Dist_llm[LLM_A][LLM_B] = dist

      C_llm = HierarchicalClustering(Dist_llm, t_llm=0.8)
      # e.g., for VE cluster 1: C_llm = [{Vicuna, Llama-2}, {Llama-3, Mistral, Qwen2}, {Phi-3, Gemma}]

      # Step 3: Cartesian product → VLM candidate clusters
      C_vlm.append(CartesianProduct(VE_cluster, C_llm))
      # e.g., for VE cluster 1: 2 × 3 = 6 candidate clusters
  # Total candidate clusters ≈ 10-15 (vs 49 individual candidates)
  ```

  **阶段2: Inter-cluster Evaluation with Early Stopping（Section 3.2 + Figure 5a）：**
  ```
  # Pick medoid candidate from each cluster as representative
  representatives = [PickMedoidCandidate(cluster) for cluster in C_vlm]
  # e.g., 10 representatives for 10 clusters

  # Successive Halving Algorithm (SHA)
  R = 0.125    # max data sample ratio (1/8 of alignment data)
  b = 0.03     # initial budget per candidate
  eta = 2      # reduction factor
  budget = b   # current budget per candidate

  while len(representatives) > top_k_inter (e.g., 3):
      for rep in representatives:
          # Train candidate with budget portion of alignment data
          train(rep, data_ratio=budget)
          # Evaluate on target task
          score[rep] = evaluate(rep, target_task)
      # Keep top 1/eta candidates
      keep = top_k(score, k=len(representatives) // eta)
      representatives = keep
      budget *= eta  # increase budget for next rung

      if convergence(representatives) or len(representatives) <= top_k_inter:
          break
  # After SHA: 3 representative candidates from 3 best clusters remain
  ```

  **阶段3: Intra-cluster Evaluation with Scaling Prediction（Section 3.2 + Algorithm 1）：**
  ```
  # Gather all individual candidates from remaining Top-K clusters
  C_remain = flatten(remaining_clusters)  # e.g., 3 clusters → 12-15 candidates

  # Scaling Prediction for each candidate
  for c in C_remain:
      P = []  # list of (log(r), log(Err)) pairs
      r = 0.125  # start from 1/8 data

      # Iteratively reduce data to find log-linear region
      while True:
          # Train from existing intermediate checkpoint if available
          train_from_checkpoint(c, data_ratio=r)
          Err = evaluate(c, target_task)
          P.append((log(r), log(Err)))

          if len(P) > p (e.g., 3):
              # Fit linear regression: log(Err) = α * log(r) + β
              f_c = LinearRegression(P)
              if fitting_loss(f_c) < delta (e.g., 5e-5):
                  break  # log-linear relationship confirmed

          r = r / u  # reduce data ratio (u = 2)

      # Predict performance at full data (r = 1)
      predicted_err = exp(f_c(log(1)))
      L.append((c, predicted_err))

  # Select candidate with best predicted performance
  best_candidate = argmin(L, key=lambda x: x[1])
  return best_candidate
  ```

  关键设计要点：
  - CKA可比较不同shape的表示（传统cosine similarity不可），且对MLP projection变换鲁棒
  - 两步聚类（先VE后LLM）避免对所有VE×LLM pair计算CKA，减少pair-wise计算量
  - SHA提供rough filtering，Scaling Prediction提供fine-grained ranking——二者正交
  - Scaling prediction利用observational scaling law：VLM alignment性能与训练数据量存在log-linear关系，但仅在一定数据量后出现
  - 聚类阈值t_ve=0.7和t_llm=0.8平衡聚类粒度和搜索效率
  - 除feature projector外，可使用LoRA fine-tune pretrained LLM
  - Flash Attention-2用于高效attention计算
  - Mordal自动将空闲GPU资源分配给未收敛candidate

- 关键实验结果：
  - Mordal 8.9×–11.6× faster than grid search（5439 GPU hours → 469-607 GPU hours）
  - 6个task中5个成功选出Top-1 candidate（1个选出Top-2，因最优candidate属于表现差的cluster被过早淘汰）
  - 所有找到的最优VLM性能均超过LLaVA-1.5-7B equivalent (CLIP-Vicuna)
  - Kendall's τ: 0.76–0.96（Top-10候选排序一致性）
  - Top-5候选中识别出4/5（GQA和AI2D）
  - Ablation: candidate clustering + early stopping + scaling prediction三者协同效果最优
  - 敏感性分析：Mordal对大多数超参数鲁棒，clustering threshold t_ve影响最大
