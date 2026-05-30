## FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 论文提出三种算法层面的训练优化，用于加速大规模 MoE 模型的分布式训练：
    1. **Dynamic Shadowing（动态影子策略，Section 4.1）**：在运行时按迭代动态选择热门 expert，将其模型参数广播复制到所有 worker，替代原本的大量 token 输入传输。通过性能模型比较 `Lat_imbl`（不均衡下的延迟）和 `Lat_shadow`（影子化后的延迟），当满足条件 `B_max > rαH` 或 `(3(B_max - B'_max)αH) / (rαH - B_max) > P/W_net` 时启用。伪代码见 Algorithm 1（SelectShadowExperts）。
    2. **Fine-grained Smart Scheduling（细粒度智能调度，Section 4.2）**：将 all-to-all 通信拆分为分组 pairwise exchange 操作，计算也相应拆分。创建独立的 computation stream 和 communication stream，将 n 个 group 的 S（send）、C（compute）、R（receive）操作按依赖关系重新排列，使通信与计算异步并行执行。将最快的操作 S_{i,0} 和 R_{i,n-1} 放在首尾以最小化开销。
    3. **Topology-aware Gate（拓扑感知门控，Section 4.3）**：修改 expert 选择策略，限制跨节点 tokens 数量上限 L = (W_net / (M·W_local))·B，超过 L 的 tokens 在本地节点内重新选择 expert，减少上层网络链路拥塞。
  - 实验比较：
    - 整体加速比 vs ZeRO stage 1/2/3（数学等价，不修改 expert 选择）和 FastMoE（expert parallelism baseline）
    - 动态影子策略单独效果（迭代延迟 vs 影子化 expert 数量）
    - 智能调度单独效果（每层实际加速比 vs 理论上界 `(Lat_comm + Lat_comp) / max{Lat_comm, Lat_comp}`）
    - 拓扑感知门控 vs GShard、BASE Layer 的收敛速度（training loss vs time/iterations）
    - 性能模型预测准确度（computation/communication 单独 + end-to-end，R²=0.987/0.967）

- 硬件平台是什么，配置是什么。
  - **johnny 集群**：16× NVIDIA Tesla V100-PCIE 32GB GPU，2 节点（每节点 8 GPU），GPU 通过 PCIe switch 连接 2 个 CPU socket。Infiniband EDR 但因缺少 ×16 PCIe 插槽实际带宽降级至 50Gb/s。
  - **trevor 集群**（天河二号超算分区）：64× NVIDIA Tesla V100-SXM2 32GB GPU，16 节点（每节点 4 GPU），节点内 NVLink 互连（异构环，半数边双链路双带宽）。Infiniband EDR 100Gb/s 节点间通信。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    | 模型 | 参数量 | 层数 | Experts | H（hidden） | α（MLP中间维度比） | 集群 |
    |------|--------|------|---------|-------------|-------------------|------|
    | MoE-GPT-S | 0.86B | 12 | 16 | 1024 | 2 | johnny |
    | MoE-GPT | 3.42B | 12 | 16 | 2048 | 2 | johnny |
    | MoE-GPT-L | 13.7B | 12 | 16 | 4096 | 2 | johnny |
    | MoE-BERT-Deep | 1.71B | 24 | 16 | 1024 | 2 | johnny |
    | MoE-BERT-Deep-L | 27.4B | 24 | 16 | 4096 | 2 | johnny |
    | MoE-BERT-Wide | 3.27B | 12 | 64 | 1024 | 2 | trevor |
    | MoE-BERT-Wide-L | 13.1B | 12 | 64 | 2048 | 2 | trevor |
  - **数据集**：
    - 性能评测使用 expert selection dataset（从真实训练过程中记录的 token-to-expert 分配），可在 https://pacman.cs.tsinghua.edu.cn/laekov/fastermoe-data/dumps.tgz 下载，为 16 experts 生成。
    - 训练实验使用预处理后的 wikidataset，可在 https://pacman.cs.tsinghua.edu.cn/laekov/fastermoe-data/wikidataset.tgz 下载。
  - **Bench**：无标准 benchmark 数据集。性能延迟测量基于重放 expert selection dataset，收敛实验基于 wikidataset 训练 loss 曲线。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：源代码开源于 https://github.com/thu-pacman/FasterMoE，artifact evaluation 脚本在 https://github.com/laekov/fastermoe-ae。
  - **系统依赖**：基于 FastMoE [7] 实现，依赖 CUDA、NCCL（≥2.9.9）、PyTorch（v1.10.0）。Baseline 系统包括 Megatron-LM（修改 MLP 模块用于 MoE）、DeepSpeed v0.4.4（ZeRO Optimizer）、FairSeq（BASE Layer）。
  - **算法 pipeline 解释（以动态影子策略为例，伪代码对应 Algorithm 1）**：

  ```
  # Dynamic Shadowing 核心流程（Algorithm 1: SelectShadowExperts）
  # 输入: B[N], 每个worker的batch size（token数量）
  # 输出: E_s, 需要影子化的expert集合
  # 每iteration在所有worker上执行

  def SelectShadowExperts(B):  # B[w] = sum_i T[i][w]
      B_max = max(B)
      c_min = Lat_imbl(B_max)    # 当前不均衡配置的延迟
      E_s = []                    # 影子化expert集合

      # 按local batch大小降序遍历
      for i, B_i in sorted(enumerate(B), key=lambda x: -x[1]):
          B_i = T[i][i]           # 保留本地tokens
          for j != i:             # 其他worker的tokens: 在本地计算
              B_i += T[i][i]      # 影子化后在本地执行

          B_max_prime = max(B)    # 影子化后的最大batch
          c = Lat_shadow(len(E_s)+1, B_max_prime)

          if c < c_min:           # 影子化降低延迟则采纳
              c_min = c
              E_s.append(i)
          else:
              return E_s          # 一旦不改善即停止
      return E_s

  # 影子化延迟模型 (Eq. 8):
  # Lat_shadow(r, B') = max_w{3 * 4*B'_w*α*H²/P} + 2r * 2αH²/W_net
  #   - 第一项: 影子化后均衡的computation
  #   - 第二项: 广播r个expert参数的通信开销（forward 1次 + backward reduce 1次）

  # 影子化启用条件 (Eq. 9/10):
  # 条件1: B_max > rαH  (token传输开销 > 模型传输开销)
  # 条件2: 3(B_max - B'_max)αH / (rαH - B_max) > P/W_net  (减少的computation > 增加的communication)
  ```

  **拓扑感知门控算法（Section 4.3）**：

  ```
  # 拓扑感知门控：限制跨节点tokens
  # L = (W_net / (M * W_local)) * B
  #   W_net: 跨节点带宽, W_local: 节点内带宽
  #   M: 每节点worker数, B: batch size

  def TopologyAwareGate(tokens, scores, L):
      for each token x with top-k expert scores:
          if expert_is_on_remote_node(x.best_expert):
              remote_candidates.append((x, x.score))
  
      # 仅允许分数最高的L个跨节点
      remote_candidates.sort(key=lambda t: -t[1])
      allowed = remote_candidates[:L]
  
      # 其余token在本地节点内重新选择expert
      for token in remote_candidates[L:]:
          token.reselect_expert(local_node_only=True)
  ```

  **智能调度张量计算流（Section 4.2）**：
  ```
  # n个group, 每个worker在step j执行:
  # S_{i,j}: 发送tokens到group (i+j) mod n，接收来自group (i-j) mod n
  # C_{i,j}: 对来自group (i-j) mod n的tokens用本地expert计算
  # R_{i,j}: 接收本地tokens输出从group (i+j) mod n，发送输出到group (i-j) mod n
  #
  # comm stream: S_{i,0}, S_{i,1}, ..., S_{i,n-1}, R_{i,0}, ..., R_{i,n-1}
  # comp stream: C_{i,0}, C_{i,1}, ..., C_{i,n-1}
  # 两stream并行执行，依赖关系由数据依赖保证
  ```
