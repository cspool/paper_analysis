## GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - GRACE-MoE 提出一套 lossless 协同优化框架，通过三个算法层面的设计联合优化 SMoE 推理中的通信开销和计算负载不均衡：
    1. **Non-Uniform Hierarchical Expert Grouping（Section 4.1）**：基于 spectral clustering 对 expert affinity matrix（共激活频率）进行分层分组——跨节点层面使用 fully non-uniform grouping 最大化 intra-node affinity 以最小化跨节点通信；节点内 GPU 间使用 controlled non-uniform grouping（由 non-uniformity ratio r 控制 size deviation δ=E·r）。通过绘制 (S(r), U(r)) 曲线选择 knee point 作为最优 r。Algorithm 2 给出完整 controlled non-uniform grouping 流程（光谱聚类→trim oversized groups→reassign overflow experts→balance undersized groups）。
    2. **Dynamic Expert Replication based on Load Skew（Section 4.2）**：定义 computational load skew factor ρ = Wmax/W，由 n_replica = min(max(1, floor(ρ)), n_gpu-1) 动态确定副本数。仅复制 heaviest group 中最热的 expert（cumulative load > Wmax·n_replica/(1+n_replica)），副本放置到 n_replica 个最空闲 GPU 作为 secondary copies。
    3. **Topology-Aware Routing with Locality Preference（Section 4.3）**：三级 locality-first 策略——(i) 优先同 GPU 副本；(ii) 其次同节点内其他 GPU 副本；(iii) 最后跨节点副本。每级内使用 Weighted Round-Robin with Load Prediction（基于 pre-replication load stats 预测 post-replication GPU 负载，weights ∝ 1/load）。
  - 实验比较：(1) 端到端性能：GRACE-MoE vs Tutel, Megablocks, vLLM, C2R, Occult（No-Prune）在 OLMoE/DeepSeek-v2-lite-chat/Qwen3-30B-A3B 三个 MoE 模型上，2 nodes×2 GPUs 和 2 nodes×4 GPUs 两种集群，两种 workload；(2) Component analysis：六种配置下（Occult→+HSC→+HG→+FR+WRR→+DR+WRR→+DR+TAR）的通信/负载指标增量影响；(3) Generalizability：跨 dataset transfer（WikiText-2→MATH→GitHub）的 end-to-end latency。

- 硬件平台是什么，配置是什么。
  - 2 节点，每节点 4× NVIDIA A100-SXM4 GPU (80GB)。节点内 NVLink（每 GPU 12 links，50 GB/s per direction）。节点间 25 Gbps Ethernet。软件：Megablocks + PyTorch 2.5 + Triton 3.1，BFloat16 精度。

- 模型是什么。数据集和bench分别是什么。
  - OLMoE（6.92B, 64 experts, top-8, 16 MoE layers）、DeepSeek-v2-lite-chat（15.7B, 64 experts, top-6, 26 MoE layers）、Qwen3-30B-A3B（30.5B, 128 experts, top-8, 48 MoE layers）。
  - 数据集：WikiText-2-v1, MATH, The Pile (GitHub subset)。指标：All-to-All time, cross-node/intra-node traffic, GPU idle time, per-layer GPU load std, MoE layer time, end-to-end latency。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文声明 "code will be released upon acceptance"，基于 Megablocks 实现。
  - GRACE-MoE 算法 pipeline（offline + online 两阶段）：

**Offline Phase: Profiling → Hierarchical Grouping → Dynamic Replication**

```
# Step 1: Profiling - 构建 expert affinity matrix per layer
for each MoE layer l:
    for each token in calibration_data:
        topk = router[l](h_t)
        for i, j in topk: A[l][i][j] += 1  # co-activation count

# Step 2: Hierarchical Grouping
# Cross-node: fully non-uniform spectral clustering (N groups → N nodes)
C_nodes = SpectralClustering(A, D=N)
# Intra-node: controlled non-uniform with ratio r
#   E = floor(n_experts / D_gpu), delta = max(1, round(E * r))
#   num_min = max(1, E - delta), num_max = E + delta
#   Select r at knee point of (S(r), U(r)) curve:
#     U(r) = sum_{group C} sum_{i,j in C} A[i,j] / sum_{i<j} A[i,j]
#     S(r) = sqrt(1/D * sum (|C_d| - E)^2)
# Algorithm 2: SpectralClustering(A, D) → trim oversized → reassign overflow → balance undersized
C_gpus = ControlledNonUniformGrouping(A, D_gpu, r_opt)

# Step 3: Dynamic Replication (per layer)
W_max = max(sum(token_count for expert in group))
W_mean = mean(group_loads)
rho = W_max / W_mean
n_replica = min(max(1, floor(rho)), n_gpu - 1)
# In heaviest group: rank experts by load, select those with
#   cumulative_load > W_max * n_replica / (1 + n_replica)
# Place replicas on n_replica least-loaded GPUs
```

**Online Phase: HSC + Topology-Aware Routing**

```
# Hierarchical Sparse Communication (HSC, Section 5):
# Stage 1: Cross-node — physically global group, logically sparse
#   GPU aggregates tokens to same dest node → single cross-node send (zero-padded)
# Stage 2: Intra-node — tokens redistributed to expert GPUs via high-BW links
# Cross-node comm overlapped w/ intra-node routing decision computation

# Topology-Aware Routing (Algorithm 4):
for each token routed to expert e (with replicas on replica_gpus):
    if token_gpu_id in replica_gpus:
        selected = token_gpu_id              # local GPU first
    elif any(g in replica_gpus with Node(g) == token_node):
        candidates = [g in replica_gpus | Node(g) == token_node]
        selected = WRR(candidates, predicted_loads)  # intra-node WRR
    else:
        selected = WRR(replica_gpus, predicted_loads) # cross-node fallback

# WRR Load Prediction (Eq. 4):
# W_p = W_max / (n_replica + 1)  # per-instance load after replication
# W'_max = W_max - W_r + W_p
# W'_i = W_i + W_p  (for target replica-hosting GPU i)
# polling_weights ∝ 1 / W'  (inverse proportional)
```

  - 关键数据：最大 speedup 4.66×（OLMoE）、3.73×（DeepSeek）、4.47×（Qwen3）。MoE layer time 降低 up to 80.11%。HSC: All-to-All time −35.19%；HG: 额外 −18.56-24.69%；DR+WRR: GPU idle −19.71%；TAR: All-to-All −9.47% vs WRR, GPU idle 仅 +2.58%。Cross-dataset transfer 最差 +4.52% latency。
