## HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - HOBBIT 提出 **Mixed Precision Expert Offloading** 算法：利用 MoE 中不同 expert 的重要性差异，对不重要的 cache-miss expert 动态替换为低精度（量化）版本以加速加载，同时保持模型精度。核心算法组件：
    1. **Expert 重要性动态估计**：使用 ||G(x)_{e_i}||（gating output 的 magnitude）作为 expert 重要性的计算高效代理，实验验证与 ||G(x)_{e_i}E_{e_i}(x)||（expert output magnitude）的 Pearson 相关系数为 0.99。
    2. **Unimportance Degree Score**：s_{e_i} = Σ_{j=0}^{i-1} ||G(x)_{e_j}||（已归一化）。对所有 top-K experts 按 ||G(x)|| 降序排列后计算累积分数。
    3. **双阈值精度决策**：T1=0.6（高精度阈值），T2=0.9（跳过阈值）。s ≤ 0.6 → 高精度 (FP16/INT8)；0.6 < s ≤ 0.9 → 低精度 (INT4/INT2)，加载量减少 4×；s > 0.9 → 跳过该 expert。分布比例约 67%/30%/3%（以 Mixtral-8x7B 为例）。
    4. **混合精度预取**：即使预测精度低时，低精度 expert 预取加载的惩罚远低于高精度（图 9），使预取在任何精度下都有正向收益。
    5. **多维混合精度缓存淘汰**：LHU (Least High Precision Frequently Used) 策略 + LRU + LFU + FLD 加权组合，H_t 记录高精度使用频次，最小化混合精度 miss penalty（高精度 miss 代价为 C，低精度为 B_l/B_h · C）。
  - 实验比较：6 个 baseline（Transformers, DeepSpeed-Inference, Llama.cpp, MoE-Offloading, MoE-Infinity, Fiddler）。精度验证使用 GSM8K（数学推理 accuracy）和 TruthfulQA（truth/info 分数）。消融实验分别验证动态加载（1.19-1.57× speedup）、自适应预取（~5% decoding speedup）、多维缓存（4.69%-8.68% miss penalty reduction vs LRU）。

- 硬件平台是什么，配置是什么。
  - **RTX 4090**（24GB GPU, 256GB CPU, 64 cores, PCIe 4.0 32GB/s）作为 edge server。
  - **Jetson AGX Orin**（32GB unified memory, 12 CPU cores）作为 end device。
  - 存储：Samsung NVMe SSD 980 PRO (7,000 MB/s read, ~3,000 MB/s 实测)。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - Mixtral-8x7B：45B 总参数，32 layers，8 experts/layer，top-2 routing，14B 激活参数/token，权重 87GB（expert 占 96%）。
    - Phi-MoE：42B 总参数，32 layers，16 experts/layer，top-2 routing，6.6B 激活参数/token，权重 78GB（expert 占 96%）。
  - **精度配置**：
    - FP16 高精度 + INT4 低精度（RTX 4090 实验）
    - INT8 高精度 + INT2 低精度（Jetson Orin 实验）
  - **数据集与 benchmark**：
    - 速度测试：Alpaca 数据集 60 高质量样本（一半 input length=16，一半 input length=128），四种 I/O 组合 [16,32]/[16,128]/[128,32]/[128,128]。
    - 精度测试：GSM8K（数学推理，top-1 accuracy）、TruthfulQA（truth 和 info 分数）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未公开 HOBBIT 完整代码；基于 Llama.cpp (https://github.com/ggerganov/llama.cpp) 修改。
  - 混合精度 Expert Offloading 算法伪代码：

```python
# === HOBBIT Dynamic Mixed Precision Expert Loading ===
# 输入: hidden state x, gating function G (Linear + Softmax + TopK)
# 输出: MoE layer output y

# 超参数
T1 = 0.6  # 高/低精度分界阈值
T2 = 0.9  # 低精度/跳过分界阈值

def moe_layer_forward(x, expert_cache, expert_storage):
    # Step 1: Gating（GPU 计算）
    gate_logits = G.linear(x)           # [1, num_experts]
    gate_probs = softmax(gate_logits)   # [1, num_experts]
    topk_vals, topk_ids = topk(gate_probs, K=K)
    
    # Step 2: 归一化 gating weights 并按降序排序
    gate_norm = topk_vals / topk_vals.sum()  # 归一化
    sorted_idx = argsort(gate_norm, descending=True)
    
    # Step 3: 计算 unimportance degree score
    scores = zeros(K)
    cumulative = 0.0
    for i in range(K):
        e_i = sorted_idx[i]
        scores[e_i] = cumulative
        cumulative += gate_norm[e_i]
    
    # Step 4: 动态决定每个 expert 的精度
    load_tasks = []
    for i, e_i in enumerate(topk_ids):
        s = scores[i]
        if s <= T1:
            precision = "high"   # FP16 或 INT8
            skip = False
        elif s <= T2:
            precision = "low"    # INT4 或 INT2
            skip = False
        else:
            skip = True
        
        if not skip and e_i not in expert_cache:
            load_tasks.append((e_i, precision))
    
    # Step 5: 异步加载缺失的 experts
    for e_i, precision in load_tasks:
        weight = expert_storage.read(e_i, precision)  # PCIe/SSD read
        expert_cache.insert(e_i, weight, precision)   # 由 Cache Manager 管理
    
    # Step 6: Expert FFN 计算（GPU）
    y = zeros_like(x)
    for i, e_i in enumerate(topk_ids):
        if not skip_for(e_i):
            weight = expert_cache.get(e_i)
            # FFN: y_e = W_o · (SiLU(W_g · x) ⊙ (W_p · x))
            gate_out = silu(matmul(x, weight.W_g))  # [1, d_ffn]
            up_out = matmul(x, weight.W_p)           # [1, d_ffn]
            expert_out = matmul(gate_out * up_out, weight.W_o)  # [1, d_model]
            y += gate_norm[i] * expert_out
    
    return y

# === Adaptive Expert Prefetching ===
def predict_experts(x, stacking_computer, num_layers_ahead=3):
    # Stacking Computer: 一次性矩阵乘计算后续层 gating
    # gate_weights_stacked: [num_layers_ahead, d_model, num_experts]
    gate_logits_all = matmul(x, gate_weights_stacked)  # [num_layers_ahead, num_experts]
    topk_ids_all = topk(gate_logits_all, K=K, dim=-1)   # [num_layers_ahead, K]
    return topk_ids_all  # 后续层预取 expert IDs

# === Multidimensional Cache Priority ===
def compute_cache_priority(expert_t, current_layer, current_token, records):
    T = current_token
    l_n = total_layers
    l_i = current_layer
    l_t = expert_t.layer_id
    
    p_lru = records.R[t] / T      # 最近使用时间
    p_lfu = records.F[t] / T      # 使用频率
    p_lhu = records.H[t] / T      # 高精度使用频率
    p_fld = 1 - ((l_t - l_i + l_n) % l_n) / l_n  # 层距离
    
    p = w_lru*p_lru + w_lfu*p_lfu + w_lhu*p_lhu + w_fld*p_fld
    return p  # 越高优先级越高，evict 最低优先级的 expert
```

  关键设计原理：
  - Pearson 相关系数 0.99 验证了 ||G(x)|| 可作为 expert 贡献度的近似，避免实际计算 E(x)。
  - 双阈值设计基于 gating 分布的统计分析：top-1 expert 始终得 0 分（保持高精度），约 67% 选高精度、30% 选低精度、3% 跳过。
  - 总 expert cache miss penalty 相比 LFU 降低 2.13%-4.19%，相比 LRU 降低 4.69%-8.68%。
