## MoE Load Skewness (MoE负载倾斜度)

术语是什么？
Skewness 是 MoE-GPS 定义的 MoE 推理负载倾斜度量指标：最热门 expert 收到的 token 数除以平均每 expert token 数（均衡分布时）。即 $\text{skewness} = \frac{\#\text{tokens in the most popular expert}}{\#\text{average tokens per expert}}$。Skewness=1 表示 perfect balance，skewness=3 表示最热门 expert 收到 3 倍于平均的 token。Skewness 仅影响 FFN compute 和 communication runtime：bottleneck GPU 的 FFN compute 时间被 scale by skewness；All-to-All 通信时间也被 scale by skewness：$(N-1)·skewness/N^2$（N=GPU 数）。

从算法pipeline角度拆解术语：
Skewness 直接影响 prediction strategy 选择（MoE-GPS Figure 7）：
- Low skewness (1.0-1.5)：Distribution-Only 由于 zero overhead 优势明显
- High skewness (1.5+)：Token-to-Expert 的高 accuracy 优势逐渐超越 predictor overhead
- Skewness 越高 → Distribution-Only 的 estimation error 越大（因冷门 expert 样本少，error 占比高）→ 效果略降
- Skewness 越高 → Token-to-Expert 的 predictor 越容易达到高 accuracy（分布更可预测）→ overhead/accuracy 比更优

实验数据：MMLU skewness=1.39, Alpaca Eval skewness=1.40, SST2 skewness=1.99（Mixtral 8×7B, seq_len=512）。

术语一般如何实现？如何使用？
Skewness 可通过 training data 的 expert activation 统计离线测量。在 MoE-GPS 中，Distribution-Only Prediction 对 skewness 敏感的 error rate 通过 testset 的 empirical probability vs. trainset MLE estimation 的差异计算：$|\hat{p} - p| / (1/E)$。SST2（skewness=1.99）error rate=16%，远高于 MMLU（1.39, error=1.80%），因为高度倾斜导致冷门 expert 训练 token 不足。Skewness 也是 MoE-GPS simulator 选择最优 prediction strategy 的关键输入参数。

涉及论文标题：
- MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing
- MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing
- Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

### MoETuner 补充

MoETuner 从硬件 placement 角度处理 skewness：即使 routing 本身就存在 skewness（某些 expert 确实处理更多 token），通过 ILP 求解将高频和低频 expert 混合分配到同一 GPU，使各 GPU 总 token 处理量趋向均衡（min Σ|T_{c,l} - T̄_l|）。这避免了 Megatron-LM contiguous placement 下 skewness 导致的严重 GPU 计算不平衡（如 layer 14 中 GPU0 处理 64% token）。在 Mixtral-8x7B 上单节点 token processing tail latency 减少 36%。

### Orders in Chaos 补充

本论文对 4 个 large-scale MoE 模型 (235B-1000B) 的 >24,000 requests 进行了系统性的 expert activation skewness profiling，发现了以下新特征：
- **量级差异**：部分 expert 被激活的频率是平均值的 16 倍以上（Llama 4 layer 7），远超早期小模型如 Mixtral 的 skewness。
- **Task/Language 特异性**：不同 MMLU subject（biology, history, math 等共 57 个）的 top-10 popular experts 既有共性（horizontal bright lines 表示跨 subject 共同热门 experts），也有明显差异。同一问题使用英文 vs 中文时，即使内容相同，仅 ~5-6 个 experts 保持 popular，且只有 2 个与英文 MMLU 的 top experts 重叠。这揭示了 language 对 expert selection 的显著影响。
- **系统影响**：skewness 导致的 workload imbalance 在 wafer-scale GPU 上尤为严重——热门 expert 所在 die 可能处理 16× 于平均的请求量，而冷门 expert 所在 die 基本空闲。论文提出 Popular Expert Decentralization (Insight 4) 策略——duplicate/replicate 热门 experts 到多个 compute unit 以均衡负载。
- **与 Expert Co-activation 的关系**：top 10% expert pairs 占 60-80% 总激活量（DeepSeek V3, Qwen3），说明 skewness 不仅体现在 single expert 层面，也体现在 expert pair 层面。

---
