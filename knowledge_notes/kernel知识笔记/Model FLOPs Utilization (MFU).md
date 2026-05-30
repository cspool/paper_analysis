## Model FLOPs Utilization (MFU)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Model FLOPs Utilization (MFU) 是衡量大规模模型训练中GPU计算效率的指标，由PaLM论文(Chowdhery et al., 2022)引入，Korthikanti et al.(2022)在Megatron-LM中使用。定义：MFU = (单次forward+backward的理论FLOPs / 每次迭代时间) / (GPU理论峰值FLOPs/s × GPU数量)。分子是实际达到的FLOPs/s（模型理论FLOPs除以实测迭代时间），分母是硬件理论峰值FLOPs/s总和。MFU考量了模型计算（而非全部操作如attention的带宽瓶颈部分）对硬件的利用率。相比于传统Hardware FLOPs Utilization(HFU)，MFU通常更高，因为模型FLOPs仅计入了矩阵乘法等主力计算。

从kernel调度角度拆解术语：
```
MFU计算流程：
1. 计算模型理论FLOPs = f(params, seq_len, hidden_dim, num_layers)
   - Transformer: 主要来自QKV投影、attention MatMul(忽略softmax)、MLP MatMul
   - Mamba-2: 主要来自input projection、SSD scan的MatMul、output projection
2. 测量迭代时间t_iter (forward+backward)
3. 查找GPU理论峰值: H100 SXM BF16 = 989.8 TFLOPS/GPU
4. MFU = (model_FLOPs / t_iter) / (989.8e12 * num_GPUs)
```

论文结果：8B Mamba-2-Hybrid在1024 H100 GPUs上MFU=29.9%，接近同规模Transformer的30.7%。这表明Hybrid模型在Megatron-LM中的实现效率与成熟Transformer实现相当。

术语一般如何实现？如何使用？
Megatron-LM和PaLM论文提供MFU计算脚本。通常TP/DP配置、micro batch size、activation checkpointing等因素影响MFU。论文中TP=4, DP=256, micro_batch=4, global_batch=1024。达到30% MFU对8B参数规模的大模型训练是合理的。

涉及论文标题：
- An_Empirical_Study_of_Mamba-based_Language_Models

---
