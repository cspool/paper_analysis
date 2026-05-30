## PowerInfer

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

PowerInfer 是 Song et al. (2023) 提出的 LLM 推理加速系统，专门利用 LLM FFN 层的激活稀疏度在消费级 GPU 上高效部署大模型。核心设计是将 FFN 神经元按激活频率分为"热神经元"（高频激活，占 ~20% 但承担 ~80% 计算）和"冷神经元"（低频激活，占 ~80% 但仅承担 ~20% 计算），热神经元参数存储在 GPU 内存中并用 GPU Tensor Core 计算，冷神经元参数卸载到 CPU 内存中用 CPU 计算。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

PowerInfer 推理流程：
```
# 离线阶段
for each FFN layer l:
    for each neuron i in layer l:
        profile on calibration data  # 统计激活频率
    classify neurons as hot (高频) or cold (低频)
    hot_neuron_params[l] → GPU memory
    cold_neuron_params[l] → CPU memory

# 在线推理阶段（per token）
for each Transformer layer:
    # Self-Attention (所有参数在GPU)
    x = SelfAttention(x)
    
    # FFN with sparse computation
    hot_neurons = GPU_predictor.predict(x)      # 预测被激活的热神经元
    cold_neurons = CPU_predictor.predict(x)     # 预测被激活的冷神经元
    
    GPU: compute hot neurons Σ n_i_hot          # GPU Tensor Core 高吞吐
    CPU: compute cold neurons Σ n_i_cold        # CPU 向量计算
    
    output = GPU_partial + transfer(CPU_partial)  # CPU结果传回GPU
```

关键设计：
1. **离线 Profiler**：统计每层每个神经元的历史激活频率，基于幂律分布（~20/80 规则）分类热/冷
2. **在线激活预测器**：基于 token-wise similarity（相邻 token 激活模式 >90% 相似）和 layer-wise correlation 预测当前输入可能激活的神经元
3. **混合计算**：GPU 处理热神经元 GEMM（高计算强度），CPU 处理冷神经元 GEMM（避免 GPU 内存瓶颈和 PCIe 传输开销）
4. **定位于消费级 GPU 场景**：相较于服务器级 GPU 方案（如 Deja Vu），PowerInfer 使 LLM 推理可在单张消费级 GPU + 主存上部署

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现基于 llama.cpp（https://github.com/ggerganov/llama.cpp），使用 CUDA 编译。在 Sparsing Law 论文中，PowerInfer 用于验证 2.4B ReLU 模型（93.52% 稀疏度）的加速效果：PowerInfer 解码速度 41.79 tok/s，llama.cpp 密集 FFN 解码速度 10.23 tok/s，4.1× 加速。测试环境：104 CPUs + 1× NVIDIA A800 GPU，100 个测试 prompt（C4 数据集，各 5 个 prefix token）。注意：llama.cpp 不支持 ReLU 激活函数（仅 SiLU），但由于 FLOPS 相同不影响加速对比。

局限性：CPU 侧内存带宽（~89.6 GB/s）远低于 GPU HBM 带宽，CPU 侧计算可能成为瓶颈；依赖激活稀疏度的 token-wise similarity 和 layer-wise correlation 局部性。后续工作 PowerInfer-2 (Xue et al., 2024) 进一步将推理加速扩展到智能手机平台。

涉及论文标题：
- Sparsing Law Towards Large Language Models with Greater Activation Sparsity
